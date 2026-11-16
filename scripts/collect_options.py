"""Collect daily NSE F&O bhavcopies and normalise stock options + futures for the universe.

Two NSE archive formats exist:
  old (to Jul 2024): content/historical/DERIVATIVES/YYYY/MON/foDDMONYYYYbhav.csv.zip
                     INSTRUMENT in {OPTSTK, FUTSTK}; no lot size, no underlying price.
  UDiFF (Jul 2024+): content/fo/BhavCopy_NSE_FO_0_0_0_YYYYMMDD_F_0000.csv.zip
                     FinInstrmTp in {STO, STF}; has NewBrdLotQty and UndrlygPric.

Units are the same in both: volume = contracts, OI = shares, turnover = notional.
The old format's lot size is recovered from futures turnover / (contracts * price).

Usage:
  python scripts/collect_options.py download [--start 2018-01-01] [--end 2025-12-31]
  python scripts/collect_options.py parse
Downloads are resume-safe (existing zips are skipped) and rate-limited.
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw/fo_bhav"
OUT = ROOT / "data/interim/options.csv.gz"
UNIVERSE = ROOT / "data/raw/universe50.csv"
PRICES = ROOT / "data/interim/prices.csv"

UDIFF_FROM = pd.Timestamp("2024-07-08")   # NSE switched the F&O bhavcopy to UDiFF here
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# Bhavcopy symbol -> universe symbol. Tata Motors PV (TMPV) traded as TATAMOTORS until the
# Oct 2025 demerger; the price history in prices.csv is continuous under TMPV.
ALIASES = {"TATAMOTORS": "TMPV"}

COLS = ["date", "symbol", "inst", "expiry", "strike", "opt_type", "open", "high", "low",
        "close", "settle", "contracts", "notional", "oi", "chg_oi", "lot", "underlying"]


def old_url(d: pd.Timestamp) -> str:
    mon = d.strftime("%b").upper()
    return (f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/"
            f"{d.year}/{mon}/fo{d.day:02d}{mon}{d.year}bhav.csv.zip")


def new_url(d: pd.Timestamp) -> str:
    return (f"https://nsearchives.nseindia.com/content/fo/"
            f"BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip")


def fetch(url: str, tries: int = 3) -> bytes | None:
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            return data if data[:2] == b"PK" else None
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 * (k + 1))
        except Exception:
            time.sleep(2 * (k + 1))
    return None


def trading_days(start: str, end: str) -> list[pd.Timestamp]:
    d = pd.to_datetime(pd.read_csv(PRICES, usecols=["date"])["date"].unique())
    d = sorted(x for x in d if pd.Timestamp(start) <= x <= pd.Timestamp(end))
    return d


def download(start: str, end: str, sleep: float) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    days = trading_days(start, end)
    missing_log = RAW_DIR / "missing.txt"
    known_missing = set(missing_log.read_text().split()) if missing_log.exists() else set()
    got = skipped = miss = 0
    t0 = time.time()
    for i, d in enumerate(days):
        dst = RAW_DIR / f"{d:%Y%m%d}.zip"
        if dst.exists() or f"{d:%Y%m%d}" in known_missing:
            skipped += 1
            continue
        urls = [new_url(d), old_url(d)] if d >= UDIFF_FROM else [old_url(d), new_url(d)]
        data = None
        for u in urls:
            data = fetch(u)
            time.sleep(sleep)
            if data:
                break
        if data:
            tmp = dst.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.rename(dst)
            got += 1
        else:
            miss += 1
            with missing_log.open("a") as f:
                f.write(f"{d:%Y%m%d}\n")
        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(days)}] got={got} skipped={skipped} missing={miss} "
                  f"{time.time()-t0:.0f}s", flush=True)
    print(f"done: got={got} skipped={skipped} missing={miss} of {len(days)}", flush=True)


def _read_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        return pd.read_csv(io.BytesIO(z.read(name)), low_memory=False)


def _norm_old(df: pd.DataFrame, d: pd.Timestamp, universe: set[str]) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    df = df[df["INSTRUMENT"].isin(["OPTSTK", "FUTSTK"])].copy()
    df["SYMBOL"] = df["SYMBOL"].str.strip().replace(ALIASES)
    df = df[df["SYMBOL"].isin(universe)]
    out = pd.DataFrame({
        "date": d, "symbol": df["SYMBOL"],
        "inst": np.where(df["INSTRUMENT"] == "OPTSTK", "OPT", "FUT"),
        "expiry": pd.to_datetime(df["EXPIRY_DT"], format="%d-%b-%Y"),
        "strike": df["STRIKE_PR"].astype(float),
        "opt_type": df["OPTION_TYP"].str.strip(),
        "open": df["OPEN"], "high": df["HIGH"], "low": df["LOW"], "close": df["CLOSE"],
        "settle": df["SETTLE_PR"], "contracts": df["CONTRACTS"],
        "notional": df["VAL_INLAKH"].astype(float) * 1e5,
        "oi": df["OPEN_INT"], "chg_oi": df["CHG_IN_OI"],
        "lot": np.nan, "underlying": np.nan,
    })
    return out


def _norm_new(df: pd.DataFrame, d: pd.Timestamp, universe: set[str]) -> pd.DataFrame:
    df = df[df["FinInstrmTp"].isin(["STO", "STF"])].copy()
    df["TckrSymb"] = df["TckrSymb"].str.strip().replace(ALIASES)
    df = df[df["TckrSymb"].isin(universe)]
    out = pd.DataFrame({
        "date": d, "symbol": df["TckrSymb"],
        "inst": np.where(df["FinInstrmTp"] == "STO", "OPT", "FUT"),
        "expiry": pd.to_datetime(df["XpryDt"]),
        "strike": df["StrkPric"].fillna(0).astype(float),
        "opt_type": df["OptnTp"].fillna("XX").str.strip(),
        "open": df["OpnPric"], "high": df["HghPric"], "low": df["LwPric"],
        "close": df["ClsPric"], "settle": df["SttlmPric"],
        "contracts": df["TtlTradgVol"], "notional": df["TtlTrfVal"].astype(float),
        "oi": df["OpnIntrst"], "chg_oi": df["ChngInOpnIntrst"],
        "lot": df["NewBrdLotQty"].astype(float), "underlying": df["UndrlygPric"],
    })
    return out


def _infer_lots(df: pd.DataFrame) -> pd.Series:
    """Lot size per (date, symbol, expiry) for the old format.

    OI and its change are always whole lots, so their GCD over a contract month is a multiple
    of the lot. Futures turnover / (contracts * price) gives an estimate (+-1%) that picks the
    right multiple. Per expiry because NSE lot revisions leave running months on the old lot.
    """
    def gcd_of(s: pd.Series) -> int:
        v = np.abs(s.to_numpy(dtype=np.int64))
        v = v[v > 0]
        return int(np.gcd.reduce(v)) if len(v) else 0

    key = ["date", "symbol", "expiry"]
    vals = pd.concat([df[key + ["oi"]].rename(columns={"oi": "v"}),
                      df[key + ["chg_oi"]].rename(columns={"chg_oi": "v"})])
    g = vals.groupby(key)["v"].apply(gcd_of)

    fut = df[(df["inst"] == "FUT") & (df["contracts"] > 0)]
    mid = (fut["open"] + fut["high"] + fut["low"] + fut["close"]) / 4
    est = (fut["notional"] / (fut["contracts"] * mid)).groupby(
        [fut["date"], fut["symbol"]]).median()

    out = g.reset_index(name="g").merge(est.rename("est").reset_index(),
                                        on=["date", "symbol"], how="left")
    out["est"] = out["est"].fillna(out["g"])
    k = np.maximum(np.round(out["g"] / out["est"]), 1)
    ok = (out["g"] > 0) & (out["g"] % k == 0) & (np.abs(out["g"] / k / out["est"] - 1) < 0.05)
    out["lot"] = np.where(ok, out["g"] / k, np.nan)

    # Unconfirmed (no OI to snap to): the confirmed lot of the same day nearest the estimate,
    # else the symbol's confirmed lot on the nearest earlier (then later) day.
    conf = out[ok]
    same_day = conf.groupby(["date", "symbol"])["lot"].apply(lambda s: np.unique(s))
    def pick(r):
        c = same_day.get((r["date"], r["symbol"]))
        return c[np.argmin(np.abs(c - r["est"]))] if c is not None else np.nan
    bad = ~ok
    out.loc[bad, "lot"] = out[bad].apply(pick, axis=1)
    day_lot = conf.groupby(["symbol", "date"])["lot"].agg(lambda s: s.mode().iloc[0])
    fill = day_lot.groupby(level=0).apply(lambda s: s.droplevel(0))
    out = out.sort_values(["symbol", "date"])
    for sym, idx in out[out["lot"].isna()].groupby("symbol").groups.items():
        ref = day_lot.get(sym) if sym in day_lot.index.get_level_values(0) else None
        if ref is None or ref.empty:
            out.loc[idx, "lot"] = np.round(out.loc[idx, "est"])
            continue
        pos = ref.index.searchsorted(out.loc[idx, "date"], side="right") - 1
        out.loc[idx, "lot"] = ref.to_numpy()[np.clip(pos, 0, len(ref) - 1)]
    return out.set_index(key)["lot"]


def parse() -> None:
    universe = set(pd.read_csv(UNIVERSE)["symbol"])
    zips = sorted(RAW_DIR.glob("*.zip"))
    parts = []
    for i, p in enumerate(zips):
        d = pd.Timestamp(p.stem)
        try:
            raw = _read_zip(p)
        except Exception as e:  # noqa: BLE001
            print(f"skip {p.name}: {e}", file=sys.stderr)
            continue
        norm = _norm_new if "FinInstrmTp" in raw.columns else _norm_old
        parts.append(norm(raw, d, universe))
        if (i + 1) % 200 == 0:
            print(f"parsed {i+1}/{len(zips)}", flush=True)
    df = pd.concat(parts, ignore_index=True)

    # Old-format lot sizes: estimate from futures, then snap to the lot seen in OI
    # (OI is always a whole number of lots), and carry within the symbol.
    est = _infer_lots(df[df["lot"].isna()]).rename("lot_est").reset_index()
    df = df.merge(est, on=["date", "symbol", "expiry"], how="left")
    df["lot"] = df["lot"].fillna(df["lot_est"].astype(float))
    df = df.drop(columns="lot_est")
    df["lot"] = df.groupby(["symbol", "expiry"])["lot"].transform(lambda s: s.ffill().bfill())

    df = df.sort_values(["date", "symbol", "inst", "expiry", "opt_type", "strike"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df[COLS].to_csv(OUT, index=False, compression="gzip")
    n_opt = (df["inst"] == "OPT").sum()
    print(f"wrote {OUT.relative_to(ROOT)}: {len(df):,} rows ({n_opt:,} options), "
          f"{df['date'].nunique()} days, {df['symbol'].nunique()} symbols")
    cov = df[df["inst"] == "OPT"].groupby("symbol")["date"].nunique().sort_values()
    print("option-days per symbol (lowest 10):")
    print(cov.head(10).to_string())
    missing = sorted(universe - set(cov.index))
    print("no options at all:", missing)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["download", "parse"])
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--sleep", type=float, default=0.4, help="seconds between requests")
    a = ap.parse_args()
    if a.mode == "download":
        download(a.start, a.end, a.sleep)
    else:
        parse()


if __name__ == "__main__":
    main()
