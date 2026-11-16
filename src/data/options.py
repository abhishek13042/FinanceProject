"""Daily option-chain features per stock-day, from the normalised F&O bhavcopy.

Everything is computed from the chain of day t alone (plus day t-1 for changes), so a
feature on day t uses only information published at that day's close - the same timing as
the price features. Implied volatility uses Black-76 on the *forward* (the same-expiry
futures close). That avoids two problems: prices.csv holds dividend/split-adjusted closes
that do not match the strikes traded on the day, and the forward already carries the
interest/dividend carry, so no rate or yield assumption leaks into the vol.

Raw features (before scaling):
  atm_iv        mean of the ATM call and put IV, near expiry
  atm_iv_chg    atm_iv(t) - atm_iv(t-1)
  iv_skew       IV(put near 0.95F) - IV(call near 1.05F), near expiry
  iv_term       ATM IV next expiry - ATM IV near expiry
  straddle      (ATM call + ATM put) / F: the market's priced move to expiry
  pcr_oi        log(put OI / call OI), all expiries
  pcr_vol       log(put contracts / call contracts), all expiries
  call_oi_chg   call OI change / previous call OI
  put_oi_chg    put OI change / previous put OI
  maxpain_gap   max-pain strike / F - 1, near expiry
  maxoi_call    strike with the largest call OI / F - 1 (resistance)
  maxoi_put     strike with the largest put OI / F - 1 (support)
  fut_carry     annualised log(F_next / F_near)
  fut_oi_chg    futures OI change / previous futures OI
  opt_activity  log1p(option contracts traded)
  has_opt       1 when the stock had a usable option chain that day
"""
from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

RATE = 0.065          # INR risk-free proxy; only discounts the premium, a small effect
MIN_DAYS = 3          # skip an expiry this close; its options are mostly intrinsic + noise

OPT_FEATS = ["atm_iv", "atm_iv_chg", "iv_skew", "iv_term", "straddle", "pcr_oi", "pcr_vol",
             "call_oi_chg", "put_oi_chg", "maxpain_gap", "maxoi_call", "maxoi_put",
             "fut_carry", "fut_oi_chg", "opt_activity"]
OPT_COLS = OPT_FEATS + ["has_opt"]


# ---------------------------------------------------------------- Black-76 implied vol
def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black76(F: float, K: float, T: float, sig: float, call: bool, r: float = RATE) -> float:
    if sig <= 0 or T <= 0:
        return math.exp(-r * T) * max((F - K) if call else (K - F), 0.0)
    v = sig * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * v * v) / v
    d2 = d1 - v
    if call:
        return math.exp(-r * T) * (F * _ncdf(d1) - K * _ncdf(d2))
    return math.exp(-r * T) * (K * _ncdf(-d2) - F * _ncdf(-d1))


def implied_vol(price: float, F: float, K: float, T: float, call: bool) -> float:
    """Bisection on [1e-3, 5]. NaN when the price is below intrinsic or above the bound."""
    if not (price > 0 and F > 0 and K > 0 and T > 0):
        return np.nan
    lo, hi = 1e-3, 5.0
    if price < black76(F, K, T, lo, call) or price > black76(F, K, T, hi, call):
        return np.nan
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if black76(F, K, T, mid, call) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- chain helpers
def pick_expiries(date: pd.Timestamp, expiries) -> tuple:
    """Nearest expiry at least MIN_DAYS away, and the one after it."""
    ex = sorted(e for e in set(expiries) if (e - date).days >= MIN_DAYS)
    return (ex[0] if ex else None, ex[1] if len(ex) > 1 else None)


def forward(fut: pd.DataFrame, chain: pd.DataFrame, expiry, T: float) -> float:
    """Same-expiry futures close; else put-call parity at the most-traded strike."""
    f = fut.loc[fut["expiry"] == expiry, "close"]
    if len(f) and f.iloc[0] > 0:
        return float(f.iloc[0])
    pv = chain.pivot_table(index="strike", columns="opt_type", values="close", aggfunc="first")
    if not {"CE", "PE"} <= set(pv.columns):
        return np.nan
    pv = pv.dropna()
    if pv.empty:
        return np.nan
    k = (pv["CE"] - pv["PE"]).abs().idxmin()
    return float(k + (pv.loc[k, "CE"] - pv.loc[k, "PE"]) * math.exp(RATE * T))


def atm_row(chain: pd.DataFrame, F: float):
    """ATM strike: nearest to F among strikes where both legs exist, preferring traded ones."""
    pv = chain.pivot_table(index="strike", columns="opt_type",
                           values=["close", "contracts"], aggfunc="first")
    if ("close", "CE") not in pv.columns or ("close", "PE") not in pv.columns:
        return None
    both = pv.dropna(subset=[("close", "CE"), ("close", "PE")])
    if both.empty:
        return None
    traded = both[(both[("contracts", "CE")] > 0) & (both[("contracts", "PE")] > 0)]
    pool = traded if len(traded) else both
    k = pool.index[np.argmin(np.abs(pool.index.to_numpy() - F))]
    return k, float(pool.loc[k, ("close", "CE")]), float(pool.loc[k, ("close", "PE")])


def max_pain(chain: pd.DataFrame) -> float:
    c = chain[chain["opt_type"] == "CE"].groupby("strike")["oi"].sum()
    p = chain[chain["opt_type"] == "PE"].groupby("strike")["oi"].sum()
    ks = np.union1d(c.index, p.index)
    if len(ks) == 0:
        return np.nan
    c, p = c.reindex(ks, fill_value=0).to_numpy(), p.reindex(ks, fill_value=0).to_numpy()
    K = ks[:, None]                              # candidate settlement prices (rows)
    pain = (np.maximum(K - ks[None, :], 0) * c).sum(1) + (np.maximum(ks[None, :] - K, 0) * p).sum(1)
    return float(ks[np.argmin(pain)])


def _ratio(num: float, den: float) -> float:
    return num / den if den > 0 else np.nan


def day_features(date: pd.Timestamp, g: pd.DataFrame) -> dict:
    opt, fut = g[g["inst"] == "OPT"], g[g["inst"] == "FUT"]
    out = {k: np.nan for k in OPT_FEATS}
    out["has_opt"] = 0.0
    # futures-only features exist even when there are no options
    if len(fut):
        fn, fx = pick_expiries(date, fut["expiry"])
        if fn is not None and fx is not None:
            a = fut.loc[fut["expiry"] == fn, "close"].iloc[0]
            b = fut.loc[fut["expiry"] == fx, "close"].iloc[0]
            dt = (fx - fn).days / 365
            if a > 0 and b > 0 and dt > 0:
                out["fut_carry"] = math.log(b / a) / dt
        prev = fut["oi"].sum() - fut["chg_oi"].sum()
        out["fut_oi_chg"] = _ratio(fut["chg_oi"].sum(), prev)
    if opt.empty:
        return out

    near, nxt = pick_expiries(date, opt["expiry"])
    if near is None:
        return out
    chain = opt[opt["expiry"] == near]
    T = max((near - date).days, 1) / 365
    F = forward(fut, chain, near, T)
    if not F > 0:
        return out
    atm = atm_row(chain, F)
    if atm is None:
        return out
    k, c, p = atm
    ivc, ivp = implied_vol(c, F, k, T, True), implied_vol(p, F, k, T, False)
    out["atm_iv"] = np.nanmean([ivc, ivp]) if not (np.isnan(ivc) and np.isnan(ivp)) else np.nan
    out["straddle"] = (c + p) / F
    out["has_opt"] = 1.0

    # skew: OTM put near 0.95F vs OTM call near 1.05F (traded strikes preferred)
    def wing(typ, target, call):
        w = chain[(chain["opt_type"] == typ) & (chain["close"] > 0)]
        tw = w[w["contracts"] > 0]
        w = tw if len(tw) else w
        if w.empty:
            return np.nan
        r = w.iloc[np.argmin(np.abs(w["strike"].to_numpy() - target))]
        return implied_vol(float(r["close"]), F, float(r["strike"]), T, call)
    out["iv_skew"] = wing("PE", 0.95 * F, False) - wing("CE", 1.05 * F, True)

    if nxt is not None:
        ch2 = opt[opt["expiry"] == nxt]
        T2 = max((nxt - date).days, 1) / 365
        F2 = forward(fut, ch2, nxt, T2)
        a2 = atm_row(ch2, F2) if F2 > 0 else None
        if a2 is not None:
            iv2 = np.nanmean([implied_vol(a2[1], F2, a2[0], T2, True),
                              implied_vol(a2[2], F2, a2[0], T2, False)])
            out["iv_term"] = iv2 - out["atm_iv"]

    ce, pe = opt[opt["opt_type"] == "CE"], opt[opt["opt_type"] == "PE"]
    if ce["oi"].sum() > 0 and pe["oi"].sum() > 0:
        out["pcr_oi"] = math.log(pe["oi"].sum() / ce["oi"].sum())
    out["pcr_vol"] = math.log((pe["contracts"].sum() + 1) / (ce["contracts"].sum() + 1))
    out["call_oi_chg"] = _ratio(ce["chg_oi"].sum(), ce["oi"].sum() - ce["chg_oi"].sum())
    out["put_oi_chg"] = _ratio(pe["chg_oi"].sum(), pe["oi"].sum() - pe["chg_oi"].sum())
    out["maxpain_gap"] = max_pain(chain) / F - 1
    cn, pn = chain[chain["opt_type"] == "CE"], chain[chain["opt_type"] == "PE"]
    if cn["oi"].sum() > 0:
        out["maxoi_call"] = cn.loc[cn["oi"].idxmax(), "strike"] / F - 1
    if pn["oi"].sum() > 0:
        out["maxoi_put"] = pn.loc[pn["oi"].idxmax(), "strike"] / F - 1
    out["opt_activity"] = math.log1p(opt["contracts"].sum())
    return out


def _symbol_features(args) -> pd.DataFrame:
    sym, g = args
    rows = []
    for date, gd in g.groupby("date"):
        r = day_features(date, gd)
        r["date"], r["symbol"] = date, sym
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("date")
    df["atm_iv_chg"] = df["atm_iv"].diff()
    return df


def build_features(options: pd.DataFrame, workers: int = 8) -> pd.DataFrame:
    """Raw (unscaled) features, one row per (date, symbol) present in the bhavcopy."""
    options = options.copy()
    options["date"] = pd.to_datetime(options["date"])
    options["expiry"] = pd.to_datetime(options["expiry"])
    groups = list(options.groupby("symbol"))
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(_symbol_features, groups))
    return pd.concat(parts, ignore_index=True)[["date", "symbol"] + OPT_COLS]


def attach(prices: pd.DataFrame, feats: pd.DataFrame, train_end: str,
           clip: float = 5.0) -> pd.DataFrame:
    """Join onto the featurised price frame and scale.

    Scaling is pooled across stocks (these are ratios, comparable between names) and uses
    training-period rows only. Stocks without options on a day get 0 everywhere (= the
    training mean) and has_opt = 0, so the model can tell "neutral" from "absent".
    """
    f = feats.copy()
    f["symbol"] = f["symbol"].where(f["symbol"].str.endswith((".NS", ".BO")), f["symbol"] + ".NS")
    f["date"] = pd.to_datetime(f["date"])
    df = prices.drop(columns=[c for c in OPT_COLS if c in prices.columns]).merge(
        f, on=["date", "symbol"], how="left")
    df[OPT_FEATS] = df[OPT_FEATS].replace([np.inf, -np.inf], np.nan)
    tr = df[(df["date"] <= pd.Timestamp(train_end)) & (df["has_opt"] == 1)]
    # Winsorise at training 1st/99th percentiles first: ratios like OI change explode when
    # the previous OI is tiny, and one such day would otherwise set the std for everyone.
    lo, hi = tr[OPT_FEATS].quantile(0.01), tr[OPT_FEATS].quantile(0.99)
    df[OPT_FEATS] = df[OPT_FEATS].clip(lo, hi, axis=1)
    tr = df[(df["date"] <= pd.Timestamp(train_end)) & (df["has_opt"] == 1)]
    mu, sd = tr[OPT_FEATS].mean(), tr[OPT_FEATS].std().replace(0, 1.0)
    df[OPT_FEATS] = ((df[OPT_FEATS] - mu) / sd).clip(-clip, clip).fillna(0.0)
    df["has_opt"] = df["has_opt"].fillna(0.0)
    return df


if __name__ == "__main__":
    import argparse, time
    ap = argparse.ArgumentParser(description="Build raw option features")
    ap.add_argument("--options", default="data/interim/options.csv.gz")
    ap.add_argument("--out", default="data/interim/option_features.csv")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    opts = pd.read_csv(a.options, parse_dates=["date", "expiry"])
    feats = build_features(opts, a.workers)
    feats.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(feats):,} rows, has_opt {feats['has_opt'].mean():.1%}, "
          f"{time.time()-t0:.0f}s")
    print(feats[OPT_FEATS].describe().T[["mean", "std", "min", "max"]].round(3).to_string())
