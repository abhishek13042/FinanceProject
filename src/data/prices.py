"""NSE price ingestion and the paper's Section 4.2 preprocessing, adapted for India.

India-specific deviations from the paper:
  * yfinance ".NS" tickers; auto_adjust=True so bonuses/splits (very common on NSE)
    do not create artificial gaps.
  * The NSE trading calendar is derived from the traded days themselves, so Diwali
    Muhurat sessions and the shifting holiday list need no hard-coded calendar.
  * Returns are clipped to the exchange price band (+/-20% by default) before the
    3-sigma rule, since a circuit-limit day is a real, informative event, not an outlier.
"""
import numpy as np
import pandas as pd
import yfinance as yf


def load_universe(path: str, suffix: str = ".NS") -> list[str]:
    syms = pd.read_csv(path)["symbol"].astype(str).str.strip().str.upper()
    return [s if s.endswith((".NS", ".BO")) else s + suffix for s in syms]


def download(symbols, start, end, out_csv=None) -> pd.DataFrame:
    """Returns long-format frame: symbol, date, open, high, low, close, volume."""
    raw = yf.download(symbols, start=start, end=end, auto_adjust=True,
                      group_by="ticker", progress=False, threads=True)
    frames = []
    for s in symbols:
        try:
            df = raw[s].dropna(how="all").reset_index()
        except KeyError:
            continue
        df.columns = [c.lower() for c in df.columns]
        df["symbol"] = s
        frames.append(df[["symbol", "date", "open", "high", "low", "close", "volume"]])
    out = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])
    if out_csv:
        out.to_csv(out_csv, index=False)
    return out


def clean(df: pd.DataFrame, cfg: dict, train_end: str) -> pd.DataFrame:
    """Missing-day handling, universe filtering, and return winsorisation (Section 4.2)."""
    m, d = cfg["market"], cfg["data"]
    cal = pd.DatetimeIndex(sorted(df["date"].unique()))          # NSE calendar from the data
    keep, out = [], []
    for sym, g in df.groupby("symbol"):
        g = g.set_index("date").reindex(cal)
        # forward-fill only short gaps (halts/holidays); long gaps stay NaN and count against the stock
        g[["open", "high", "low", "close"]] = g[["open", "high", "low", "close"]].ffill(
            limit=d["ffill_max_gap"])
        g["volume"] = g["volume"].fillna(0.0)
        if g["close"].isna().mean() > d["max_missing_frac"]:
            continue                                              # delisted / illiquid -> drop
        g["symbol"] = sym
        keep.append(sym)
        out.append(g.rename_axis("date").reset_index())
    df = pd.concat(out, ignore_index=True).dropna(subset=["close"])

    df["ret"] = df.groupby("symbol")["close"].pct_change()
    band = m["circuit_pct"] / 100.0
    df["ret"] = df["ret"].clip(-band, band)                       # NSE price band
    tr = df[df["date"] <= pd.Timestamp(train_end)]
    stats = tr.groupby("symbol")["ret"].agg(["mean", "std"])      # training-only statistics
    df = df.join(stats, on="symbol")
    k = d["sigma_clip"]
    df["ret"] = df["ret"].clip(df["mean"] - k * df["std"], df["mean"] + k * df["std"])
    return df.drop(columns=["mean", "std"])


def _indicators(g: pd.DataFrame) -> pd.DataFrame:
    """The paper's 11 technical indicators (Section 4.6): momentum, volatility, trend, volume."""
    c, h, l, v = g["close"], g["high"], g["low"], g["volume"]
    out = pd.DataFrame(index=g.index)
    out["ret1"] = c.pct_change()                                            # momentum
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    out["rsi14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_sig"] = out["macd"].ewm(span=9, adjust=False).mean()
    ma20, sd20 = c.rolling(20, min_periods=5).mean(), c.rolling(20, min_periods=5).std()
    out["bb_pctb"] = (c - (ma20 - 2 * sd20)) / (4 * sd20).replace(0, np.nan)  # volatility
    out["bb_width"] = (4 * sd20) / ma20.replace(0, np.nan)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    out["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / c.replace(0, np.nan)
    out["sma20_gap"] = c / ma20.replace(0, np.nan) - 1                       # trend
    out["sma50_gap"] = c / c.rolling(50, min_periods=10).mean().replace(0, np.nan) - 1
    out["obv"] = (np.sign(d.fillna(0)) * v).cumsum()                        # volume
    out["mom10"] = c / c.shift(10) - 1
    return out


# 6 OHLCV-family + 11 technical indicators = the paper's 17-dimensional price representation
OHLCV_COLS = ["open", "high", "low", "close", "volume", "turnover"]
TA_COLS = ["ret1", "rsi14", "macd", "macd_sig", "bb_pctb", "bb_width", "atr14",
           "sma20_gap", "sma50_gap", "obv", "mom10"]
FEATURE_COLS = OHLCV_COLS + TA_COLS


def featurize(df: pd.DataFrame, train_end: str) -> pd.DataFrame:
    """Build the 17 price channels (d_p = 17, Table 4 'Price+TA+Text' row).

    z-scores use training-period statistics only, per symbol, to avoid leakage.
    """
    df = df.sort_values(["symbol", "date"]).copy()
    df["close_raw"] = df["close"]
    df["turnover"] = df["close"] * df["volume"]
    df = df.join(df.groupby("symbol", group_keys=False)[["close","high","low","volume"]].apply(_indicators))
    df["volume"] = np.log1p(df["volume"])
    df["turnover"] = np.log1p(df["turnover"])
    df["obv"] = np.sign(df["obv"]) * np.log1p(df["obv"].abs())

    tr = df[df["date"] <= pd.Timestamp(train_end)]
    mu = tr.groupby("symbol")[FEATURE_COLS].mean()
    sd = tr.groupby("symbol")[FEATURE_COLS].std().replace(0, 1.0)
    df[FEATURE_COLS] = ((df[FEATURE_COLS] - mu.reindex(df["symbol"]).to_numpy())
                        / sd.reindex(df["symbol"]).to_numpy())
    df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Eq. (17): next-day direction; strictly-up = 1, flat or down = 0 (no magnitude threshold)
    df["label"] = (df.groupby("symbol")["close_raw"].shift(-1) > df["close_raw"]).astype(int)
    return df
