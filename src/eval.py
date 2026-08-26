"""Spec 05 R4/R7 and the spec 06 gates, in one place.

Everything here operates on *arrays of predictions plus the trading day each prediction was
made on*. That extra column is the whole point: stock-days are not independent (spec 06,
rho = 0.289), so any interval computed by resampling rows is fake-tight. We resample days.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

METRICS = ("acc", "mcc", "f1")


def metrics(y, prob, thresh: float = 0.5) -> dict:
    pred = (prob >= thresh).astype(int)
    return {"acc": accuracy_score(y, pred) * 100,
            "mcc": matthews_corrcoef(y, pred),
            "f1": f1_score(y, pred, zero_division=0),
            "n": int(len(y))}


def constant_up(y) -> dict:
    """Spec 05 R5 floor. MCC is 0 by construction - that is exactly why MCC is primary."""
    return metrics(y, np.ones(len(y), dtype=np.float32))


def block_bootstrap(y, prob, dates, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Spec 05 R7 / spec 06 G2. Resample whole trading days with replacement.

    A trading day is the block because that is the unit correlation lives in: on a given day
    the 50 names share a market factor, so they carry roughly one day's worth of independent
    information between them, not 50. Resampling rows instead would divide the interval width
    by ~sqrt(50) and manufacture significance that is not there.
    """
    rng = np.random.default_rng(seed)
    days, inv = np.unique(dates, return_inverse=True)
    by_day = [np.flatnonzero(inv == i) for i in range(len(days))]
    draws = {k: [] for k in METRICS}
    for _ in range(n_boot):
        pick = rng.integers(0, len(days), len(days))
        idx = np.concatenate([by_day[i] for i in pick])
        m = metrics(y[idx], prob[idx])
        for k in METRICS:
            draws[k].append(m[k])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: {"lo": float(np.percentile(v, lo)), "hi": float(np.percentile(v, hi)),
                "se": float(np.std(v, ddof=1))} for k, v in draws.items()}


def agreement_days(prices, thresh: float = 0.8) -> set:
    """Spec 06 beta-contamination gate. Days where >= `thresh` of the universe moved the same
    way. Computed from the label column, so it needs no extra data and matches exactly what
    the model is scored against."""
    g = prices.groupby("date")["label"]
    share = g.mean()                                    # fraction of the universe that rose
    hi = share[(share >= thresh) | (share <= 1 - thresh)]
    return {d.strftime("%Y-%m-%d") for d in hi.index}


def split_by_agreement(y, prob, dates, hi_days: set) -> dict:
    """Metrics on index-driven days vs the rest.

    The gate: if the edge is confined to the high-agreement bucket, the model is timing the
    market, not the stock, and the headline number is not a stock-prediction result.
    """
    m = np.isin(dates, list(hi_days))
    out = {"high_agreement": metrics(y[m], prob[m]) if m.any() else None,
           "low_agreement": metrics(y[~m], prob[~m]) if (~m).any() else None}
    if out["high_agreement"] and out["low_agreement"]:
        out["mcc_gap"] = out["high_agreement"]["mcc"] - out["low_agreement"]["mcc"]
    return out


def across_seeds(runs: list) -> dict:
    """Spec 05 R6 / spec 06 G1: mean +/- std over seeds. Reported for every headline number;
    a single-seed figure is not reportable."""
    return {k: {"mean": float(np.mean([r[k] for r in runs])),
                "std": float(np.std([r[k] for r in runs], ddof=1)) if len(runs) > 1 else 0.0,
                "seeds": [float(r[k]) for r in runs]} for k in METRICS}


def intervals_overlap(a: dict, b: dict, key: str = "mcc") -> bool:
    """Spec 06 G3. Two results whose bootstrap intervals overlap are not distinguishable, and
    the difference between them must not be written up as a finding."""
    return not (a[key]["hi"] < b[key]["lo"] or b[key]["hi"] < a[key]["lo"])
