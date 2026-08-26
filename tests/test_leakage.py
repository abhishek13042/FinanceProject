"""Spec 07 — leakage guards.

Every test here encodes a way this project could produce a good-looking number that is
wrong. They are cheap, they run on CPU in seconds, and they are the only thing standing
between a refactor and a silently invalid result.

    python -m pytest tests/ -q

Tests that need the real panel skip cleanly when `data/interim/prices.csv` is absent, so the
suite is runnable before spec 01 has been executed.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import StockWindowDataset, chronological_split
from src.data.prices import FEATURE_COLS, featurize
from src.models.model import StockTrendModel, n_patches
from src.summarize.hierarchical import attribution_date, bucket_by_session

CFG = yaml.safe_load(open("configs/india.yaml"))
PRICE_CSV = CFG["data"]["price_csv"]
needs_panel = pytest.mark.skipif(not os.path.exists(PRICE_CSV),
                                 reason="run scripts/build_prices.py first")


@pytest.fixture(scope="module")
def panel():
    return pd.read_csv(PRICE_CSV, parse_dates=["date"])


# --------------------------------------------------------------------------- splits


@needs_panel
def test_splits_are_disjoint_and_ordered(panel):
    tr, va, te = chronological_split(panel, CFG)
    assert tr["date"].max() < va["date"].min() < va["date"].max() < te["date"].min()
    assert len(tr) + len(va) + len(te) <= len(panel)


@needs_panel
def test_no_window_crosses_a_split_boundary(panel):
    """Windows are built inside each split independently. If that ever changes, a validation
    window would reach back into training days and the split would stop meaning anything.

    Checked by identity rather than by inspection: a split holding n days per symbol can
    yield exactly n - L windows (the last day has no next-day label), so matching that count
    proves no window borrowed a day from the neighbouring split."""
    L = CFG["model"]["seq_len"]
    for split in chronological_split(panel, CFG):
        expected = sum(max(0, n - L) for n in split.groupby("symbol").size())
        ds = StockWindowDataset(split, {}, L, 8)
        assert len(ds) == expected


# --------------------------------------------------------------------------- scaling


def test_scaler_uses_training_rows_only():
    """z-scores must be computed from training statistics. Fitting on the full panel leaks the
    test period's mean and variance backwards - a small effect that is impossible to see in
    the metrics and impossible to defend afterwards."""
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2020-01-01", periods=600)
    n = len(days)
    # A deliberate regime break: the post-train period is shifted and far more volatile. If
    # the scaler saw it, the training slice would not standardise to (0, 1).
    scale = np.where(np.arange(n) < 400, 1.0, 8.0)
    shift = np.where(np.arange(n) < 400, 0.0, 50.0)
    df = pd.DataFrame({"symbol": "A", "date": days,
                       "close": 100 + np.cumsum(rng.normal(0, 1, n)) * scale + shift})
    df["open"] = df["high"] = df["low"] = df["close"]
    df["volume"] = rng.integers(1e5, 1e6, n).astype(float)
    df["ret"] = df["close"].pct_change()
    train_end = days[399].strftime("%Y-%m-%d")

    out = featurize(df, train_end)
    tr = out[out["date"] <= train_end]
    te = out[out["date"] > train_end]
    assert abs(tr["close"].mean()) < 0.05 and abs(tr["close"].std() - 1) < 0.05
    # the test period must NOT be standardised - that is the proof the fit was train-only
    assert abs(te["close"].mean()) > 0.5


@needs_panel
def test_close_raw_is_left_unscaled(panel):
    """Labels are derived from close_raw. If it were z-scored per symbol the comparison would
    still 'work' (monotone within a symbol) but any later use of it as a price would be wrong."""
    assert (panel["close_raw"] > 0).all()
    assert panel["close_raw"].std() > 10          # rupees, not standard deviations


# --------------------------------------------------------------------------- labels


@needs_panel
def test_label_is_strictly_next_day_direction(panel):
    """Eq. (17): strictly-up = 1, flat or down = 0, no magnitude threshold, no peeking beyond
    t+1."""
    df = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    nxt = df.groupby("symbol")["close_raw"].shift(-1)
    valid = nxt.notna()
    expected = (nxt > df["close_raw"]).astype(int)
    assert (df.loc[valid, "label"] == expected[valid]).all()
    # and the fabricated final-day labels are all zero, never a silent 1
    assert (df.loc[~valid, "label"] == 0).all()


@needs_panel
def test_final_day_per_symbol_is_never_a_training_target(panel):
    """The last day has no t+1 close, so its label is fabricated. It must not become a sample."""
    L = CFG["model"]["seq_len"]
    tr = chronological_split(panel, CFG)[0]
    ds = StockWindowDataset(tr, {}, L, 8)
    last = tr.groupby("symbol")["date"].max().dt.strftime("%Y-%m-%d")
    assert not set(zip(ds.syms, ds.dates)) & set(zip(last.index, last.values))


# --------------------------------------------------------------------------- text alignment


def _synth_panel(days=120, syms=("A", "B")):
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2021-01-04", periods=days)
    out = []
    for s in syms:
        d = pd.DataFrame({"symbol": s, "date": dates, "close_raw": 100.0})
        for c in FEATURE_COLS:
            d[c] = rng.normal(size=days)
        d["label"] = rng.integers(0, 2, days)
        out.append(d)
    return pd.concat(out, ignore_index=True)


def test_window_text_never_comes_from_the_future():
    """The strongest guard in this file. Each embedding is a one-hot of its own date ordinal,
    so the text tensor of a window is a readable record of which days it drew text from. A
    window ending on day i must contain days i-L+1..i and nothing later."""
    L, df = CFG["model"]["seq_len"], _synth_panel()
    dates = sorted(df["date"].unique())
    ordinal = {pd.Timestamp(d).strftime("%Y-%m-%d"): i for i, d in enumerate(dates)}
    dim = len(dates)
    emb = {(s, d): np.eye(dim, dtype=np.float32)[i]
           for s in df["symbol"].unique() for d, i in ordinal.items()}

    ds = StockWindowDataset(df, emb, L, dim)
    for j in range(0, len(ds), 37):                       # stride: the check is O(L) per window
        end = ordinal[ds.dates[j]]
        used = ds.text_window(j).argmax(axis=1)
        assert used.max() <= end, "text from a future trading day entered the window"
        assert list(used) == list(range(end - L + 1, end + 1))


def test_post_close_news_moves_to_the_next_session():
    """A 16:00 IST filing is not tradable until the next open. Attributing it to the same day
    hands the model tomorrow's news - the single largest look-ahead risk in the text pipeline."""
    days = pd.DatetimeIndex(pd.bdate_range("2023-01-02", periods=10))
    ts = pd.Series(pd.to_datetime(
        ["2023-01-03 09:00:00+05:30",    # pre-open  -> same day
         "2023-01-03 11:30:00+05:30",    # intraday  -> same day
         "2023-01-03 16:00:00+05:30",    # post-close-> next session
         "2023-01-06 18:30:00+05:30"]))  # Friday PM -> Monday
    got = attribution_date(ts, days, CFG)
    assert list(got.dt.strftime("%Y-%m-%d")) == ["2023-01-03", "2023-01-03",
                                                 "2023-01-04", "2023-01-09"]
    assert (got.values >= ts.dt.tz_convert(CFG["market"]["tz"]).dt.normalize()
            .dt.tz_localize(None).values).all(), "news attributed to an earlier day"


def test_session_buckets_are_ordered_and_bounded():
    nb = CFG["market"]["hourly_buckets"]
    ts = pd.Series(pd.to_datetime([f"2023-01-03 {h:02d}:30:00+05:30" for h in range(6, 21)]))
    b = bucket_by_session(ts, CFG)
    assert b.min() == 0 and b.max() == nb - 1
    assert (b.diff().dropna() >= 0).all(), "buckets must be monotone in time of day"


# --------------------------------------------------------------------------- architecture


def test_patch_count_matches_equation_4():
    m = CFG["model"]
    assert n_patches(m["seq_len"], m["patch_len"], m["stride"]) == 6
    assert n_patches(30, 10, 5) == 6 and n_patches(96, 16, 8) == 12


def test_paper_configuration_parameter_count():
    """1,254,312 parameters at d_align = 17. This number is the tripwire for an accidental
    architecture change: any edit that alters capacity will move it."""
    m = dict(CFG["model"]); m["d_align"] = m["n_price_feats"]
    assert sum(p.numel() for p in StockTrendModel(m).parameters()) == 1_254_312


def test_zero_text_gives_finite_gradients():
    """The price-only ablation feeds an all-zero text stream through a softmax over text keys.
    A uniform-attention degenerate case is fine; NaN is not."""
    m = dict(CFG["model"]); m["n_layers"] = 1
    model = StockTrendModel(m)
    x = torch.randn(4, m["seq_len"], m["n_price_feats"])
    t = torch.zeros(4, m["seq_len"], m["text_dim"])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model(x, t), torch.zeros(4))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_text_stream_actually_reaches_the_output():
    """Guards against the opposite failure: a wiring bug that silently drops text would make
    every text ablation report 'no effect' and look like a finding."""
    m = dict(CFG["model"]); m["n_layers"] = 1
    model = StockTrendModel(m).eval()
    x = torch.randn(4, m["seq_len"], m["n_price_feats"])
    a = model(x, torch.zeros(4, m["seq_len"], m["text_dim"]))
    b = model(x, torch.randn(4, m["seq_len"], m["text_dim"]))
    assert not torch.allclose(a, b, atol=1e-6)


@needs_panel
def test_panel_on_disk_was_built_with_the_current_train_end(panel):
    """A stale rebuild is invisible leakage. `prices.csv` carries no record of which
    `train_end` its z-scores were fitted on, so if the split boundary moves and the panel is
    not rebuilt, the features silently encode statistics from what is now validation data.

    Detected by the arithmetic that must hold if the fit was train-only: pooled across
    symbols, the training slice standardises to (0, 1) per symbol and the later periods do
    not have to."""
    tr = chronological_split(panel, CFG)[0]
    mu = tr.groupby("symbol")[FEATURE_COLS].mean().abs().max().max()
    sd = tr.groupby("symbol")[FEATURE_COLS].std()
    assert mu < 0.02, f"training slice not centred (max |mean| = {mu:.3f}) - rebuild prices.csv"
    # constant channels are legitimately left at std 0 by the /0 guard in featurize
    off = (sd[sd > 0] - 1).abs().max().max()
    assert off < 0.02, f"training slice not unit-variance (max dev = {off:.3f}) - rebuild"
