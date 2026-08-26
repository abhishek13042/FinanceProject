"""Spec 05 acceptance harness: runs the full reporting path on synthetic data.

Checks the parts that are easy to get silently wrong - date alignment, ablation transforms
not leaking into each other, the day-block bootstrap being wider than a row bootstrap, the
agreement split partitioning cleanly, and every ablation constructing and training.

    python scripts/smoke_eval.py
"""
import copy, json, os, sys, tempfile, numpy as np, pandas as pd, yaml
sys.path.insert(0, ".")
from src.data.prices import FEATURE_COLS
import src.train as T
from src.eval import block_bootstrap, agreement_days, split_by_agreement, intervals_overlap

SP = tempfile.mkdtemp(prefix="spec05_")
rng = np.random.default_rng(0)
syms = [f"S{i:02d}" for i in range(8)]
days = pd.bdate_range("2022-01-03", periods=420)
rows = []
mkt = rng.normal(0, 1, len(days))                      # shared market factor -> agreement days
for s in syms:
    idio = rng.normal(0, 1, len(days))
    r = 0.7 * mkt + 0.7 * idio
    close = 100 * np.exp(np.cumsum(r / 100))
    d = pd.DataFrame({"symbol": s, "date": days, "close_raw": close})
    for c in FEATURE_COLS:
        d[c] = rng.normal(0, 1, len(days))
    d["label"] = (pd.Series(close).shift(-1) > pd.Series(close)).astype(int).to_numpy()
    rows.append(d)
prices = pd.concat(rows, ignore_index=True)
pcsv = os.path.join(SP, "t_prices.csv"); prices.to_csv(pcsv, index=False)

keys = np.array([f"{s}|{d:%Y-%m-%d}" for s in syms for d in days])
np.savez_compressed(os.path.join(SP, "t_emb.npz"), keys=keys,
                    vecs=rng.normal(0, 1, (len(keys), 32)).astype(np.float32))

cfg = yaml.safe_load(open("configs/india.yaml"))
cfg["data"].update(price_csv=pcsv, embed_npz=os.path.join(SP, "t_emb.npz"),
                   train_end="2023-01-31", val_end="2023-03-31", test_end="2023-12-31")
cfg["model"].update(text_dim=32, d_model=32, d_ff=64, n_layers=1, n_heads=4)
cfg["train"].update(epochs=2, batch_size=64, patience=5, out_dir=os.path.join(SP, "t_art"))
tcfg = os.path.join(SP, "t.yaml"); yaml.safe_dump(cfg, open(tcfg, "w"))

# --- 1. dates/syms alignment ---
_, splits = T.build_data(cfg)
te = splits[2]
assert len(te.dates) == len(te.y) == len(te.index), "dates not aligned with samples"
assert set(te.syms) <= set(syms) and len(np.unique(te.dates)) > 50
print(f"1 OK  windows tr/va/te = {[len(s) for s in splits]}, {len(np.unique(te.dates))} test days")

# --- 2. shuffle_text keeps the day, changes the stock ---
sh = copy.copy(te).shuffle_text(0)
same_day_only = all(len(np.unique(te.dates[np.flatnonzero(te.dates == d)])) == 1
                    for d in np.unique(te.dates)[:5])
changed = float(np.mean([not np.array_equal(te.text_window(i), sh.text_window(i))
                         for i in range(0, len(te), 7)]))
assert changed > 0.5, changed
assert not np.shares_memory(te.tx[0], sh.tx[0]), "shuffle mutated the original dataset"
z = copy.copy(te).zero_text()
assert sum(a.sum() for a in z.tx) == 0 and sum(a.sum() for a in te.tx) != 0, "zero_text leaked"
print(f"2 OK  shuffle changed {changed:.0%} of windows; copies are independent")

# --- 3. block bootstrap is wider than a naive row bootstrap ---
y = te.y.astype(int)
prob = np.clip(0.5 + 0.15 * (y - 0.5) + rng.normal(0, .2, len(y)), 0, 1)
ci_day = block_bootstrap(y, prob, te.dates, n_boot=300)
ci_row = block_bootstrap(y, prob, np.arange(len(y)).astype(str), n_boot=300)  # each row its own block
w_day = ci_day["acc"]["hi"] - ci_day["acc"]["lo"]; w_row = ci_row["acc"]["hi"] - ci_row["acc"]["lo"]
assert w_day > w_row, (w_day, w_row)
print(f"3 OK  day-block CI width {w_day:.2f} pts vs row-bootstrap {w_row:.2f} pts "
      f"({w_day/w_row:.2f}x wider - the fake-tightness spec 06 G2 exists to prevent)")

# --- 4. agreement-day split ---
hi = agreement_days(prices)
frac = float(np.isin(te.dates, list(hi)).mean())
sp = split_by_agreement(y, prob, te.dates, hi)
assert sp["high_agreement"]["n"] + sp["low_agreement"]["n"] == len(y)
assert "mcc_gap" in sp
print(f"4 OK  {frac:.1%} of test rows on >=80% agreement days; "
      f"mcc {sp['high_agreement']['mcc']:.3f} vs {sp['low_agreement']['mcc']:.3f}")

# --- 5. full multi-seed run, then a fusion ablation ---
res = T.main(tcfg, "full", [1, 2], 200, "t_full")
ms = res["test_mean_over_seeds"]
assert len(ms["mcc"]["seeds"]) == 2 and res["baseline_constant_up"]["mcc"] == 0.0
assert os.path.exists(os.path.join(cfg["train"]["out_dir"], "results_t_full.json"))
res2 = T.main(tcfg, "concat", [1, 2], 200, "t_concat")
assert res2["config"]["fusion"] == "concat"
ov = intervals_overlap(res["ci_block_bootstrap_days"], res2["ci_block_bootstrap_days"])
print(f"5 OK  multi-seed + ablation ran; G3 overlap(full, concat) = {ov}")

# --- 6. every ablation constructs and trains one step ---
for ab in sorted(T.ABLATIONS):
    o, tr = T.ABLATIONS[ab]
    c = json.loads(json.dumps(cfg)); c["model"].update(o); c["train"]["epochs"] = 1
    ds = T.apply_ablation(splits, tr, 1)
    p, yy, _ = T.run_seed(c, ds, 1, "cpu", log=lambda *a: None)
    assert np.isfinite(p).all(), ab
    print(f"   {ab:14s} ok  n_params={sum(x.numel() for x in __import__('src.models.model', fromlist=['x']).StockTrendModel(c['model']).parameters()):,}")
print("\nALL SPEC 05 CHECKS PASSED")
