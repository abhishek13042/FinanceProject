# Spec 05 — Training & evaluation

**Status: implemented incl. R6/R7 and the spec 06 gates, NOT yet run on real data.**
`src/train.py`, `src/eval.py`, `src/data/dataset.py`. Exercised by `scripts/smoke_eval.py`
on synthetic tensors — the full reporting path, not just the loop.

## Contract

- **In:** `prices.csv` (spec 01), `daily_embeddings.npz` (spec 03), `configs/india.yaml`.
- **Out:** `artifacts/best_seed<N>.pt` (one per seed), `artifacts/results_<tag>.json` — a
  single self-describing record per ablation carrying per-seed metrics, mean ± std, block
  bootstrap intervals, the constant-up floor, and the agreement-day split.

## Running it

```bash
python -m src.train --ablation full --seeds 42,43,44,45,46
python -m src.train --ablation price_only
python -m src.train --ablation shuffled_text
```

`--ablation` names come from `ABLATIONS` in `src/train.py`, one per row of the table below.
Each is a (config override, dataset transform) pair, so the table and the runnable set cannot
drift apart. Dataset transforms apply to **every** split: shuffling text in the test set only
would measure a distribution shift, not the absence of signal.

## Requirements

1. **R1 — Chronological split, zero overlap.** Train ≤2022-12-31, validation ≤2023-12-31,
   test 2024–2025. Same 62.5 / 12.5 / 25 proportions as the paper — measured on the real
   panel: **1,234 / 247 / 494 trading days = 62.5 / 12.5 / 25.0%**.

   The original `train_end` of 2023-06-30 was carried over from a draft and did *not* deliver
   those proportions on 2018–2025 data: it gave 1,358 / 123 / 494 days, a validation set of
   6.2% — half the intended size, and roughly 123 effective observations to make every model
   selection decision on. Corrected before any run; the test boundary was never touched, so
   spec 06 G7 is intact.
2. **R2 — Validation owns every decision.** Model selection, early stopping and all
   hyperparameter choices use validation MCC only.
3. **R3 — The test set is touched once**, after selection is final. Any tuning that consults
   test results invalidates the run; restart from a fresh split rather than rationalising.
4. **R4 — Metrics.** Accuracy, MCC and F1. MCC is primary: up-share is 0.513, so accuracy
   alone flatters a constant-up predictor.
5. **R5 — Report against baselines, always.** Constant-up (51.3%), price-only (text stream
   zeroed), and the full model. A number without these two reference points is uninterpretable.
6. **R6 — Multi-seed.** ≥5 seeds; report mean ± std. Single-seed results are not reportable.
   **Built** — `--seeds` defaults to five, `across_seeds` in `src/eval.py` aggregates.
7. **R7 — Block bootstrap for intervals.** Resample **trading days**, not samples. Bootstrapping
   individual stock-days inherits the independence error of spec 06 and produces fake-tight
   intervals. **Built** — `block_bootstrap` resamples whole days with replacement.
8. **R8 — Seed variance and sampling variance are reported separately.** Intervals are computed
   on the seed-averaged probability; mean ± std across seeds is a separate field. Pooling them
   would double-count the same uncertainty, and the two answer different questions: "would a
   different initialisation change this?" versus "would a different test period?"
9. **R9 — Every metric carries the trading day it was earned on.** `StockWindowDataset` keeps
   `dates` and `syms` per sample, aligned with prediction order (val/test loaders never
   shuffle). Without that column neither R7 nor spec 06's beta gate is computable at all.
10. **R10 — Windows are indexed, never materialised.** `StockWindowDataset` stores per-symbol
    daily matrices and slices them in `__getitem__`. Materialising each window's text costs
    `n_windows × 30 × 768` floats — **5.8 GB for the training split alone**, because
    consecutive windows overlap in 29 of their 30 days and every copy is paid for in full.
    The first real-data run died on exactly that allocation. Indexed, the whole panel's text
    is **303 MB** and the three splits build in one second.

## Planned ablations

Each reported with confidence intervals, or not reported at all:

| `--ablation` | Question |
|---|---|
| `price_only` (text zeroed) | does text contribute anything at all? |
| `concat` fusion vs co-attention | is the paper's central claim reproduced? |
| `oneway` cross-attn vs bidirectional | is symmetry the thing that matters? |
| no hierarchical summary (raw-headline embedding) | does the two-pass distillation earn its cost? — run `full` against the `_raw` embeddings from spec 03 |
| `layers2` — `n_layers` 4 vs 2 | spec 04's pending change |
| `d_p` 17 vs 6 (OHLCV only) | do the 11 technical indicators help? (paper Table 4: +0.86 pts) |
| `align32` / `align64` vs 17 | **spec 04 R5** — what does Eq. (1)'s 768→17 squeeze cost? |
| `shuffled_text` control | **negative control** — must land at chance |

The shuffled-text control permutes text **within each trading day**, so every window still
carries a real same-day summary — belonging to a different stock. Date alignment survives,
stock alignment does not. A control that instead paired windows with random-date text would
also destroy the date match, and could "pass" for the wrong reason.

This control is non-optional: if accuracy does not collapse to chance, the text stream is
being exploited through a leak, not through signal.

## Acceptance criteria

- [x] Loop runs end to end on synthetic data; loss decreases; early stopping fires.
- [x] Runs on the real panel without NaN — all 17 price channels finite across
      60,240 / 10,750 / 23,200 train/val/test windows.
- [x] Memory fits a laptop: 303 MB of text, 7 MB of price features, one-second build.
- [x] R6 and R7 implemented; all eight ablations construct, train and produce finite
      predictions (`scripts/smoke_eval.py`).
- [x] Day-block intervals are wider than a row bootstrap on the same predictions — the
      synthetic check measures 1.06×, which is small only because the synthetic universe is
      8 weakly-linked names. **Re-check on real data**, where ρ = 0.289 across 50 names should
      widen the ratio substantially. If it does not, the blocking is not doing its job.
- [x] Ablation transforms are non-destructive — `zero_text`/`shuffle_text` operate on copies
      and cannot leak into the next run.
- [ ] Full model beats price-only **outside overlapping confidence intervals**.
- [ ] Shuffled-text control lands at chance.
- [ ] Test metrics written once, with intervals and seed count.

## Cost, measured

460 ms/step on this laptop's CPU at `batch_size` 16, 3,765 steps/epoch → **0.48 h/epoch**,
so 20 epochs × 5 seeds is **48 CPU-hours per ablation** and the full eight-ablation matrix is
384. That is why training goes to a GPU. `scripts/run_all.py` skips ablations already on
disk, so a borrowed session can be resumed rather than restarted; see [HANDOFF.md](../HANDOFF.md).

## Honest expectation

The paper reports 67.01% / 0.346 MCC (US) and 70.24% / 0.293 MCC (CN). Those figures sit far
above what stock-direction work sustains out-of-sample; treat them as an upper bound, not a
target. Realistic NSE band: **53–57% accuracy, MCC 0.05–0.15**. Per spec 06, beating the
51.3% baseline significantly requires roughly **54.5%+**. A result of 53% is not a weak
success — it is indistinguishable from the baseline, and should be reported as such.
