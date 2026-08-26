# Specs

The project is split into eight specs. Each is independently buildable and testable, with an
explicit input/output contract so a spec can be worked on without holding the others in mind.

| # | Spec | Owns | Status |
|---|---|---|---|
| 00 | [System overview](00-system.md) | architecture contract, module boundaries, glossary | contract fixed |
| 01 | [Price data](01-price-data.md) | NSE ingestion, cleaning, features, labels | **built, live-verified** |
| 02 | [News collection](02-news-collection.md) | BSE announcements, RSS, symbol mapping | **built, live-verified** |
| 03 | [Hierarchical summarization](03-summarization.md) | two-pass LLM summary → embeddings | built, **not yet run** |
| 04 | [Model](04-model.md) | alignment, patching, co-attention, classifier | **built, smoke-tested** |
| 05 | [Training & evaluation](05-training-eval.md) | loop, splits, metrics, confidence intervals | built, **not yet run on real data** |
| 06 | [Statistical validity](06-statistical-validity.md) | effective sample size, power, acceptance gates | **gates defined and enforced in `src/eval.py`** |
| 07 | [Leakage guards](07-leakage-tests.md) | the test suite that keeps a result meaningful | **13 tests passing** |

## Build order

```
01 price data ──┐
                ├──> 03 summarization ──> 05 training ──> 06 gates
02 news ────────┘         (needs 01 for the trading calendar)
04 model ───────────────────────────────┘
```

01, 02 and 04 are done. The critical path to a first result is **03 → 05**, and 06 decides
whether that result means anything.

## Conventions binding on every spec

- **No look-ahead, ever.** Any statistic (z-score means, sigma clips, dedup thresholds,
  model selection) is computed on training data only and applied unchanged downstream.
- **Post-close news belongs to the next session.** Measured: most BSE filings land
  16:00–20:00 IST. See spec 02.
- **The test window is evaluated once**, after model selection is final on validation.
- Config is the single source of truth: `configs/india.yaml`. No hard-coded constants
  in module code.
