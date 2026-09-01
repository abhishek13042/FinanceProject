# Study Guide 3 — Data contracts and the leakage-guard suite

Document 2 traced *behavior*. This document covers the *shapes* the pipeline agrees on
between stages, and the test suite that exists specifically to catch violations of those
shapes before they become a silently-inflated result.

Read after document 2. Time: 1–1.5 hours.

---

## 1. The five data contracts

| Stage boundary | File(s) | Status |
|---|---|---|
| Price download → clean → featurize | `data/interim/prices.csv` | **live, on disk, 39,055,211 bytes** |
| Filing collection → boilerplate filter | `data/raw/news.csv` | **live, on disk, 18,479,415 bytes** |
| Summarisation → embedding | `data/interim/daily_summaries.csv` (9,424 stock-days), embeddings | **executed** (Qwen2.5-1.5B-Instruct generation, mT5-encoder embedding — document 4 §5) |
| Windowing → training | in-memory (`StockWindowDataset`) | **executed against real embeddings** |
| Training → reporting | `artifacts/results_<tag>.json` (6 ablations × 5 seeds) | **executed** — document 4 covers the result |

### 1.1 `prices.csv` — one row per (symbol, date)

Columns: `symbol, date, open, high, low, close, volume, turnover, close_raw, ret1, rsi14,
macd, macd_sig, bb_pctb, bb_width, atr14, sma20_gap, sma50_gap, obv, mom10, label`.

The 17 feature columns (`FEATURE_COLS = OHLCV_COLS + TA_COLS`) are z-scored per symbol on
training-period statistics (document 2, decision 3/5); `close_raw` is the one column
deliberately left unscaled (decision 4); `label` is the strict next-day-direction target,
computed from `close_raw`, never from a z-scored column.

**This is the contract behind a README claim that was stale earlier in this project's
life and has since been corrected at the source.** The README once read *"Price channels
(d_p=5): close, log volume, return, high–low range, 20-day MA gap"* — an early draft of
the feature set, never updated after the design grew to 17 channels. It has since been
fixed directly in `README.md` (as part of the same commit that ran the real training
session, document 4). The contract itself, confirmed independently this session —

```bash
grep -n "n_price_feats" configs/india.yaml        # n_price_feats: 17
grep -n "FEATURE_COLS\s*=" -A3 src/data/prices.py  # OHLCV_COLS + TA_COLS, 6 + 11
```

— is **17 channels**, matching spec 01 R5 ("Seventeen channels") exactly, and now also
matching the README. The general lesson still holds for any *future* discrepancy: trust
the config and spec over prose when they disagree.

### 1.2 `news.csv` — one row per filing, pre-boilerplate-filter

Columns: `symbol, timestamp, headline, source` (`source ∈ {bse, rss}`). Deduplicated on
`(symbol, timestamp, headline)` (document 2, decision 8). This file holds the **raw**
61,791-row corpus; `drop_boilerplate()` is applied downstream, not baked into this file,
so the boilerplate rows are still inspectable here if a filtering-precision question ever
comes up.

### 1.3 `daily_summaries.csv` / embeddings — executed

By contract (spec 03), the summary file carries one row per `(symbol, attribution_date)`
with the pass-2 summary text; the embedding store carries one 768-d vector per same key,
keyed for `O(1)` lookup by `StockWindowDataset.__init__` (document 2, decision 16's
`emb.get((sym, d), zero)`). This stage has now run: 9,424 stock-days summarised, using
Qwen2.5-1.5B-Instruct for generation and the mT5 encoder for embedding — a mid-session
model swap covered in full in document 4 §5, after the originally-planned mT5-only
summariser was caught inventing content at the preflight gate.

### 1.4 `results_<tag>.json` — executed

By contract (`train.py::main`), one file per ablation tag, holding `test_mean_over_seeds`,
`test_seed_averaged`, `ci_block_bootstrap_days`, `baseline_constant_up`, and
`beta_contamination` — i.e., a number is never reported alone; the schema itself forces a
baseline comparison and an interval alongside every metric. Six such files now exist in
`artifacts/` (`results_full.json`, `results_price_only.json`, `results_shuffled_text.json`,
`results_concat.json`, `results_oneway.json`, `results_raw_text.json`), plus
`SESSION_REPORT.md` summarising all six. Document 4 covers what they contain.

## 2. The leakage-guard suite — `tests/test_leakage.py`, 14 guards

This is spec 07 turned into code. Its own words on why it exists: *"a leak also
flatters"* — a broken pipeline that leaks future information into training doesn't
produce an obviously-wrong number, it produces a **better-looking** one. Nothing in a
shipped accuracy figure signals the leak on its own; only a guard that fails *before* the
number is ever computed can catch it.

| # | Guard | What it actually checks |
|---|---|---|
| 1 | Split disjointness/ordering | train/val/test date ranges don't overlap and are chronological |
| 2 | No window crosses a split boundary | exact expected-count arithmetic, not a spot check |
| 3 | Scaler trained on training rows only | a synthetic regime-break (deliberately different `scale`/`shift` before/after `train_end`) — if the scaler saw post-boundary rows, the test rows would come out at the wrong mean |
| 4 | `close_raw` is never scaled | direct equality check against the raw download |
| 5 | Label is strict next-day direction | including the edge case: a fabricated final day always labels `0` |
| 6 | The final day in the panel is never a training target | there's no "next day" to compare against |
| 7 | **Window text never reaches into the future** | the strongest guard in the file — see §2.1 |
| 8 | Post-close news attributes to the next session | parametrized over explicit timestamps, including a Friday-evening → Monday case |
| 9 | Session buckets are ordered | bucket IDs increase monotonically through the trading day |
| 10 | Patch count matches Eq. 4 | `N = floor((L-P)/S) + 2` computed both ways and compared |
| 11 | Parameter count is pinned | exactly 1,254,312 at `d_align=17` — a capacity tripwire, not a leakage check, but lives in this file because an accidental capacity change is the kind of silent regression this suite is built to catch |
| 12 | Zero-text input yields finite gradients | the `price_only` ablation's correctness depends on this |
| 13 | **Text stream actually reaches the output** | the mirror-image bug — see §2.2 |
| 14 | On-disk panel matches the current `train_end` | the regime-break guard — see §3 |

### 2.1 Guard 7 — the one-hot date-ordinal trick

Rather than trust a generic "no future dates in this window" assertion (which could pass
by accident if the test data happens not to exercise the boundary), the test constructs
text embeddings that **encode their own date as a one-hot ordinal** — embedding `i`'s
only non-zero entry is at index `i`. A window's embeddings, summed and `argmax`'d, then
directly reveal the *latest date index actually present* in the window the model would
see. If windowing ever included one extra (future) day, the `argmax` would point past the
window's intended end and the test fails on a concrete integer mismatch, not a fuzzy
comparison.

### 2.2 Guard 13 — checking for the opposite bug

Every other guard checks "did something forbidden leak in." Guard 13 checks the reverse:
"does the text stream reach the output at all." It runs the same model twice — once with
real text, once with `zero_text()` — and asserts the two output tensors are **not**
`torch.allclose`. A model architecturally capable of ignoring its text input entirely
(e.g., a wiring bug that never connects `CoAttentionFusion`'s output to the head) would
pass every leakage guard trivially and still be broken — this guard exists so that
"the text pathway is provably load-bearing" is itself a tested fact, not an assumption.

## 3. A real incident this suite exists to catch — `train_end`

Spec 05 R1, in its own words: `train_end` was originally `2023-06-30`, "carried over from
a draft." Measured against the real 2018–2025 panel, that boundary produced a validation
set of **6.2%** of the data — roughly half the intended 12.5% — because the draft date
predated most of the collected history. The value was corrected to `2022-12-31`,
confirmed live in `configs/india.yaml` this session.

The dangerous part isn't the wrong date itself — it's what happens *after* fixing it.
`prices.csv` on disk was built with the *old* boundary's training statistics baked into
its z-scores (document 2, decision 3). Correcting the config alone does not retroactively
fix a file already on disk. **Guard 14** (`test_panel_on_disk_was_built_with_the_current_
train_end`) exists purely to catch this: it re-derives what the training slice's mean/std
*should* be under the current config, and checks that the on-disk panel's training rows
actually standardize to (0, 1) under that definition — if `prices.csv` were stale, this
guard fails even though every other guard would pass, because the *code* is correct and
only the *artifact* is wrong.

This is the same class of bug as the `append_csv` regression below: not a logic error
anyone would catch by reading the code, but a state-on-disk error only a check against
the real file catches.

## 4. A second real incident — the `append_csv` tz-dtype regression

`tests/test_resume.py`'s `test_append_csv_is_idempotent_across_a_reload`, docstring
quoted directly:

> *"Regression. Rows already on disk read back as strings while incoming rows are
> tz-aware Timestamps; deduping across that boundary matched nothing and doubled
> news.csv on every re-run. Caught live, not by the fakes — the fakes never wrote the
> file twice."*

The bug: `pandas.read_csv` deserialises a timestamp column as plain strings unless told
otherwise; `dedup(symbol, timestamp, headline)` compares that string against a live
`pd.Timestamp` object from the current run and never finds a match, because `"2024-03-
01 16:45:00+05:30" != Timestamp('2024-03-01 16:45:00+0530')` even though they represent
the same instant. Every resumed run silently re-appended everything it had already
collected. The test's own note is the honest part: unit tests using in-memory fakes
never exercised an actual disk round-trip, so the bug was invisible until a real
multi-session collection run hit it.

## 5. A third real incident — the 5.8GB allocation

Already covered in document 2, decision 15, from the angle of the *design* it produced
(indexed windows, not materialised ones). From the angle of *how it was found*: spec 05
R10 states plainly, "the first real-data run died on exactly that allocation" — this was
not a back-of-envelope estimate that motivated a design choice preemptively; the naive
materialised-window implementation was tried first, against real data, and it ran out of
memory. The indexed design (document 2, §C.1) followed from that failure, not before it.

## 6. What's genuinely untested

Being honest about the suite's own boundaries (spec 07's explicit scope section):

- It does not test **model quality** — a model that trains correctly but predicts badly
  passes every guard here.
- It cannot enforce **G7** (test set touched exactly once) — that's a process discipline,
  not a checkable code invariant (document 2, decision 19).
- It does not cover **summariser output quality** by automated test — that was checked
  manually instead, at the preflight gate, on 8 real filings, and it caught exactly the
  degeneracy risk spec 03 flagged: the originally-planned mT5 model invented facts rather
  than summarising (2/8 correct company names). It was swapped for Qwen2.5-1.5B-Instruct
  (8/8 correct) before any GPU time went into a full run — document 4 §5 has the story.
  This class of defect still isn't something the automated guard suite checks for itself.

## Diagram — where each guard sits in the pipeline

```
prices.csv ──────► guards 3,4,5,6,14 (scaling, label, train_end-on-disk)
                          │
news.csv → boilerplate → guards 8,9 (attribution, session ordering)
                          │
     (summaries/embeddings — executed, document 4 §5)
                          │
StockWindowDataset ──────► guards 2,7,10,12,13 (window boundary, patch count,
                                                  zero-text gradients, text reaches output)
                          │
model.py ─────────────────► guard 11 (parameter count pinned)
                          │
train.py split ──────────► guard 1 (disjointness/ordering)
```

---

## Self-test

**Level 1.** Which guard would fail if `prices.csv` were rebuilt under an old `train_end`
value after the config had already been corrected? What does guard 13 check that no
other guard checks?

**Level 2.** Explain the `append_csv` regression in your own words — what made it
invisible to the in-memory fakes, and what made it visible in practice?

**Level 3.** *"Your leakage tests all pass — doesn't that mean the pipeline is
correct?"* Answer using §6: passing guards means no *known-checkable* leakage exists;
it says nothing about model quality, and it cannot enforce the once-only test-touch
discipline (G7) — that's a human process, not a code invariant. Summariser output quality
also isn't something the automated suite checks; it was caught by manual inspection at
the preflight gate instead (§6, document 4 §5).

Next: [04-results-and-how-to-defend-them.md](04-results-and-how-to-defend-them.md).
