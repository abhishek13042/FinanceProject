# Study Guide 4 — What's actually been run, and what defends it

Documents 1–3 covered the design. This one covers **evidence**: what has genuinely been
executed in this project, what hasn't, and how to answer questions about both honestly.

Read after document 3. Time: 45–60 minutes.

---

## 1. The headline fact, stated first, not buried

**There is no trained-model accuracy or MCC number in this project yet.** `artifacts/` is
empty — zero `results_*.json` files. Spec 03 (LLM summarisation) has never been run
because it needs a GPU this machine doesn't have; spec 05 (training) is fully built and
tested on synthetic tensors but has never executed against real embeddings. Anyone asking
"what's the accuracy" gets this answer first, not last, per document 1 §7.5.

## 2. What HAS actually been run — live-verified this session

```
$ python -m pytest tests/ -q --no-header
........................                                                 [100%]
24 passed in 383.81s (0:06:23)
```

**24 passed, 0 failed, run twice independently this session with identical results.**
This is the entire test suite: 14 leakage guards (`test_leakage.py`, document 3 §2) plus
the news-resume suite (`test_resume.py`, document 3 §4). Both files use synthetic
tensors, fakes, or stub models — this result verifies the pipeline's **logic and data
contracts** are correct, not that any real accuracy number is good.

**What is genuinely built and executed at full scale, confirmed by direct file
inspection** (not by reading a spec's prose):

| Stage | Evidence | Status |
|---|---|---|
| Price download/clean/featurize | `data/interim/prices.csv`, 39,055,211 bytes | **executed, full scale** |
| News collection | `data/raw/news.csv`, 18,479,415 bytes; `bse_scrip_map.csv`; `universe50.csv`; `news.bse.done.txt` | **executed, full scale** |
| Summarisation | `daily_summaries.csv`, `daily_embeddings.npz` | **not present — never run** |
| Training | `artifacts/results_*.json` | **not present — never run** |

Specs 01 and 02's acceptance criteria (98,690 price rows across 50 symbols, up-share
0.510; 61,791 raw filings, 38.1% stock-day coverage) describe numbers measured against
these real files, not synthetic placeholders — this part of the pipeline is genuinely
done, not aspirational.

## 3. Declared gaps — what this session did not verify

1. **No real model output exists.** Every architecture claim in document 1/2 (parameter
   count, forward-pass shapes) is verified by the parameter-count guard and by code
   reading, not by an actual training run producing real predictions.
2. **`src/summarize/hierarchical.py`'s generation quality is unverified.** The prompts
   and pooling logic were read in full, but no mT5 call has actually been made this
   session — spec 03's own flagged risk (degenerate summaries on compliance filings) is
   unresolved either way.
3. **`src/data/news_sources.py` was read only partially** (~40 of 336 lines) — the header
   constants and the `TATAMOTORS→TMCV` alias were confirmed, but the bulk of the BSE/RSS
   fetch logic was not read line-by-line this session.
4. **`scripts/build_prices.py`, `build_text.py`, `collect_news.py`, `run_all.py`,
   `smoke_eval.py`, `smoke_test.py`, and `src/summarize/runner.py` were listed but not
   opened this session** — their existence is confirmed, their contents are not.
5. **The two documentation staleness gaps are corrections, not fixes** — the README
   itself was not edited this session; document 3 §1.1 is the canonical correction to
   read instead, but a reader going only to `README.md` will still see the stale d_p=5
   and six-specs claims until someone updates that file.

## 4. The statistical gates that will govern any future number — spec 06's G1–G8

No accuracy figure from this pipeline is reportable on its own; it must clear these
before being quoted as a result:

| Gate | What it requires | Enforcement |
|---|---|---|
| G1 | ≥5 seeds, mean±std reported | code (`train.py --seeds`, `eval.py::across_seeds`) |
| G2 | Day-level block bootstrap, never row-level | code (`eval.py::block_bootstrap`) |
| G3 | Ablation comparisons need non-overlapping intervals | code (`eval.py::intervals_overlap`) |
| G4 | Shuffled-text negative control must collapse to chance | code (`dataset.py::shuffle_text` + the `shuffled_text` ablation) |
| G5 | Effective sample size (Kish) reported alongside nominal N | code, ties into §5 below |
| G6 | Cross-sectional correlation ρ measured, not assumed zero | code (measured at 0.289 for the adopted universe) |
| G7 | Test set touched exactly once, after model selection is final | **process only — not enforceable by code** (document 2, decision 19) |
| G8 | Market-agreement / beta-contamination split reported | code (`eval.py::agreement_days`, `split_by_agreement`) |

The honest framing from spec 06 itself: the source paper is *not* methodologically naive
— it does report mean±std over 10 seeds with paired t-tests. These gates are **stricter
than the paper's**, specifically because the paper never discounts for cross-sectional
correlation between the 50–500 stocks it evaluates jointly. This isn't "the paper did it
wrong," it's "this project holds a higher bar on one specific dimension."

## 5. Why 98,700 rows isn't 98,700 observations

Measured directly (spec 06): cross-sectional correlation **ρ = 0.289** on the adopted
50-symbol sector-stratified universe. Feeding that into the Kish effective-sample-size
formula, `n_eff = n/(1+(k-1)ρ)`, the number of *independent* observations saturates near
**~6,000** almost regardless of how many stocks are added — going from 50 to 500 symbols
(987,000 nominal rows) buys roughly 3.7% more effective information, because correlated
stocks mostly repeat information the panel already has. The binding constraint is 1,974
trading days of history, not stock count. This is *why* a 10x larger universe was
rejected as not worth its extra cost (spec 06, document 1 §22).

The consequence for interpreting any future accuracy number: with an effective test-
window N on the order of ~1,000, the standard error on accuracy is roughly ±1.6%, giving
roughly a ±3% 95% interval — beating the 51.3% constant-up baseline in a way that's
*statistically* meaningful requires something in the neighbourhood of **54.5%+**, not
merely "any number above 51.3%."

## 6. Five hard questions and honest answers

**Q1 — "So does the model work?"**
A: "I can't say yet — it's never been trained on real data. What I can say: the data
pipeline that would feed it is built and verified (98,690 real price rows, 61,791 real
filings), the architecture is parameter-count-pinned and passes 24/24 tests on synthetic
data, and the statistical gates that would decide whether any future number is
meaningful are already implemented and tested."

**Q2 — "Why hasn't it been trained yet?"**
A: "Two stages need a GPU this machine doesn't have — the LLM summarisation pass and the
actual training loop. `HANDOFF.md` is the runbook for borrowing one; it hasn't happened
yet this session."

**Q3 — "Isn't 98,700 rows already an unusually large dataset?"**
A: Use §5 directly — nominal row count is misleading here because the panel is 50
correlated stocks over a fixed history, not 98,700 independent draws; effective sample
size saturates near 6,000 no matter how many more correlated stocks are added.

**Q4 — "What happens if the shuffled-text control doesn't collapse to chance?"**
A: "That's the signal that the text stream is being exploited through a data leak rather
than through real signal — G4 exists specifically to catch that before any text-driven
result gets reported as real."

**Q5 — "How do I know the test set wasn't touched more than once?"**
A: Answer honestly using document 2 decision 19 / G7 above: the code enforces
select-on-validation/touch-test-once *within a single run*, but nothing stops re-running
the whole script and keeping the best test result across runs — that's a discipline the
people running it have to hold themselves to, not something the code can force.

## 7. A five-minute cold pitch

1. What it is (document 1 §7.1) — one sentence.
2. The task (document 1 §7.2) — binary, no threshold.
3. **Lead with what's honestly not done** (§1 above) — before the architecture, so
   nobody thinks a real number is coming next.
4. What IS live-verified — 24/24 tests, full-scale price and news data on disk.
5. The one correctness rule worth naming cold — post-close news attribution
   (document 1 §16, document 2 decision 11).
6. Close with the statistical gates (§4) as the reason a future number, once it exists,
   will actually be trustworthy — not just produced.

---

## Self-test

**Level 1.** What was the exact pytest result, and what does — and doesn't — it verify?
Name the one gate (G-number) spec 06 says code alone cannot enforce.

**Level 2.** Why does a ~6,000 effective-sample-size ceiling mean "beat the baseline by
1%" isn't a meaningful claim, but "beat it by 4%" might be?

**Level 3.** *"Just give me a number — did the retargeting to NSE data work or not?"*
Answer using §1 and §6 Q1 together: state plainly that no number exists yet, name
specifically why (GPU-dependent stages unrun), then pivot to what IS verified (pipeline
correctness, full-scale real data, statistical gates already in place) rather than
guessing at a number or apologizing for not having one.

---

This closes the FinanceProject study set. Return to [INDEX.md](INDEX.md) for the full
reading order.
