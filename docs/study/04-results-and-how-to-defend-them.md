# Study Guide 4 — The result, and how to defend it

Documents 1–3 covered the design. This one covers **evidence**: what was actually run,
what it found, and how to talk about a null result honestly instead of apologetically.

**This document was rewritten after the study set was first drafted.** The first draft
said no trained-model result existed yet. Between that draft and this one, a teammate
(Adarsh T U) ran the full spec 03→05 pipeline end to end on a borrowed RTX 4050 — commit
`0d40f8d`, "Run spec 03->05 on real data" — and the result changes what this document
needs to say. Read this version.

Read after document 3. Time: 45–60 minutes.

---

## 1. The headline fact, stated first, not buried

**The measured result is a null.** Every ablation lands at the constant-up floor
(~50.3% accuracy, MCC ≈ 0.01), statistically indistinguishable from every other ablation
and from a model that always predicts "up." This is a real, defensible finding — not a
missing one — and the honest way to state it is exactly that plainly, not hedged.

## 2. The numbers — `artifacts/SESSION_REPORT.md`, run 2026-08-27/28

RTX 4050 Laptop (6 GB), 5 seeds per ablation, test window 2024–2025 (464 trading days,
23,200 windows). Intervals block-bootstrapped over **trading days** (spec 05 R7, document
1 §24), not stock-days.

| Ablation | Accuracy | MCC | MCC 95% CI | vs `full` (G3) |
|---|---|---|---|---|
| `full` | 50.31% ± 0.33 | 0.010 ± 0.007 | [-0.006, 0.035] | — |
| `price_only` | 50.32% ± 0.56 | 0.010 ± 0.009 | [-0.013, 0.027] | noise |
| `shuffled_text` | 50.14% ± 0.10 | 0.007 ± 0.003 | [-0.014, 0.027] | noise |
| `concat` | 50.52% ± 0.45 | 0.013 ± 0.008 | [-0.004, 0.034] | noise |
| `oneway` | 50.09% ± 0.31 | 0.009 ± 0.007 | [-0.013, 0.025] | noise |
| `raw_text` | 50.25% ± 0.68 | 0.008 ± 0.012 | [-0.014, 0.023] | noise |

**Constant-up floor: 50.45% accuracy, 0.000 MCC.** Under spec 06 G3 (document 3 §2, G3 —
non-overlapping confidence intervals required for any comparison claim), none of these
six numbers can be reported as beating any other. `layers2`, `align32`, `align64`
(capacity-variant ablations) were not run before the 10-hour session deadline; the
linear-baseline check below is why they were deprioritised rather than treated as
missing evidence.

## 3. Is this a null result or a broken pipeline? — three independent checks

A null is only worth reporting if the pipeline can be shown to actually work. This is the
question the leakage-guard suite (document 3) exists to let someone answer with evidence,
not reassurance:

1. **The negative control passes.** `shuffled_text`'s MCC CI is `[-0.014, 0.027]` —
   contains zero, i.e. lands at chance. This is the correct outcome for G4 (document 3
   §2); had it beaten chance, that would mean the text stream was leaking rather than
   informing (document 2, decision 20's rationale, exercised for real here).
2. **A linear baseline matches the deep model exactly.** Logistic regression on the same
   17 price features, same split: 49.83% accuracy, 0.003 MCC — indistinguishable from the
   4-layer patch transformer. The ceiling is in the *data*, not in model capacity. This is
   also why the capacity-variant ablations (`layers2`, `align32`, `align64`) were skipped
   — a linear model already matching the deep one is direct evidence more capacity was
   not the bottleneck being tested.
3. **Training is healthy.** Loss fell smoothly (0.721 → 0.690 across seeds), early
   stopping fired at varied epochs (6–18) rather than collapsing immediately, no seed
   collapsed to a constant prediction, and 24/24 leakage and resume guards (document 3)
   pass on the real data used for this run.

Together: this is a genuine absence of predictive signal under the conditions tested, not
an artifact of a broken pipeline.

## 4. The caveat that matters most — text coverage was much lower than assumed

**Real text exists on only 9.5% of stock-days, not the 23.4% originally assumed** (the
figure `HANDOFF.md` had carried before this run). The cause: BSE serves most Regulation 30
filings as a category label plus a PDF attachment; `collect_news.py` captures the
attachment's *name* but never fetches its content, so 60.2% of the post-boilerplate corpus
had an empty body. One string — `"Announcement under Regulation 30 (LODR)-Press Release /
Media Release. A."` — appeared **3,969 times**, and identical empty stubs produced
identical *invented* summaries across unrelated stocks (see §5) — a constant injected into
the text stream, not signal. These rows are now filtered (`min_body_chars: 30`).

**This narrows the honest claim.** `full` tying `price_only` means *"text did not help on
the 9.5% of stock-days where real text actually exists,"* not the broader "news doesn't
predict Indian equities." The real repair — fetching the BSE PDF attachments
`news_sources.py` already captures the name of but discards — would restore coverage
toward the original 23.4% estimate with genuine text rather than stubs, and was out of
scope for one borrowed GPU session (`SESSION_REPORT.md`'s own "honest limitation" section).

## 5. A real incident that changed the pipeline mid-session — the summariser swap

`mT5_multilingual_XLSum` (the summariser named throughout documents 1–3 as the India-
appropriate multilingual choice) was **replaced mid-session** after the preflight gate
(spec 03 R7, document 1) caught it inventing content rather than summarising it. Measured
on 8 real filings: mT5 named the correct company **2/8**, rendering Axis Bank as "The Bank
of England," ITC as "the British Stock Exchange," and misattributing a Cipla allotment to
"the Financial Conduct Authority." Generation now uses **Qwen2.5-1.5B-Instruct**: 8/8
correct, with accurate dates and share counts.

**What did and didn't change downstream:** embeddings still come from the mT5 encoder
(document 2 §B.4's pooling code, decision 13, is unaffected), so `text_dim` stays 768 and
nothing past the summarisation stage moved — the two models are never resident on the GPU
at once. A second fix landed alongside it: instruction prefixes were removed from the
seq2seq prompts, because a fine-tuned summariser has never seen an instruction and treated
the directive as just more article to summarise — measured grounding-in-source rose from
13.4% (with the prefix) to 31.1% (without).

This is exactly the kind of finding the preflight gate (spec 03 R7) exists to catch
*before* GPU time is spent on a full run — and here it caught a real defect, not a
hypothetical one, mid-session.

## 6. What the on-disk evidence actually is

| Stage | Evidence | Status |
|---|---|---|
| Price data | `data/interim/prices.csv` | executed, full scale |
| News collection | `data/raw/news.csv` | executed, full scale |
| Summarisation | `data/interim/daily_summaries.csv` (Qwen-generated), embeddings via mT5 encoder | **executed** — 9,424 stock-days summarised |
| Training | `artifacts/results_{full,price_only,shuffled_text,concat,oneway,raw_text}.json`, `artifacts/SESSION_REPORT.md` | **executed** — 6 ablations × 5 seeds |
| Leakage/resume tests | `pytest tests/` | **24 passed, 0 failed** (verified live this session, twice, before this run's commit was pulled in) |

`layers2`, `align32`, `align64` remain unrun — the one honest gap left in the ablation
matrix, deprioritised (not abandoned) once the linear-baseline check (§3.2) showed
capacity wasn't the limiting factor being probed.

## 7. Five hard questions and honest answers

**Q1 — "So does the model beat the baseline?"**
A: "No — every ablation, including the full model, sits within noise of the constant-up
floor (50.3–50.5% accuracy, MCC ≈ 0.01). That's not a partial result; it's the finding."

**Q2 — "Could this just be a bug?"**
A: Use §3 — negative control at chance, linear baseline matching the deep model exactly,
healthy training curves, 24/24 leakage guards passing. Three independent checks, not one,
all consistent with "real absence of signal" rather than "broken pipeline."

**Q3 — "Why didn't the news help, when the paper claims it does?"**
A: Use §4 — real text existed on only 9.5% of stock-days, not the ~23% assumed when the
project was designed; most of the announced corpus was empty boilerplate stubs, now
filtered out. The honest claim is narrower than "news doesn't predict Indian stocks" — it's
"news didn't help on the fraction of days it was actually present and non-degenerate."

**Q4 — "Why change the summariser mid-project?"**
A: Use §5 — the originally-planned mT5 model was caught inventing facts at the preflight
gate (2/8 correct company names on real filings), and was swapped for one that scored 8/8
before any GPU time went into a full run.

**Q5 — "What would change this result?"**
A: The real repair identified in `SESSION_REPORT.md`: fetch the BSE PDF attachments that
`news_sources.py` already names but discards, which would raise real (non-stub) coverage
back toward the original ~23% estimate. Until that's done, this null result describes a
9.5%-coverage regime, not a full-coverage one.

## 8. A five-minute cold pitch

1. What it is (document 1 §7.1) — one sentence.
2. The task (document 1 §7.2) — binary, no threshold.
3. **Lead with the result** — a defensible null: every ablation at the constant-up floor,
   confirmed by three independent robustness checks (§3), not a missing result.
4. The one caveat that narrows the claim — 9.5% real text coverage, not ~23% (§4).
5. The one correctness rule worth naming cold — post-close news attribution (document 1
   §16, document 2 decision 11) — still true, unaffected by this run.
6. Close with what would change the finding — attachment-PDF fetching (§7 Q5).

---

## Self-test

**Level 1.** State the six ablations' approximate accuracy range and the constant-up
floor. Name the three checks used to rule out a broken pipeline.

**Level 2.** Why does "text existed on only 9.5% of stock-days" narrow the claim from
"news doesn't predict Indian equities" to something more specific? What single defect did
the preflight gate catch in the originally-planned summariser?

**Level 3.** *"Just give me a number — did the retargeting to NSE data work or not?"*
Answer using §1 and §7 Q1: yes, it produced a result — a defensible null, not "no data" —
then immediately name the one caveat (§4) that keeps the finding from being read more
broadly than the evidence supports.

---

This closes the FinanceProject study set. Return to [INDEX.md](INDEX.md) for the full
reading order.
