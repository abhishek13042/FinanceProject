# Study Guide 0 — How to study this project

This is a research replication: **"Integrative stock price trend prediction via
hierarchical LLM text processing and patch-based transformer with co-attention"**
(Zhang, Dong & Xu, ESWA 302, 2026), retargeted from the paper's US/China news corpora to
Indian (NSE) equities. Read this document first — ten minutes — before any code.

---

## 1. The standard

**You have understood something when you can explain it, from memory, to someone who does
not know it — and survive their follow-up question.**

## 2. The method

1. Read one section.
2. Close it and explain it **out loud**, from memory.
3. Notice exactly where you go vague — that vagueness is the gap, not a style problem.
4. Go back to that specific gap, not the whole section.

## 3. The documents, in order

| Read | Document | Why here |
|---|---|---|
| 1st | `01-concepts-from-zero.md` | Every term the codebase assumes — the paper's architecture, the statistical apparatus, and the India-specific pipeline, in dependency order |
| 2nd | `02-one-request-explained-slowly.md` | One stock-day's journey end to end: raw price + news → windowed sample → prediction, traced through the real files |
| 3rd | `03-the-data-model-explained-slowly.md` | The data contracts between every stage, the leakage-guard test suite, and a real bug this project's own tests caught |
| 4th | `04-results-and-how-to-defend-them.md` | The measured result (a null, at the constant-up floor), the three checks that rule out a broken pipeline, and the coverage caveat that narrows the claim |

**Total: roughly 4–5.5 hours** for a thorough first pass.

## 4. The one thing that makes this project different from a typical ML repo

Most ML projects treat "get a number out" as the hard part and "is the number real" as an
afterthought. **This project inverts that.** Four of its eight specs (`05` training, `06`
statistical validity, `07` leakage guards, and large parts of `01`–`03`) exist purely to
make sure that whatever number eventually comes out means what it claims to mean. Read
[specs/06-statistical-validity.md](../../specs/06-statistical-validity.md) with that in
mind: it exists because 98,700 rows of stock-day data are not 98,700 independent
observations — stocks move together — and a project that doesn't account for that will
report "significant" results that are actually noise.

**The second thing:** the headline result is a **null**, and it's a real one. All eight
specs have now run end to end on real data (a borrowed-GPU session, commit `0d40f8d`) —
every ablation lands at the constant-up floor (~50.3% accuracy, MCC ≈ 0.01), verified by
three independent checks (negative control at chance, a linear baseline matching the deep
model, healthy training curves) rather than assumed. If asked "what's the accuracy," the
honest answer is "no ablation beats the baseline, and here's why that's a defensible
finding, not a missing one" — document 4 covers this in full, including the one caveat
(real news text covered only 9.5% of stock-days, not the ~23% assumed at design time)
that narrows the claim.

## 5. Four traps

**Trap 1 — treating this as an ordinary supervised-learning project.** The model
(`src/models/model.py`) is a small piece of this codebase. The statistical apparatus
around it (`src/eval.py`, the eight specs) is at least as large and is the part that
determines whether any result is defensible.

**Trap 2 — reading numbers pre-registered before the real run as if they were the
outcome.** `HANDOFF.md`'s original coverage estimate (23.4%) and the paper's 53–57%
accuracy expectation were budgeting/hope figures, not measurements. The real run measured
9.5% coverage and a null result — document 3 §1.1 covers the price-channel-count
correction (`d_p=17`, now also fixed in the README itself), and document 4 covers the
result correction in full.

**Trap 3 — assuming a null result means the pipeline didn't work.** It ran completely —
6 ablations, 5 seeds each, 24/24 leakage guards passing on real data — and three
independent checks (document 4 §3) rule out a broken pipeline as the explanation for the
null. A working pipeline that finds no signal is a different, more defensible claim than
"this was never tested."

**Trap 4 — skipping the test files.** `tests/test_leakage.py` and `tests/test_resume.py`
are not incidental — spec 07 exists specifically because "a leak also flatters" (its own
words): a broken pipeline that leaks future information produces a *better-looking*
result, not a worse one, so nothing in a shipped number will look wrong on its own.

## 6. Answering a question you cannot answer

1. Say the boundary: "I don't know that one."
2. Say what's adjacent: "I know the price pipeline is fully built and executed; I haven't
   traced the exact RSI/MACD formulas in `_indicators`."
3. Say where to look: "`src/data/prices.py`, `_indicators` — thirty seconds."

## 7. Five things to say cold

1. **What it is.** "A replication of a 2026 ESWA paper — hierarchical LLM news
   summarisation fused with a patch-transformer on price data, for next-day stock
   direction — retargeted from US/China data to 50 NSE large-caps."
2. **The task.** "Binary: did tomorrow's close beat today's? No magnitude threshold."
3. **The architecture in one sentence.** "Frozen mT5 summarises news into daily
   embeddings offline; a small (~1.25M-parameter) trainable transformer patches 30-day
   price and text windows and fuses them with symmetric co-attention before a linear
   classifier."
4. **The single most important correctness rule.** "News filed after the 15:30 IST close
   is attributed to the next trading session, not the same one — most BSE filings land
   after close, so getting this backwards would leak tomorrow's catalyst into today."
5. **The result.** "A defensible null — every ablation lands at the constant-up floor,
   confirmed by a negative control at chance and a linear baseline matching the deep
   model, with one caveat: real news text covered only 9.5% of stock-days, so the
   narrower honest claim is 'text didn't help where it existed,' not 'news doesn't
   matter.'"

## 8. The whole project in one breath

> Eight years of NSE price history (2018–2025, 50 sector-stratified names) and 61,791
> timestamped BSE corporate filings get cleaned, session-bucketed, and summarised two-pass
> (Qwen2.5-1.5B-Instruct generation, mT5-encoder embedding) into one 768-d daily embedding
> per stock-day → a 30-day window of 17 price channels and 30 daily text embeddings gets
> aligned, patched, and fused by a small co-attention transformer into one next-day-
> direction logit → run for real across six ablations and five seeds each, every one lands
> at the constant-up floor (~50.3% accuracy, MCC ≈ 0.01) — a defensible null, checked
> against a negative control, a linear baseline, and healthy training curves, narrowed by
> one caveat: real news text covered only 9.5% of stock-days, far below the ~23% assumed
> when the project was designed.

---

## Self-test

**Level 1.** State the four-step method. Name the two specs that exist purely to protect
result validity rather than to produce the result itself.

**Level 2.** Why does §4 say this project "inverts" the usual ML-repo priority order? What
concretely is missing from `artifacts/` right now, and why?

**Level 3.** *"So what's the model's accuracy?"* Answer using §4/§7.5 directly: every
ablation lands at the constant-up floor (~50.3%, MCC ≈ 0.01) — a defensible null, verified
by a negative control, a linear baseline, and healthy training curves rather than assumed
— then name the one caveat that narrows the claim: real news text covered only 9.5% of
stock-days, so the honest reading is "text didn't help where it existed," not "news
doesn't matter for Indian equities."
