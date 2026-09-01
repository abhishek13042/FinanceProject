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
| 4th | `04-results-and-how-to-defend-them.md` | What has actually been run, what hasn't, the live-verified test result, and the statistical gates that decide whether any future number is reportable |

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

**The second thing:** as of this session, **the headline result does not exist yet.**
Specs 01, 02 and 04 are built and executed. Spec 03 (the LLM summarisation stage) has
never been run — it needs a GPU this project doesn't have on its own machine. Read
[HANDOFF.md](../../HANDOFF.md) to understand why: it's a runbook written *for a borrowed
GPU session*, not documentation of something already done. If asked "what's the accuracy,"
the honest answer is "not measured yet" — say that plainly, per document 4.

## 5. Four traps

**Trap 1 — treating this as an ordinary supervised-learning project.** The model
(`src/models/model.py`) is a small piece of this codebase. The statistical apparatus
around it (`src/eval.py`, the eight specs) is at least as large and is the part that
determines whether any result is defensible.

**Trap 2 — reading the top-level `README.md` as fully current.** It's a good overview, but
one number in it (price channel count, "d_p=5") is stale against the actual config and
spec 01 (`n_price_feats: 17`) — document 3 §2 covers this directly. Trust the specs and the
config over the README's prose summary when they disagree.

**Trap 3 — assuming "58 CPU-hours per ablation" means this has been run 8 times.** It
hasn't. `HANDOFF.md`'s cost estimates are pre-registered budgeting for a session that, as
of this document, has not yet happened. `artifacts/` is empty.

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
5. **What's honestly not done yet.** "The LLM summarisation stage and every training run
   depend on a GPU this project doesn't have locally — say that first, not last, if asked
   for a result."

## 8. The whole project in one breath

> Eight years of NSE price history (2018–2025, 50 sector-stratified names) and 61,791
> timestamped BSE corporate filings get cleaned, session-bucketed, and — once a GPU
> session runs it — summarised two-pass by a frozen multilingual LLM into one 768-d daily
> embedding per stock-day → a 30-day window of 17 price channels and 30 daily text
> embeddings gets aligned, patched, and fused by a small co-attention transformer into one
> next-day-direction logit → and no number from that pipeline is reportable until it
> clears five seeds, day-level block-bootstrap intervals, a shuffled-text negative control,
> and a market-agreement split — because 50 correlated stocks over 1,974 days carry far
> fewer independent observations than their row count suggests.

---

## Self-test

**Level 1.** State the four-step method. Name the two specs that exist purely to protect
result validity rather than to produce the result itself.

**Level 2.** Why does §4 say this project "inverts" the usual ML-repo priority order? What
concretely is missing from `artifacts/` right now, and why?

**Level 3.** *"So what's the model's accuracy?"* Answer using §4/§7.5 directly: no number
exists yet — spec 03 (summarisation) has never been run because it needs a GPU this
machine doesn't have; say what *is* built (specs 01, 02, 04) before saying what isn't, and
name `HANDOFF.md` as the plan for closing that gap rather than a report of having done so.
