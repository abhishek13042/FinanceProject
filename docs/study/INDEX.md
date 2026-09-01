# FinanceProject — Study Set

A beginner-level companion to the FinanceProject codebase: a replication of "Integrative
stock price trend prediction via hierarchical LLM text processing and patch-based
transformer with co-attention" (Zhang, Dong & Xu, ESWA 302, 2026), retargeted from the
paper's US/China news corpora to 50 sector-stratified NSE (Indian) large-cap equities,
2018–2025.

## Reading order

| # | Document | Covers | Time |
|---|---|---|---|
| 0 | [00-how-to-study-this.md](00-how-to-study-this.md) | Study method, reading order, four traps, five things to say cold | 10 min |
| 1 | [01-concepts-from-zero.md](01-concepts-from-zero.md) | Every term the codebase assumes — architecture, India-specific adaptations, statistical apparatus | 1.5–2 hrs |
| 2 | [02-one-request-explained-slowly.md](02-one-request-explained-slowly.md) | One stock-day's journey end to end, traced through the real files | 1.5–2 hrs |
| 3 | [03-the-data-model-explained-slowly.md](03-the-data-model-explained-slowly.md) | Data contracts, the 14-guard leakage-test suite, three real incidents the tests caught | 1–1.5 hrs |
| 4 | [04-results-and-how-to-defend-them.md](04-results-and-how-to-defend-them.md) | The measured result — a defensible null at the constant-up floor, three checks ruling out a broken pipeline, and the coverage caveat that narrows the claim | 45–60 min |

**Total: roughly 4–5.5 hours** for a thorough first pass.

## In one paragraph

Eight years of NSE price history and 61,791 timestamped BSE corporate filings feed a
four-stage pipeline: hierarchical news summarisation (Qwen2.5-1.5B-Instruct generation,
mT5-encoder embedding), a cross-attention alignment module that squeezes 768-d text
embeddings against 17-channel price data, overlapping-patch transformer encoders per
modality, and a symmetric bidirectional co-attention fusion before a linear next-day-
direction classifier (~1.25M trainable parameters). The full pipeline has now run end to
end on a borrowed GPU: six ablations, five seeds each, all landing at the constant-up
floor (~50.3% accuracy, MCC ≈ 0.01) — a defensible null result, checked against a
negative control, a linear baseline, and healthy training curves, narrowed by one
caveat: real news text covered only 9.5% of stock-days, far below the ~23% assumed at
design time. Roughly half of the project's own specs exist purely to make sure that
result is statistically meaningful, not an artifact of 50 correlated stocks being
mistaken for 98,700 independent observations.

Start at [00-how-to-study-this.md](00-how-to-study-this.md).
