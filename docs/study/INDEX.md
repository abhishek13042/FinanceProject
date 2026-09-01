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
| 4 | [04-results-and-how-to-defend-them.md](04-results-and-how-to-defend-them.md) | What's actually been run (24/24 tests, live-verified), what hasn't (no trained-model result yet), and the statistical gates that will govern any future number | 45–60 min |

**Total: roughly 4–5.5 hours** for a thorough first pass.

## In one paragraph

Eight years of NSE price history and 61,791 timestamped BSE corporate filings feed a
four-stage pipeline: frozen-mT5 hierarchical news summarisation, a cross-attention
alignment module that squeezes 768-d text embeddings against 17-channel price data,
overlapping-patch transformer encoders per modality, and a symmetric bidirectional
co-attention fusion before a linear next-day-direction classifier (~1.25M trainable
parameters). The price and news pipelines are built and executed at full scale; the
summarisation and training stages are built and tested but have never run against real
data, because they need a GPU this project doesn't have locally. Roughly half of the
project's own specs exist purely to make sure that whatever accuracy number eventually
comes out is statistically meaningful, not an artifact of 50 correlated stocks being
mistaken for 98,700 independent observations.

Start at [00-how-to-study-this.md](00-how-to-study-this.md).
