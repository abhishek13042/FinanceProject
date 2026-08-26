# Spec 02 — News collection

**Status: built and live-verified.** `src/data/news_sources.py`, `scripts/collect_news.py`.

The paper consumes CMIN, a ready-made timestamped news corpus. No Indian equivalent exists,
so this spec exists entirely to manufacture one.

## Contract

- **In:** universe symbol list; `data/interim/prices.csv` (for the trading calendar).
- **Out:** `data/raw/news.csv` — columns `symbol, timestamp, headline, body`.
  `timestamp` is ISO8601 **with offset**; tz-naive rows are invalid.

## Requirements

1. **R1 — Timestamps are the product.** An article without a real publication time is
   unusable: the two-pass summarisation has nothing to bucket. Reject such rows at ingest
   rather than defaulting them to midnight.
2. **R2 — BSE announcements are the historical backbone.** Free, unauthenticated,
   exchange-official, exact IST filing times, queryable to ~2018. Symbol mapping is automatic
   via the BSE scrip master.
3. **R3 — RSS is forward-only.** ET / Business Standard / Moneycontrol accumulate coverage
   from the day collection starts. Schedule every 30–60 min. Back-archives are thin; do not
   plan around them.
4. **R4 — Next-session attribution.** News timestamped at or after the 15:30 IST close is
   attributed to the **next** trading session. This is the single most important correctness
   rule in the project — see the measurement below.
5. **R5 — Conservative symbol tagging.** RSS feeds are market-wide. Attribute an article to a
   symbol only on a whole-word company-name match. A false attribution injects a competitor's
   news into this stock's daily summary, which is worse than no news at all.
6. **R6 — Idempotent collection.** Re-running a collector must not duplicate rows.
   Deduplication key: `(symbol, timestamp, headline)`.
7. **R7 — Wire dedup.** Near-identical syndicated stories on the same stock-day collapse to
   the earliest-timestamped version at >90% character overlap.

## The measurement that drives R4

Filing-hour distribution, Q1-2024, RELIANCE/TCS/INFY:

```
09–15 IST (market open):   ~35 filings
16–20 IST (after close):  ~145 filings
```

Most Indian corporate announcements are filed after the close. Same-day attribution — the
naive reading of the paper — would leak tomorrow's catalyst into today's features across the
majority of the corpus, and would inflate accuracy in the flattering direction.

## Acceptance criteria

- [x] BSE backfill returns real timestamped rows (verified: 700 announcements, 5 symbols,
      H1-2024).
- [x] Coverage measured: **64.8% of stock-days** carry ≥1 announcement; mean 1.8 articles per
      covered day; **every** 30-day window contains 15–26 news days, zero empty windows.
- [x] Scrip mapping covers 49/50 of the bundled universe.
- [x] RSS parses with `+0530` timestamps and tags conservatively (~1 in 50 items names a
      specific stock — low yield is correct behaviour, not a bug).
- [ ] Full 2018–2025 backfill across the universe completes.
- [ ] Zero tz-naive timestamps in the committed `news.csv`.

## Known issues

- `LTIM` does not map to a BSE scrip id; skipped with a warning.
- GDELT (free, 2018+, 15-min timestamps) timed out twice from this machine. Worth retrying
  as a coverage supplement — it would add press coverage, whereas BSE only gives filings.

## Open question

BSE filings are *regulatory disclosures*, not *market commentary*. The paper's corpus is news
headlines. These are different text distributions, and the hierarchical summariser's prompts
are currently written for the latter. Whether filings alone carry the same signal is an
empirical question for spec 05, and a genuine threat to comparability with the paper.
