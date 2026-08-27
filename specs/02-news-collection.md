# Spec 02 — News collection

**Status: built, resumable, and EXECUTED at full scale.** `src/data/news_sources.py`, `scripts/collect_news.py`.

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
   plan around them. **Dead weight for the current run:** the config window ends 2025-12-31
   and today is later than that, so every split including test is already in the past. RSS
   can add nothing to this corpus; it only matters if the window is extended forward.
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
8. **R8 — The backfill must survive the connection.** *(Built.)* The 2018-2025 backfill is
   roughly an hour of paginated requests against an unauthenticated API, on a link that
   drops. Three properties, in `src/data/news_sources.py`:
   - **Per-symbol checkpoint.** Each symbol's rows are appended to
     `data/raw/news.bse.partial.csv` and its name to `news.bse.done.txt`, both `fsync`'d
     before the next symbol starts. Rows are written *before* the done-marker, so a crash
     between the two re-fetches a symbol rather than losing it — and R6's dedup absorbs it.
   - **Resume, not restart.** `collect_bse` skips symbols already in the done-file. Re-running
     the identical command continues. `collect_news.py merge` folds the checkpoint into
     `news.csv` with no network at all.
   - **Retry, and the right thing not retried.** Each HTTP GET retries four times with 3/6/12s
     backoff. A network failure leaves the symbol *unmarked* so the next resume retries it; a
     missing scrip code is marked done, because it will never resolve itself.

   The scrip master is cached to `data/raw/bse_scrip_map.csv`, so a resume does not need a
   multi-megabyte download before it can reach the symbol it meant to continue from.

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
- [x] Coverage measured on the **full corpus** (61,791 announcements, 2018-2025, 50 symbols):
      **38.1% of stock-days** carry >=1 announcement; 1.64 articles per covered day; a 30-day
      window holds a median of 11 news days, and **2.91% of windows are entirely text-free**.

      These supersede an earlier reading of 64.8% coverage and "zero empty windows", which was
      taken from 5 mega-caps in H1-2024. Those names file almost daily; the universe does not.
      Per-symbol coverage ranges 20.3% to 61.5%, and rises over time (21.9% in 2018 to 48.8%
      in 2024) as disclosure requirements tightened. The honest number is 38.1%.
- [x] **R9 — Administrative filings are dropped before summarisation.** *(Built.)* 42.1% of
      the corpus (26,008 of 61,791) is paperwork no price can react to: lost share
      certificates, newspaper-publication copies, trading-window closures, scheduled
      investor-meet notices. `news_sources.BOILERPLATE` is a flat headline regex list applied
      in `build_text.load_news`, switchable via `data.filter_boilerplate` so filtered-vs-raw
      stays a cheap ablation rather than a baked-in choice.

      After filtering, the honest density is **23.4% of stock-days**, 1.55 filings per covered
      day, a median of **6 news days per 30-day window**, and **3.86% of windows entirely
      text-free**. This — not 38.1% — is the number the text stream actually has to work with.
- [x] Scrip mapping covers 49/50 of the bundled universe.
- [x] RSS parses with `+0530` timestamps and tags conservatively (~1 in 50 items names a
      specific stock — low yield is correct behaviour, not a bug).
- [x] Backfill is crash-resumable: 8 offline tests in `tests/test_resume.py` kill a run
      partway through and assert what survived.
- [x] Full 2018-2025 backfill across the universe completes: 61,791 rows, 50/50 symbols,
      2018-01-01 to 2025-12-31, one 52-minute run.
- [x] Zero tz-naive timestamps in `news.csv`; zero unparseable, zero empty headlines.

## Known issues

- `LTIM` does not map to a BSE scrip id; skipped with a warning — and now marked done, so
  resumes stop re-walking it.
- GDELT (free, 2018+, 15-min timestamps) timed out twice from this machine. Worth retrying
  as a coverage supplement — it would add press coverage, whereas BSE only gives filings.

## Open question

BSE filings are *regulatory disclosures*, not *market commentary*. The paper's corpus is news
headlines. These are different text distributions, and the hierarchical summariser's prompts
are currently written for the latter. Whether filings alone carry the same signal is an
empirical question for spec 05, and a genuine threat to comparability with the paper.
