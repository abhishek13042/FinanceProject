# Spec 03 — Hierarchical summarization

**Status: implemented incl. R6 resume, NOT yet executed.** `src/summarize/hierarchical.py`,
`src/summarize/runner.py`, `scripts/build_text.py`. The mT5 model has never been downloaded
or run in this project. Everything below the acceptance line is therefore unverified.

Implements the paper's Section 3.2 / Algorithm 1.

## Contract

- **In:** `data/raw/news.csv` (spec 02) + trading calendar from `prices.csv` (spec 01).
- **Out:** `data/interim/daily_summaries.csv` (`symbol, date, summary`) and
  `data/interim/daily_embeddings.npz` (`keys` as `"SYMBOL|YYYY-MM-DD"`, `vecs` float32 `[n,768]`).
- **Also out:** `daily_summaries_raw.csv` / `daily_embeddings_raw.npz` — the same stock-days
  with headlines concatenated instead of summarised. Spec 05's "no hierarchical summary"
  ablation needs these, and producing them in the same sitting avoids a second GPU session.
- **Intermediate:** `daily_summaries.partial.csv` — append-only resume checkpoint (R6).
  Safe to delete only after `daily_summaries.csv` exists and its row count matches.

## Requirements

1. **R1 — Two passes, frozen model.** Pass 1 summarises each intraday bucket; pass 2
   summarises the concatenated bucket summaries into one daily synopsis. The LLM is never
   fine-tuned — only stages 2–4 train.
2. **R2 — Session buckets, not clock hours.** The paper's "hourly" segmentation becomes NSE
   session buckets: pre-open (<09:15), six intraday hours, post-close (≥15:30).
3. **R3 — Multilingual by default.** `mT5_multilingual_XLSum` covers English plus Hindi,
   Bengali, Tamil, Telugu, Marathi, Gujarati, so vernacular wires need no model swap.
4. **R4 — Indian catalyst prompts.** Prompts name the catalysts that actually move NSE
   names: earnings, order wins, SEBI/RBI action, promoter pledging, FII/DII flows, guidance,
   analyst actions.
5. **R5 — Embeddings, not text, cross the boundary.** Only the pooled encoder vector is
   forwarded downstream, per the paper's token-budget argument.
6. **R6 — Cache and resume.** A full backfill is many GPU-hours. Summarisation must be
   restartable without recomputing completed stock-days. **Built** — `runner.run_summarization`
   chunks work by stock-day, fsyncs an append-only checkpoint after every chunk, and on
   restart skips whatever `completed_units()` finds on disk. `--chunk` sets the amount of
   work at risk from a single crash. `finalize()` de-duplicates (last write wins) before
   committing the output.
7. **R7 — Preflight before booking GPU time.** `build_text.py preflight` loads the model,
   summarises a handful of stock-days, and reports: corpus coverage, seconds per stock-day,
   an extrapolated GPU-hour estimate for the full backfill, three input/output samples, a
   verbatim-copy warning, the all-zero-row count, and the emitted key format. This is the
   cheap check that catches the degenerate-filing risk below *before* the borrowed laptop
   is in front of you.
8. **R8 — Text crosses the boundary, not just vectors.** Summaries are committed as CSV
   text. Embedding is a separate `embed` mode that runs acceptably on CPU, so scarce GPU
   time is spent only on generation and a re-embed never needs the borrowed machine.

## Acceptance criteria

- [ ] mT5 loads and generates on this machine (CPU fallback works but will be slow).
- [ ] Pass-1 summaries are shorter than their inputs and are not verbatim copies.
- [ ] Embedding matrix has no all-zero rows for covered stock-days.
- [ ] Key format round-trips exactly against `StockWindowDataset`'s `(symbol, date_str)` lookup.
- [ ] Uncovered stock-days (≈35%, spec 02) resolve to the zero vector, not to a crash.
- [x] Resume verified **against a stub summariser** (90 stock-days, crash at 40): restart
      regenerated exactly the missing 50, final output 90 unique rows, raw-variant keys
      align exactly with summary keys. Still to be re-checked once on the real mT5 run.
- [ ] Raw-headline variant produced alongside the summaries in the same sitting.

## Risks

- **Cost is the main risk.** Two LLM passes over 8 years × N symbols. At 50 symbols and 65%
  coverage that is ~64k daily summaries plus a larger number of bucket summaries. On CPU this
  is not feasible; budget GPU time or cut the universe for a first run.
- ~~R6 is not built yet.~~ **Resolved.** Resume is implemented; a crash now costs at most
  one `--chunk` of work.
- **Summarising a regulatory filing may be degenerate.** XLSum is trained on news prose; a
  compliance disclosure may summarise to near-boilerplate. If pass-1 output is largely
  identical across filings, the text stream carries no differential signal and spec 05's
  text ablation will (correctly) show nothing. Inspect a sample before committing GPU hours.
