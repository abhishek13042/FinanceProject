# Session report

Unattended run, 2026-08-27 22:53 → 2026-08-28. RTX 4050 Laptop (6 GB), 5 seeds per ablation.
Six ablations completed; `layers2`, `align32`, `align64` were not started before the
10-hour deadline.

## Results

Test window 2024–2025: 464 trading days, 23,200 windows. Intervals block-bootstrapped over
trading days (spec 05 R7), not stock-days.

| Ablation | Accuracy | MCC | MCC 95% CI | min | vs `full` (G3) |
|---|---|---|---|---|---|
| `full` | 50.31% ± 0.33 | 0.010 ± 0.007 | [-0.006, 0.035] | 74 | — |
| `price_only` | 50.32% ± 0.56 | 0.010 ± 0.009 | [-0.013, 0.027] | 91 | noise |
| `shuffled_text` | 50.14% ± 0.10 | 0.007 ± 0.003 | [-0.014, 0.027] | 185 | noise |
| `concat` | 50.52% ± 0.45 | 0.013 ± 0.008 | [-0.004, 0.034] | 66 | noise |
| `oneway` | 50.09% ± 0.31 | 0.009 ± 0.007 | [-0.013, 0.025] | 88 | noise |
| `raw_text` | 50.25% ± 0.68 | 0.008 ± 0.012 | [-0.014, 0.023] | 340 | noise |

**Constant-up floor: 50.45% accuracy, 0.000 MCC.**

Every ablation overlaps every other and the floor. Per spec 06 G3, none of these differences
is reportable as a finding.

## Gates

- **Negative control passes.** `shuffled_text` lands at chance (MCC CI
  [-0.014,
  0.027], containing zero). Had it
  beaten chance, the text stream would have been leaking.
- **No ablation near 60%.** Highest is 50.52%.
- **24/24 leakage and resume guards pass** on real data.
- **The test set was evaluated once per ablation.** `run_all.py` skips any ablation already
  on disk; no ablation was re-scored.

## Is this a null result or a broken pipeline?

1. **Linear baseline matches the deep model.** Logistic regression on the same 17 price
   features and split: 49.83% acc, 0.003 MCC — indistinguishable from the 4-layer patch
   transformer. The ceiling is in the data, not the architecture.
2. **Training is healthy.** Loss 0.721 → 0.690, early stopping at varied epochs (6–18) across
   seeds, no collapse to constant prediction.
3. **The negative control behaves correctly** (above).

## Deviations from the original design, and why

Each was measured before being adopted.

1. **Summariser swapped** — `mT5_multilingual_XLSum` → `Qwen2.5-1.5B-Instruct` for generation.
   On 8 real filings mT5 named the correct company 2/8, rendering Axis Bank as "The Bank of
   England" and ITC as "the British Stock Exchange". Qwen: 8/8 with correct dates and amounts.
   Embeddings remain mT5's encoder, so `text_dim` stays 768 and nothing downstream moved.
2. **Instruction prefix removed** for seq2seq models. A fine-tuned summariser has never seen
   an instruction, so the directive is just more article — and first, so it dominates.
   Grounding in the source: 13.4% with prefix, 31.1% without.
3. **Empty-body filings dropped** (`min_body_chars: 30`). 60.2% of the post-boilerplate corpus
   had no body; `"Announcement under Regulation 30 (LODR)-Press Release / Media Release. A."`
   appeared 3,969 times, and identical stubs produced identical fabrications across unrelated
   stocks.
4. **Coverage is 9.5% of stock-days, not 23.4%.** The old figure counted stub filings.
5. **Post-close news sorted by timestamp, not bucket id.** Attribution already moves it to the
   next session, where it is that day's earliest information; the bucket sort filed it last.
   Most BSE filings are post-close.
6. **Per-ablation checkpoint names.** Previously every ablation wrote `best_seed<N>.pt`;
   identical-shaped ablations could silently load each other's weights.
7. **Seed-level resume** (`artifacts/seed_<tag>_<seed>.npz`). An interrupted ablation no longer
   discards the seeds it already finished.

## The honest limitation

The real repair is fetching the BSE PDF attachments holding the missing bodies —
`news_sources.py` already captures `ATTACHMENTNAME` and then drops the column. That would
restore coverage toward 23.4% with real text rather than stubs. It is a multi-hour re-scrape
plus PDF extraction and was out of scope for a borrowed GPU.

Until then, `full` tying `price_only` means **"text did not help on the 9.5% of stock-days
where text actually exists"** — narrower than "news does not predict Indian equities".

## Not run

`layers2`, `align32`, `align64` — capacity/width variants. The linear-baseline check
identified model capacity as not being the bottleneck, so these were deprioritised against the
10-hour deadline. `python scripts/run_all.py --only layers2,align32,align64` resumes them;
everything already on disk is skipped. Estimated ~10.5 h combined.
