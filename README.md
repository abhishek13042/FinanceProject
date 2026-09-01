# Hierarchical-LLM + Patch Transformer + Co-Attention — Indian Market (NSE)

Re-implementation of Zhang, Dong & Xu, *"Integrative stock price trend prediction via
hierarchical LLM text processing and patch-based transformer with co-attention"*
(Expert Systems With Applications 302, 2026), retargeted from CMIN-US/CMIN-CN to NSE equities.

Task: binary next-day direction. `y(τ+1) = 1` iff `close(τ+1) > close(τ)`, no magnitude threshold.

## Specs

The project is split into six specs in [`specs/`](specs/README.md) — read that index first.
Each spec owns one module, states its input/output contract, and carries its own acceptance
criteria and known issues.

| # | Spec | Status |
|---|---|---|
| 00 | System overview | contract fixed |
| 01 | Price data | built, live-verified |
| 02 | News collection | built, live-verified |
| 03 | Hierarchical summarization | **executed** — 9,424 stock-days summarised |
| 04 | Model | built, smoke-tested |
| 05 | Training & evaluation | **executed** — 6 ablations, 5 seeds each |
| 06 | Statistical validity | gates defined, binding on 05 |

Critical path to a first real result: **03 → 05**, gated by **06**. Executed 2026-08-27/28 on an RTX 4050; see [Results](#results).

## Architecture (unchanged from the paper)

| Stage | Module | File |
|---|---|---|
| 1 | Hierarchical summary: intraday-bucket pass → daily pass (frozen Qwen2.5-1.5B-Instruct) → embedding (frozen mT5 encoder) | `src/summarize/hierarchical.py` |
| 2 | Embedding & alignment: text→d_p projection, cross-attention alignment, overlapping patches (P=10, S=5), per-modality self-attention | `EmbeddingAlignment` in `src/models/model.py` |
| 3 | Co-attention fusion: symmetric text→price and price→text streams, concat + projection + residual + LayerNorm | `CoAttentionFusion` |
| 4 | Classification: flatten → FC → sigmoid, BCE loss | `StockTrendModel.head` |

Hyperparameters match Table 2: L=30, P=10, S=5, d_model=128, d_ff=256, 4 layers,
dropout 0.2, GELU, Adam lr 1e-4 with 1e-4 decay every 5 epochs, batch 16, 20 epochs, patience 5.
Model size ≈ 1.24M trainable params; the summarisation LLM is frozen and never backpropagated through.

## What changed for India

- **Universe/prices** — NSE via `yfinance` `.NS` tickers, `auto_adjust=True` so the frequent
  bonus/split actions don't create artificial return jumps. The trading calendar (including
  Muhurat sessions and the moving holiday list) is inferred from the data rather than hard-coded.
- **Price bands** — returns are clipped to the ±20% circuit limit *before* the paper's 3-sigma
  winsorisation. A circuit-hit day is a genuine event, not a data error.
- **Session-aware buckets** — the paper's "hourly" summarisation becomes NSE-session buckets:
  pre-open, six intraday hours (09:15–15:30 IST), post-close.
- **No look-ahead on news** — anything timestamped at or after the 15:30 close is attributed to
  the *next* trading session. This is the single most important correctness detail, and the
  paper is silent on it.
- **Price channels (d_p=17)** — 6 OHLCV-family (open, high, low, close, log volume,
  log turnover) plus 11 technical indicators (ret1, RSI-14, MACD + signal, Bollinger %b and
  width, ATR-14, 20/50-day MA gaps, OBV, 10-day momentum). Matches the paper's Table 4
  "Price+TA+Text" row. All z-scored per symbol on training-period statistics only.
- **Split** — same 62.5 / 12.5 / 25 chronological proportions, shifted to 2018→2025:
  train ≤2022-12-31, val ≤2023-12-31, test 2024–2025. Measured on the real panel:
  1,234 / 247 / 494 trading days = 62.5 / 12.5 / 25.0%. (An earlier draft used
  `train_end: 2023-06-30`, which gave a 6.2% validation set — half the intended size.)
- **Summariser — swapped, and this mattered.** `mT5_multilingual_XLSum` is fine-tuned on
  BBC articles and does not summarise Indian filings so much as re-imagine them: measured on
  8 real filings it named the correct company **2/8**, rendering Axis Bank as "The Bank of
  England", ITC as "the British Stock Exchange" and Cipla's allotment as published by "the
  Financial Conduct Authority". Generation now uses **Qwen2.5-1.5B-Instruct** (8/8 correct,
  with accurate dates and share counts). Embeddings still come from the mT5 encoder, so
  `text_dim` stays 768 and nothing downstream moved; the two models are never resident at
  once. Instruction prefixes were also removed for seq2seq models — a fine-tuned summariser
  has never seen an instruction, so the directive is just more article to summarise, and
  being first it dominates (grounding fell from 31.1% to 13.4% with the prefix).

## Run

```bash
pip install -r requirements.txt
```

```bash
python scripts/smoke_test.py
```

```bash
python scripts/build_prices.py configs/india.yaml
```

```bash
python scripts/pipeline_test.py
```

End-to-end rehearsal on synthetic data — corpus → attribution → summarise → **kill and
resume** → embed → key lookup → train → results JSON → table, in ~2 minutes with a stub
summariser and no GPU. Run this before booking GPU time.

```bash
python scripts/build_text.py preflight configs/india.yaml --n 8
```

**A gate, not a formality.** Reports the extrapolated GPU-hours, sample IN/OUT pairs, and the
all-zero-row count. Read the samples: a summariser that copies the headline verbatim, or
invents content, makes the whole text arm meaningless.

```bash
python scripts/build_text.py summarize configs/india.yaml --chunk 200
python scripts/build_text.py summarize configs/india.yaml --raw
python scripts/build_text.py embed configs/india.yaml
python scripts/build_text.py embed configs/india.yaml --raw
```

Checkpointed every 200 stock-days and fsync'd — killing it is safe, the same command resumes.

```bash
python scripts/run_session.py --deadline-hours 10
```

Unattended driver for the whole thing: summarise → embed → guards → ablations, with the
shuffled-text and 60%-accuracy stop conditions enforced in code. Resumes at every level
(chunk, seed, ablation). Writes `artifacts/SESSION_REPORT.md`.

Metrics reported: accuracy, MCC and F1 on the held-out test window, evaluated once after
model selection on validation MCC.

## Getting the news (the hourly/daily part)

There is no "hourly news feed" to download - you collect **timestamped articles** and the
pipeline buckets them by IST session clock. Timestamps are the whole requirement; without
them the paper's two-pass summarisation has nothing to bucket. Two collectors are provided:

```bash
python scripts/collect_news.py bse
```

**BSE corporate announcements - the historical backbone.** Free, unauthenticated, exchange-
official, filed with exact IST timestamps, queryable by date range back to ~2018. Verified
working: 208 announcements for RELIANCE/TCS/INFY in Q1-2024 alone. Symbol mapping is
automatic via the BSE scrip master (49/50 of the bundled NIFTY universe map cleanly; LTIM
does not, and is skipped with a warning).

```bash
python scripts/collect_news.py rss
```

**RSS - forward collection only.** Economic Times, Business Standard, Moneycontrol. Back-
archives are thin, so this accumulates from the day you start; schedule it every 30-60 min.
Feeds are market-wide, so `tag_symbols()` attributes articles by whole-word company-name
match. Yield is low and honestly so - roughly 1 in 50 market items names a specific stock -
and the matching is deliberately conservative, because a false attribution injects another
company's news into this stock's daily summary.

### What the real timestamps revealed

The filing-hour histogram for Q1-2024 is lopsided:

```
09-15 IST (market open):  ~35 filings
16-20 IST (after close):  ~145 filings
```

**Most corporate announcements are filed after the 15:30 close.** This is why
`attribution_date()` pushes post-close news to the *next* trading session. Attributing it to
the same day - which is the naive reading of the paper - would leak tomorrow's catalyst into
today's features and inflate accuracy substantially. If you replicate this paper on Indian
data and skip that rule, your results will be wrong in the flattering direction.

### Coverage ladder

| Source | Cost | History | Timestamps | Verdict |
|---|---|---|---|---|
| BSE announcements | free | 2018+ | exact IST | start here; defensible corpus on its own |
| RSS (ET/BS/MC) | free | forward only | exact, +0530 | run it now so 2027 has archives |
| GDELT doc API | free | 2018+ | 15-min | good in principle; timed out from this machine |
| yfinance `.news` | free | ~10 items | yes | not usable - returns off-symbol results |
| Trendlyne / Refinitiv | paid | full | exact | the real answer if this is funded |

Until `data/raw/news.csv` exists the text stream zero-fills and the model degrades to a
price-only patch transformer - a useful ablation baseline, but not the paper's model.

## Results

Run 2026-08-27/28, RTX 4050 Laptop (6 GB), 5 seeds per ablation, test window 2024–2025
(464 trading days, 23,200 windows). Intervals are block-bootstrapped over **trading days**,
not stock-days.

| Ablation | Accuracy | MCC | MCC 95% CI | vs `full` (G3) |
|---|---|---|---|---|
| `full` | 50.31% ± 0.33 | 0.010 ± 0.007 | [-0.006, 0.035] | — |
| `price_only` | 50.32% ± 0.56 | 0.010 ± 0.009 | [-0.013, 0.027] | noise |
| `shuffled_text` | 50.14% ± 0.10 | 0.007 ± 0.003 | [-0.014, 0.027] | noise |
| `concat` | 50.52% ± 0.45 | 0.013 ± 0.008 | [-0.004, 0.034] | noise |
| `oneway` | 50.09% ± 0.31 | 0.009 ± 0.007 | [-0.013, 0.025] | noise |
| `raw_text` | 50.25% ± 0.68 | 0.008 ± 0.012 | [-0.014, 0.023] | noise |

**Constant-up floor: 50.45% accuracy, 0.000 MCC.**

Every ablation is statistically indistinguishable from every other and from the floor. The
negative control (`shuffled_text`) lands at chance, as it must — had it beaten chance, the
text stream would have been leaking rather than informing.

### Is this a null result or a broken pipeline?

A null is only worth reporting if the pipeline can be shown to work. Three independent checks:

1. **The negative control passes.** `shuffled_text` sits at chance rather than above it.
2. **A linear baseline matches the deep model exactly.** Logistic regression on the same 17
   price features and the same split scores 49.83% / 0.003 MCC — indistinguishable from the
   4-layer patch transformer. The ceiling is in the data, not the architecture.
3. **Training is healthy.** Loss falls smoothly (0.721 → 0.690), early stopping fires at
   varied epochs across seeds (6–18), no collapse to a constant prediction, 24/24 leakage
   and resume guards pass on real data.

### The caveat that matters most

**Real text exists on only 9.5% of stock-days, not the 23.4% originally assumed.** BSE serves
most Regulation 30 filings as a category label plus a PDF attachment, and `collect_news.py`
captures the attachment *name* and then discards it — so 60.2% of the post-boilerplate corpus
has an empty body. The single string `"Announcement under Regulation 30 (LODR)-Press Release
/ Media Release. A."` appears **3,969 times**, and identical stubs produce identical invented
summaries across unrelated stocks: a constant injected into the text stream, not signal.
Those rows are now filtered (`min_body_chars: 30`).

So `full` tying `price_only` means *"text did not help on the 9.5% of stock-days where text
actually exists"* — a narrower claim than "news does not predict Indian equities". Fetching
the attachment PDFs is the real repair and remains open.

## Expected results, honestly

The paper reports 67.01% / 0.346 MCC (US) and 70.24% / 0.293 MCC (CN). Those numbers are far
above what most published stock-direction work sustains out-of-sample; treat them as an upper
bound, not a target. On NSE, with a comparable corpus, 53–57% accuracy and MCC in the
0.05–0.15 range was the pre-registered realistic band. **The measured outcome came in below
even that** — ~50.3% / 0.010, at the constant-up floor — for the corpus reason documented
above. Report MCC alongside accuracy — the up-share on NSE
runs slightly above 50%, so accuracy alone flatters a constant-up predictor.

This is a research replication, not a trading system. It has no transaction costs, no STT or
slippage, no position sizing, and no execution model — directional accuracy above 50% does not
imply profitability after Indian round-trip costs.
