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
| 03 | Hierarchical summarization | built, not yet run |
| 04 | Model | built, smoke-tested |
| 05 | Training & evaluation | built, not yet run on real data |
| 06 | Statistical validity | gates defined, binding on 05 |

Critical path to a first real result: **03 → 05**, gated by **06**.

## Architecture (unchanged from the paper)

| Stage | Module | File |
|---|---|---|
| 1 | Hierarchical summary: intraday-bucket pass → daily pass → embedding (frozen mT5) | `src/summarize/hierarchical.py` |
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
- **Price channels (d_p=5)** — close, log volume, return, high–low range, 20-day MA gap.
  The paper uses close only; NSE delivery-percentage is a natural 6th channel if you have
  the NSE bhavcopy (set `n_price_feats: 6`).
- **Split** — same 62.5 / 12.5 / 25 chronological proportions, shifted to 2018→2025:
  train ≤2023-06-30, val ≤2023-12-31, test 2024–2025.
- **Summariser** — `mT5_multilingual_XLSum` is kept as-is: it covers English plus Hindi,
  Bengali, Tamil, Telugu, Marathi and Gujarati, so vernacular business wires work without
  a model swap. Prompts are rewritten for Indian catalysts (SEBI/RBI action, promoter
  pledging, FII/DII flows, order wins, guidance).

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
python scripts/build_text.py configs/india.yaml
```

```bash
python -m src.train --config configs/india.yaml
```

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

## Expected results, honestly

The paper reports 67.01% / 0.346 MCC (US) and 70.24% / 0.293 MCC (CN). Those numbers are far
above what most published stock-direction work sustains out-of-sample; treat them as an upper
bound, not a target. On NSE, with a comparable corpus, 53–57% accuracy and MCC in the
0.05–0.15 range is the realistic band. Report MCC alongside accuracy — the up-share on NSE
runs slightly above 50%, so accuracy alone flatters a constant-up predictor.

This is a research replication, not a trading system. It has no transaction costs, no STT or
slippage, no position sizing, and no execution model — directional accuracy above 50% does not
imply profitability after Indian round-trip costs.
