# Study Guide 2 — One stock-day, traced end to end

Rather than trace an HTTP request (there's no server here), this document traces **one
piece of data — a single stock-day** — from raw source through to the number the model
produces from it, through the actual files. Every decision below is numbered and points
at the line that makes it.

Read after document 1. Time: 1.5–2 hours.

---

## Part A — from raw download to a feature row

**File:** [`src/data/prices.py`](../../src/data/prices.py), driven by
[`scripts/build_prices.py`](../../scripts/build_prices.py).

### A.1 Download

```python
raw = yf.download(symbols, start=start, end=end, auto_adjust=True,
                  group_by="ticker", progress=False, threads=True)
```

**Decision 1 — `auto_adjust=True` is not optional.** Document 1 §14 already named why;
seeing it in the actual call confirms it's applied at the download step itself, not
patched in afterward — there is no code path in this project that ever sees unadjusted
NSE prices.

### A.2 Clean — gaps, calendar, and the circuit-then-sigma order

```python
cal = pd.DatetimeIndex(sorted(df["date"].unique()))          # NSE calendar from the data
...
g[["open","high","low","close"]] = g[[...]].ffill(limit=d["ffill_max_gap"])
if g["close"].isna().mean() > d["max_missing_frac"]:
    continue                                                  # delisted / illiquid -> drop
```

**Decision 2 — a gap of ≤3 sessions is forward-filled; a stock missing >5% of the full
period is dropped entirely,** not partially patched. The two thresholds (`ffill_max_gap:
3`, `max_missing_frac: 0.05` — both in `configs/india.yaml`) encode two different beliefs
about *why* data is missing: a short gap is probably a trading halt (real, recoverable);
persistent absence over 5% of eight years signals delisting or chronic illiquidity (not
recoverable, and forward-filling it would fabricate a price history).

```python
band = m["circuit_pct"] / 100.0
df["ret"] = df["ret"].clip(-band, band)                       # NSE price band, FIRST
tr = df[df["date"] <= pd.Timestamp(train_end)]
stats = tr.groupby("symbol")["ret"].agg(["mean", "std"])      # training-only statistics
...
df["ret"] = df["ret"].clip(df["mean"] - k * df["std"], df["mean"] + k * df["std"])
```

**Decision 3 — the ±20% circuit clip runs before the 3-sigma winsorisation, and the sigma
statistics are computed on the training slice only.** Two separate correctness rules
share this one code block: the *order* (document 1 §15 — a circuit day is real, not an
outlier to squash further) and the *slice* (only `tr`, the pre-`train_end` rows, compute
`mean`/`std` — using the full panel would leak future volatility statistics backward into
every period's z-score, which is exactly what `test_scaler_uses_training_rows_only`
(document 3 §5) exists to catch).

### A.3 Featurize — 17 channels, training-only z-scoring

```python
df["close_raw"] = df["close"]              # kept unscaled — labels are built from this
...
tr = df[df["date"] <= pd.Timestamp(train_end)]
mu = tr.groupby("symbol")[FEATURE_COLS].mean()
sd = tr.groupby("symbol")[FEATURE_COLS].std().replace(0, 1.0)
df[FEATURE_COLS] = (df[FEATURE_COLS] - mu...) / sd...
```

**Decision 4 — `close_raw` is preserved unscaled, deliberately, before any z-scoring
touches `close`.** The label (`close.shift(-1) > close`) must compare real prices; a
z-scored "close" would still work *within* one symbol's own scale but would be silently
wrong the moment anyone reused it as a price, and `test_close_raw_is_left_unscaled`
(document 3 §5) checks exactly this.

**Decision 5 — 17 channels, not the paper's "close-only" base config.** Six OHLCV-family
columns (`open, high, low, close, volume, turnover`) plus eleven technical indicators
(`ret1, rsi14, macd, macd_sig, bb_pctb, bb_width, atr14, sma20_gap, sma50_gap, obv,
mom10`), each z-scored per symbol on training-only statistics (decision 3's slice reused
here). `turnover = close * volume`, log-transformed along with `volume` and `obv`
(log-transforming a heavy-tailed quantity before z-scoring keeps outlying days from
dominating the training mean/std).

**Decision 6 — the label is computed *after* z-scoring, from `close_raw`, not from any
z-scored column** — `df["label"] = (df.groupby("symbol")["close_raw"].shift(-1) >
df["close_raw"]).astype(int)`. This is the same discipline as decision 4, applied at the
point the label is actually built.

## Part B — from raw filings to a same-day-safe embedding

**Files:** [`src/data/news_sources.py`](../../src/data/news_sources.py) →
[`src/summarize/hierarchical.py`](../../src/summarize/hierarchical.py).

### B.1 Collection, resumable by construction

```python
# each symbol's rows are fsync'd to news.bse.partial.csv, then its name to news.bse.done.txt
```

**Decision 7 — a symbol's data hits disk *before* it's marked done, in that order, not
the reverse.** If the process crashes between the two writes, the symbol is re-fetched on
resume — safe, because decision 8 makes re-fetching idempotent — rather than lost, which
is what would happen if the done-marker were written first. `test_a_symbol_written_twice_
does_not_duplicate_rows` (document 3 §5) pins this ordering down directly.

**Decision 8 — dedup key `(symbol, timestamp, headline)` absorbs a re-fetch.** A resumed
collection run re-requests whatever wasn't marked done; the append step's dedup means
that's a safe no-op rather than doubled rows. A live regression caught exactly this
boundary (document 3 §5's `test_append_csv_is_idempotent_across_a_reload` comment):
"rows already on disk read back as strings while incoming rows are tz-aware Timestamps;
deduping across that boundary matched nothing" — a type mismatch, not a logic bug, and
one only a real re-run surfaced.

**Decision 9 — a network failure and a missing scrip code get opposite resume
treatment.** A dropped connection leaves the symbol *unmarked*, so the next resume
retries it. A symbol whose BSE scrip code can't be resolved is marked done anyway —
because it will never resolve itself, and retrying it every resume forever would be a
silent infinite loop disguised as progress.

### B.2 Boilerplate filtering, before any GPU time is spent

```python
kept = NS.drop_boilerplate(df, log=None)["headline"].tolist()
# "Loss of Share Certificate", "Closure of Trading Window", "Analyst / Investor Meet
# Intimation" dropped; "Board approves acquisition..." and earnings results kept
```

**Decision 10 — 42.1% of the raw corpus is filtered out before summarisation, by
headline pattern, not by content the LLM would have to read first.** The filter is
deliberately blunt in one direction only, per its own test's docstring: "administrative
notices out, anything that could move a price left alone" — false negatives (missing a
real filing) are treated as far worse than false positives (summarising something
harmless).

### B.3 Session bucketing and next-session attribution

```python
def bucket_by_session(ts, cfg):
    ...
    b[t.dt.time < start] = 0
    b[t.dt.time >= end] = cfg["market"]["hourly_buckets"] - 1

def attribution_date(ts, trading_days, cfg):
    day = t.dt.normalize()...
    day = day.where(t.dt.time < end, day + pd.Timedelta(days=1))
    idx = trading_days.searchsorted(day.values, side="left")...
    return trading_days[idx]
```

**Decision 11 — `attribution_date` snaps to the *next real trading day* via
`searchsorted`, not simply "tomorrow's calendar date."** A Friday 18:30 filing must
attribute to Monday, not Saturday — `searchsorted` against the actual trading-day index
handles weekends and holidays correctly without a separate calendar lookup, reusing the
same calendar concept as decision 2's `cal`.

**Decision 12 — the two functions are independent and serve different purposes.**
`bucket_by_session` decides which *intraday pass-1 group* an article joins (for the
two-pass summary, document 1 §6); `attribution_date` decides which *trading day's
window* the resulting daily summary is allowed to appear in. A post-close article is
bucketed into "post-close" (bucket 6) for pass-1 purposes, but its *day* is already the
next session by the time pass-2 runs — the two functions agree on direction (both push
post-close forward) but answer different questions.

### B.4 Two-pass summarisation and embedding (not yet executed — read as design, not as a result)

```python
news["hourly"] = self._gen(news["prompt"].tolist(), ...)             # pass 1, per bucket
grp["summary"] = self._gen(grp["prompt"].tolist(), ...)              # pass 2, per stock-day
...
h = self.model.get_encoder()(**b).last_hidden_state
mask = b["attention_mask"].unsqueeze(-1).float()
vecs.append(((h * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy())
```

**Decision 13 — embedding is attention-mask-weighted mean pooling over the encoder's last
hidden state, not the `[CLS]`-style single-token output** (mT5 doesn't have one; this is
the standard pooling choice for an encoder-decoder model). Padding tokens are zeroed out
of the mean by the mask, not merely truncated after the fact — necessary because batched
generation pads variable-length inputs.

**Decision 14 — this stage has never run in this project as of this session.** Every
decision in §B.4 is read from source, not from an observed output — `data/interim/
daily_summaries.csv` and `daily_embeddings.npz` don't exist on disk. Document 4 covers
this gap directly.

## Part C — one window, into the model, into a logit

**Files:** [`src/data/dataset.py`](../../src/data/dataset.py) →
[`src/models/model.py`](../../src/models/model.py).

### C.1 Building the window — indexed, not materialised

```python
self.px.append(g[PRICE_COLS].to_numpy(np.float32))
self.tx.append(np.stack([emb.get((sym, d), zero) for d in ds]).astype(np.float32))
...
def __getitem__(self, k):
    si, i = self.index[k]
    s = i - self.L + 1
    return (torch.from_numpy(self.px[si][s:i+1]), torch.from_numpy(self.tx[si][s:i+1]), ...)
```

**Decision 15 — each symbol's full daily price/text matrix is stored once; a window is a
slice, not a copy** (spec 05 R10). The comment atop the file states the reason directly:
materialising every window's text separately would cost `n_windows × 30 × 768` floats —
5.8 GB for the training split alone, because consecutive windows overlap in 29 of their
30 days and each overlapping day would otherwise be stored once per window that touches
it. Indexed, the whole panel's text is ~303 MB. **The first real-data training run
actually died on this exact allocation before the indexed design was adopted** (spec 05,
R10's note) — this wasn't a hypothetical optimisation.

**Decision 16 — an uncovered stock-day (no news that day) gets a zero vector, not an
omitted row or an error.** `emb.get((sym, d), zero)` — the model must handle a
genuinely-empty text input gracefully (this is exactly what `test_zero_text_gives_finite_
gradients`, document 3 §5, checks, and it's also the entire mechanism behind the
`price_only` ablation, decision 20 below).

### C.2 Alignment — price queries text, before either is patched

```python
x_t = self.text_proj(x_text)              # Eq. 1: 768 -> d_align
x_price = self.price_proj(x_price)        # Identity when d_align == n_price_feats
q, k, v = self.wq(x_price), self.wk(x_t), self.wv(x_t)
att = torch.softmax(q @ k.transpose(-2,-1) / sqrt(q.size(-1)), dim=-1)
x_tp = att @ v                            # Eq. 3: aligned text stream
```

**Decision 17 — the alignment squeeze (768 → 17 at default `d_align`) happens before any
encoder layer, and cannot be undone downstream.** Spec 04 R5 states this explicitly as a
finding, not just a design note: the paper never justifies tying `d_align` to
`n_price_feats`; decoupling them (`configs/india.yaml`'s `d_align: 17` with `align32`/
`align64` as spec-05 ablations) exists specifically to measure what that squeeze costs.

### C.3 Patching, fusion, and the classification head

```python
z = self.fusion(xt, xp)                    # (B*d_align, N, D)
z = z.view(-1, self.da, self.N, self.D)
return self.head(z).squeeze(-1)            # one logit per sample
```

**Decision 18 — the model returns a raw logit, not a probability**, and `train.py`'s
`predict()` applies `torch.sigmoid` only at evaluation time, while training uses
`BCEWithLogitsLoss` directly on the logit — the numerically-stable combined form, chosen
over a separate `sigmoid()` + `BCELoss()` specifically to avoid the loss saturating to
`-log(0)` when the model is confidently wrong.

### C.4 Training loop — model selection strictly on validation

```python
if m["mcc"] > best:                       # R2: model selection on validation MCC only
    best, bad = m["mcc"], 0
    torch.save(...)
...
model.load_state_dict(torch.load(ckpt))
prob, y = predict(model, dl_te, dev)       # R3: test set touched once, after selection
```

**Decision 19 — the checkpoint saved is the best-validation-MCC epoch, and the test set
is only ever touched once, after that selection is already final** (spec 05 R2/R3). This
is the code-level enforcement of spec 06's G7 — the one gate the code *cannot* fully
enforce (nothing stops a human from running this script five times and keeping the best
test number), which is exactly why spec 06 calls G7 "process only."

### C.5 The ablation mechanism — one dictionary, one dataset transform

```python
ABLATIONS = {
    "full": ({}, None),
    "price_only": ({}, "zero_text"),
    "shuffled_text": ({}, "shuffle_text"),
    ...
}
```

**Decision 20 — every ablation is a `(config override, dataset transform)` pair applied
identically to *all three* splits, never just the test split.** `apply_ablation`'s own
comment explains why: shuffling text only in the test set would measure a *distribution
shift* between train and test, not the absence of a learnable signal — the model would
be evaluated on data unlike anything it trained on, which is a different (and less
useful) question than "does the text stream carry real information."

### Flow diagram — one stock-day's whole journey

```
yfinance .NS download (auto_adjust=True)
  │
  ▼
clean(): calendar-from-data, gap fill/drop, circuit-clip THEN train-only sigma-clip
  │
  ▼
featurize(): close_raw preserved · 17 channels z-scored on training stats · label from close_raw
  │
  ▼                                              BSE filings + RSS  ──▶  drop_boilerplate (42.1% out)
prices.csv (one row per stock-day)                        │
  │                                                        ▼
  │                                          bucket_by_session (pass-1 grouping)
  │                                          attribution_date (post-close → next session)
  │                                                        │
  │                                          HierarchicalSummarizer: pass-1 → pass-2 → embed
  │                                          (not yet executed this session)
  │                                                        │
  └─────────────────────┬────────────────────────────────┘
                          ▼
           StockWindowDataset: indexed 30-day slice, price + text (or zero if uncovered)
                          │
                          ▼
           EmbeddingAlignment: text 768→d_align, price→d_align, price-queries-text attention
                          │
                          ▼
           patchify (N=6 overlapping patches) → per-modality encoder (d_p folded into batch)
                          │
                          ▼
           CoAttentionFusion: text→price ⊕ price→text, concat, project, residual, LayerNorm
                          │
                          ▼
           flatten → linear → one logit → BCEWithLogitsLoss (train) / sigmoid (eval)
```

---

## Self-test

**Level 1.** In what order does `clean()` apply the circuit clip and the sigma clip, and
why does the order matter? What does `zero_text()` do, and which ablation depends on it?

**Level 2.** Why does `apply_ablation` transform *every* split, not just the test split,
for the `shuffled_text` control? What real allocation failure drove the "indexed, not
materialised" window design (decision 15)?

**Level 3.**
1. *"Walk me through what happens to a single BSE filing from download to the model."*
   Use the flow diagram: collected and checkpointed (§B.1) → boilerplate-filtered (§B.2)
   → session-bucketed for pass-1, but date-attributed to its *tradable* session which may
   be the next day (§B.3) → summarised two-pass and embedded (§B.4, not yet run) → zero-
   filled if the stock-day it lands on has no coverage (§C.1) → attention-aligned against
   that day's price row before patching (§C.2).
2. *"Why can't the code itself guarantee the test set was only evaluated once?"* §C.4 —
   nothing prevents a human from re-running the training script and keeping whichever
   run's test number looks best; this is explicitly named as spec 06's G7, the one gate
   that's "process only."

Next: [03-the-data-model-explained-slowly.md](03-the-data-model-explained-slowly.md).
