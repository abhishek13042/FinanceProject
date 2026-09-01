# Study Guide 1 — Concepts from zero

Every concept this codebase assumes, built up in the order it's actually needed. Skimming
this and jumping to document 2 will leave you unable to explain *why* a design choice was
made — only *what* it is.

Read after document 0. Time: 45–60 minutes.

---

## Part A — the task and why it's hard

**1. The task.** Binary next-day direction: `y(τ+1) = 1` iff `close(τ+1) > close(τ)`,
strict inequality, no magnitude threshold ([`prices.py:120`](../../src/data/prices.py)).
Flat days are label 0, same as down days — direction, not magnitude.

**2. Why "just predict up" is a real baseline, not a joke one.** NSE up-share (fraction of
stock-days that closed higher than they opened the prior day) is **0.513** — measured on
the full 50-symbol panel. A model that always predicts "up" scores ~51.3% accuracy and
exactly 0 MCC by construction. Every result in this project must be reported *next to*
that number (spec 05 R5), because accuracy alone cannot distinguish a real edge from
predicting the market's baseline drift.

**3. MCC (Matthews correlation coefficient), and why it's primary, not accuracy.**
`sklearn.metrics.matthews_corrcoef` — ranges -1 to 1, 0 for a model with no better-than-
chance information (including the constant-up predictor). Because up-share isn't exactly
50/50, MCC is the metric that can't be gamed by a model that just learned the base rate.

## Part B — the paper's architecture, stage by stage

**4. The four-stage pipeline** ([`specs/00-system.md`](../../specs/00-system.md)):
hierarchical summary → embedding & alignment → co-attention fusion → classification.
Stage 1 is **frozen** (an LLM, never fine-tuned, run once offline); stages 2–4 are the
~1.25M trainable parameters.

**5. Why the LLM is frozen.** Fine-tuning an LLM alongside a ~1.25M-parameter classifier
on a few thousand effective training examples (see §16 below) would massively overfit.
Freezing it and caching its output to disk keeps the trainable parameter count small and
decouples an expensive, one-time GPU cost from every subsequent training run.

**6. Hierarchical (two-pass) summarisation** — the paper's Algorithm 1, implemented in
[`hierarchical.py`](../../src/summarize/hierarchical.py). Pass 1 summarises each
*intraday session bucket* of articles (pre-open, six intraday hours, post-close — India's
adaptation of the paper's plain "hourly" scheme, since NSE trades 09:15–15:30 IST only).
Pass 2 concatenates the bucket summaries for a stock-day and summarises *that* into one
daily synopsis. Only the final daily summary is embedded and kept.

**7. Why two passes instead of one.** A stock-day with many articles concatenated
directly would blow past the summariser's input token budget and dilute a single
market-moving fact among boilerplate. Distilling per-bucket first, then per-day, keeps
each summarisation call's input short and each pass focused.

**8. Embedding, not text, crosses into the trainable model.** Only the pooled encoder
vector (mean-pooled over token positions, masked by attention — `HierarchicalSummarizer.
embed`) is forwarded. This matches the paper's stated token-budget argument: passing raw
text into the trainable transformer would require its own tokenizer and vastly more
parameters for no benefit once a frozen encoder already exists.

**9. `EmbeddingAlignment` (Eqs. 1–7, [`model.py`](../../src/models/model.py)).** Text
(768-d) is linearly projected down to an *alignment width* `d_align`; price (17-d) is
projected up (or left as `Identity` when `d_align == n_price_feats`) to meet it. Price
then **queries** text via one cross-attention pass (`wq`/`wk`/`wv` on the aligned
dimension) — this produces an "aligned text stream" that's been filtered through what the
price series is actually asking about, before either stream is patched.

**10. Overlapping patches (Eq. 4).** A 30-day window is split into `N = floor((L-P)/S)+2 =
6` overlapping patches of length `P=10`, stride `S=5`, with the trailing partial window
replicate-padded rather than dropped. Patching turns a long sequence into a short one
(`N=6` instead of `L=30` positions) the attention layers can process cheaply, at the cost
of each patch overlapping its neighbours by half.

**11. `d_p` folded into the batch axis — "single-channel" encoders (Eq. 6–7).** Rather
than 17 separate encoders (one per price channel), all channels share one encoder by
reshaping `(B, d_p, N, D) → (B·d_p, N, D)` before the transformer layers. This is *why*
the model stays at ~1.25M parameters with 17 price channels instead of scaling linearly
with channel count.

**12. Co-attention fusion (Eqs. 8–13), symmetric by design.** `CoAttentionFusion` runs
*two* independent multi-head attention passes — text-queries-price and price-queries-text
— then concatenates, projects, and adds a residual with LayerNorm. The paper's central
claim is that *bidirectional* fusion (both directions) beats a one-way or non-attentive
alternative; `ConcatFusion` and `OneWayFusion` in the same file exist specifically as
ablation controls for that claim (document 2 §9, document 4 §3).

**13. Classification head (Eqs. 14–16).** Flatten the fused `(B·d_align, N, D)` tensor
back to per-sample, one linear layer to a single logit, `BCEWithLogitsLoss` (numerically
stable sigmoid+BCE combined) in the training loop rather than a separate sigmoid.

## Part C — what changed for India, and why each change is load-bearing

**14. `auto_adjust=True`.** NSE bonus issues and stock splits are frequent enough that an
unadjusted series produces price "gaps" indistinguishable from real moves — this single
`yfinance` flag prevents an entire class of false signal.

**15. Circuit-band clipping *before* sigma-winsorisation** (`prices.py:clean`, R4). NSE
enforces a ±20% daily price band. A circuit-limit day is clipped to that band first,
*then* the training-period 3-sigma winsorisation runs on what's left. Reversing the
order — winsorising first — would treat a circuit-hit day (a genuinely informative event)
as a statistical outlier to be squashed, destroying exactly the kind of move a text
stream ought to be explaining.

**16. Post-close news attribution — the single most-repeated correctness rule in this
project** (`hierarchical.py:attribution_date`, spec 02 R4, spec 07's guard list). News
timestamped at or after the 15:30 IST close is attributed to the **next** trading
session, not the session it fell during. The measured reason: in Q1-2024, roughly 145 of
180 sampled BSE filings landed 16:00–20:00 IST — *after* the close, not during trading.
Attributing them same-day would hand the model tomorrow's catalyst as if it were already
known, inflating accuracy in the flattering direction. This rule appears in the README,
three separate specs, and its own dedicated leakage test (document 3 §5).

**17. Trading calendar derived from the data, not hard-coded** (`clean`, R2). NSE's
holiday list shifts yearly and includes the Muhurat (Diwali) session, which trades on an
otherwise-closed day at unusual hours. Deriving the calendar from `sorted(df["date"].
unique())` instead of a maintained holiday table means the calendar is always correct for
whatever data was actually fetched.

**18. `mT5_multilingual_XLSum`** kept as the summariser specifically because Indian
business news is multilingual — the same model covers English plus Hindi, Bengali, Tamil,
Telugu, Marathi, and Gujarati wires without a model swap, and prompts are rewritten for
India-specific catalysts (SEBI/RBI action, promoter pledging, FII/DII flows — see
`PROMPT_HOUR`/`PROMPT_DAY` in `hierarchical.py`).

**19. Two independent news sources with opposite trade-offs** (spec 02). **BSE corporate
announcements** are the historical backbone — free, exchange-official, exact IST
timestamps, queryable back to ~2018 — because CMIN (the paper's source corpus) has no
Indian equivalent. **RSS feeds** (Economic Times, Business Standard, Moneycontrol) are
forward-collection-only with thin back-archives, useful for extending the corpus forward
in time but contributing nothing to a fixed 2018–2025 backfill.

**20. Administrative-filing filtering (R9).** 42.1% of the raw BSE corpus (26,008 of
61,791 filings) is paperwork — lost share certificates, trading-window closures, analyst-
meet notices — that no price could plausibly react to. `drop_boilerplate` removes these
*before* summarisation, because feeding boilerplate into the summariser wastes GPU time
generating a summary of nothing.

## Part D — the statistical apparatus (why so much of this repo isn't ML at all)

**21. Cross-sectional correlation, ρ, and why 98,700 rows aren't 98,700 observations.**
50 NSE large-caps move together on a shared market factor — measured mean pairwise daily
correlation **ρ = 0.289** on the adopted (sector-stratified) universe. On a day the whole
market moves up 2%, the 50 stock-days sampled from it carry closer to one day's worth of
independent information than fifty.

**22. Effective sample size (Kish formula).** `n_eff = n / (1 + (k-1)ρ)` where `k` is
stocks-per-day. This is the number that actually governs statistical power — measured at
**~6,000** effective observations for the adopted universe, regardless of whether the
nominal row count is 98,700 (50 stocks) or 987,000 (500 stocks): the curve saturates
almost immediately (spec 06). This is *the* number to cite if asked "how much data does
this project really have" — not the row count.

**23. Block bootstrap over trading days, never over stock-days** (`eval.py:
block_bootstrap`, spec 05 R7 / spec 06 G2). Confidence intervals are built by resampling
whole *days* with replacement (all ~50 stocks' predictions for a sampled day move
together), not individual rows. Resampling rows instead would divide interval width by
roughly `sqrt(50)` and manufacture statistical significance that isn't there — this is
the direct code-level consequence of concept 21.

**24. The market-agreement ("beta-contamination") gate** (`eval.py:agreement_days` /
`split_by_agreement`, spec 06 G8). Days where ≥80% of the 50-stock universe moved the
same direction are split out and scored separately. If a model's apparent edge is
confined to those index-driven days, it's timing the market, not predicting the stock —
and the headline number doesn't clear this spec regardless of its MCC.

**25. The shuffled-text negative control** (`dataset.py:shuffle_text`, spec 05). Text
embeddings are permuted *within each trading day* — every window still gets a real,
same-day summary, just belonging to a *different* stock. If accuracy doesn't collapse to
chance under this control, the text stream is being exploited through some leak, not
through genuine signal — and the control is specifically designed so the leak this test
would catch (date-based, rather than stock-based) can't accidentally "pass."

**26. Multi-seed reporting, always mean ± std over ≥5 seeds** (`eval.py:across_seeds`,
spec 05 R6 / spec 06 G1). A single training run's result is not reportable on its own —
initialization variance alone can move a small model's test metric by more than the
effect being measured.

**27. Non-overlapping confidence intervals as the bar for any comparison claim**
(`eval.py:intervals_overlap`, spec 06 G3). Two ablations whose bootstrap intervals
overlap are, by this project's own rule, indistinguishable — a ranked ablation table with
overlapping intervals is not a finding, no matter how orderly it looks.

## Glossary

| Term | Meaning |
|---|---|
| **stock-day** | one `(symbol, trading_date)` pair — the atomic unit of this dataset |
| **window** | one training sample: 30 consecutive stock-days of one symbol, plus the next-day label |
| **effective sample size (n_eff)** | window count discounted for cross-sectional correlation — the number that actually governs statistical power |
| **attribution date** | the trading session a news item is allowed to inform (not its publish date) |
| **patch** | a length-`P` overlapping slice of a 30-day window; `N=6` patches per window here |
| **d_align / da** | the shared width text and price are both projected to before cross-attention; 17 reproduces the paper exactly |
| **ablation** | a controlled variant (text zeroed, fusion swapped, etc.) run to isolate what a design choice actually contributes |
| **block bootstrap** | resampling whole trading days (not rows) to build a confidence interval that respects cross-sectional correlation |
| **agreement day** | a day where ≥80% of the universe moved the same direction — the index-timing risk axis |
| **beta contamination** | the failure mode where a model's apparent stock-prediction edge is actually just market-timing |

---

## Self-test

**Level 1.** State the four pipeline stages. What is `ρ`, and what was it measured at for
the adopted universe?

**Level 2.** Why is MCC reported as the primary metric rather than accuracy? Why does the
block bootstrap resample days rather than rows — what would happen to the interval width
if it resampled rows instead?

**Level 3.**
1. *"You have 98,700 training rows — that's plenty of data."* Answer using §22: the
   nominal row count is not the number that matters; effective sample size for this
   universe is ~6,000 and barely rises even at 10x the stock count, because the binding
   constraint is 1,974 trading days of history, not stock count.
2. *"Why bother with a shuffled-text control — just check if the text ablation beats
   price-only?"* Answer using §25: a text ablation beating price-only could still be a
   leak rather than signal; the shuffled control specifically tests whether the model is
   exploiting the *date* alignment rather than genuine stock-specific content, which a
   simple beats-baseline comparison can't distinguish.

Next: [02-one-request-explained-slowly.md](02-one-request-explained-slowly.md).
