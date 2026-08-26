# Spec 00 — System overview

## Goal

Replicate Zhang, Dong & Xu (ESWA 302, 2026) — hierarchical LLM text processing + patch-based
transformer with co-attention — on Indian (NSE) equities, and establish honestly whether the
architecture's claimed benefit survives contact with Indian data.

**Task.** Binary next-day direction. `y(τ+1) = 1` iff `close(τ+1) > close(τ)`. No magnitude
threshold, matching the paper's Eq. (17).

**Explicit non-goal.** This is a research replication, not a trading system. No transaction
costs, STT, slippage, impact, position sizing or execution model. Directional accuracy above
50% does not imply profitability after Indian round-trip costs, and no spec here should be
read as claiming otherwise.

## Module contract

| Stage | Module | Input | Output |
|---|---|---|---|
| 1 | Hierarchical summary | raw timestamped articles | `(symbol, date) → R^768` embedding |
| 2 | Embedding & alignment | price `(B,L,d_p)` + text `(B,L,768)` | patch tensors `(B·d_p, N, D)` ×2 |
| 3 | Co-attention fusion | two patch tensors | fused `(B·d_p, N, D)` |
| 4 | Classification | fused tensor | one logit per sample |

Stage 1 is a **frozen** LLM run offline; its output is cached to disk. Stages 2–4 are trained
end to end. This split is deliberate and load-bearing: it keeps the trainable parameter count
at ~1.25M, which matters given spec 06's effective-sample ceiling.

## Fixed dimensions

`L=30` trading days · `P=10` patch · `S=5` stride · `N=6` patches · `d_model=128` ·
`d_ff=256` · `d_p=5` price channels · text dim 768 (mT5-base encoder).

`N = floor((L-P)/S) + 2`, the paper's Eq. (4).

## Glossary

- **stock-day** — one (symbol, trading date) pair.
- **window** — one training sample: 30 consecutive stock-days plus the next-day label.
- **effective sample size** — window count discounted for cross-sectional correlation. The
  number that actually governs statistical power. See spec 06.
- **attribution date** — the trading session a news item is allowed to inform. Not the
  publication date. See spec 02.
