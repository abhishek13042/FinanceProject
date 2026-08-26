# Spec 04 — Model

**Status: built and smoke-tested.** `src/models/model.py`. Forward and backward verified on
synthetic tensors: 1.25M params, `(B,30,17)` + `(B,30,768)` → `(B,)` logits, gradients flow.

Implements the paper's Sections 3.3–3.5 faithfully. Deviations from the paper belong in
spec 05 as ablations, not here.

## Contract

- **In:** price `(B, L, d_p)` z-scored, text `(B, L, 768)` daily embeddings.
- **Out:** `(B,)` raw logits. Sigmoid and BCE live in the training loop
  (`BCEWithLogitsLoss` for numerical stability).

## Components

| Component | Paper | Class |
|---|---|---|
| Text→price projection + dropout | Eq. (1) | `EmbeddingAlignment.text_proj` |
| Cross-attention alignment (price queries text) | Eqs. (2)–(3) | `EmbeddingAlignment.forward` |
| Overlapping patches, `N = floor((L-P)/S)+2` | Eq. (4) | `_patchify` |
| Patch projection + learnable positions | Eq. (5) | `patch_proj`, `pos` |
| Per-modality self-attention, `d_p` folded into batch | Eqs. (6)–(7) | `_encode` |
| Co-attention, text→price and price→text | Eqs. (8)–(13) | `CoAttentionFusion` |
| Flatten → FC → sigmoid, BCE | Eqs. (14)–(16) | `StockTrendModel.head` |

**Single-channel** means the encoders are shared across the `d_p` price channels, which are
folded into the batch axis. This is what keeps the parameter count at 1.25M with 17
channels instead of 17 separate encoders.

## Requirements

1. **R1 — Patch tail.** `_patchify` replicate-pads so the final partial window is preserved;
   the `+2` in Eq. (4) accounts for it. Patch count must equal Eq. (4) exactly.
2. **R2 — Symmetry.** Both co-attention streams use identical architecture and independent
   weights. Neither modality may be privileged — that is the paper's whole argument against
   one-way cross-attention.
3. **R3 — Identical treatment.** Patch projection and position embeddings apply the same way
   to both modalities.
4. **R4 — Graceful text absence.** A zero text vector (uncovered stock-day) must produce a
   finite loss and finite gradients, not NaN.
5. **R5 — Alignment width is a parameter, not a constant.** Eq. (1) projects the 768-d text
   embedding down to `d_p = 17` — a 45x linear squeeze applied *before* the encoder, so no
   downstream layer can recover what it discards. The paper never justifies tying the
   alignment width to the price-channel count; the two quantities are unrelated. `d_align`
   (`EmbeddingAlignment.da`) decouples them: text is projected to `d_align`, and price is
   projected **up** to meet it via `price_proj`. `d_align == n_price_feats` sets `price_proj`
   to `nn.Identity()` and reproduces the paper bit-for-bit at 1,254,312 parameters — so the
   default is a faithful replication and widening is opt-in.

   Cost scales with `d_align` on **both** streams, since channel count enters the batch axis
   (measured, 2 CPU threads, batch 16): 17 → 405 ms/step, 32 → 736, 64 → 1508. Parameters
   barely move (1.254M / 1.280M / 1.339M) — this is a compute axis, not a capacity axis,
   which is what makes it safe against spec 06's ~6,000 effective samples.

## Acceptance criteria

- [x] Forward pass returns `(B,)`; backward produces finite gradients.
- [x] `N = 6` for L=30, P=10, S=5 — matches Eq. (4).
- [x] Parameter count 1.25M at `d_align = 17`.
- [x] R4 verified explicitly with an all-zero text batch: loss and all gradients finite, grad norm 21.88.
- [ ] Attention weights inspected once on real data (sanity, not a metric).
- [x] `d_align` ∈ {17, 32, 64} all build and forward; `d_align = 17` preserves the baseline
      parameter count exactly.
- [ ] PCA variance retained at 17 / 32 / 64 dims measured on training summary embeddings,
      and used to pick the ablation grid (spec 03 must have run first).

## Pending change, driven by spec 06

The two 4-layer encoders account for ~1.05M of the 1.25M parameters. Against spec 06's
~6,000 effective samples this is over-parameterised. **Proposal: `n_layers: 4 → 2`**, roughly
halving the model. The paper's depth was tuned on a corpus with the same effective-N ceiling
and was never justified by data volume. Treat 4-layer vs 2-layer as a spec-05 ablation with
confidence intervals, and adopt the smaller model unless depth demonstrably wins.
