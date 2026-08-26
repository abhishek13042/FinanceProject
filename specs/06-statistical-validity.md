# Spec 06 — Statistical validity

**Status: constraints measured; gates defined, binding on spec 05, and now enforced in code.**
`src/eval.py` implements every gate that can be implemented; the rest are process rules that
no amount of code can enforce. This spec exists to stop the project from producing a
confident wrong answer.

## The measurement

12 NIFTY large-caps, 2018–2025, 1,974 trading days:

- Mean pairwise daily-return correlation: **ρ = 0.328**
- Days where ≥80% of the universe shares direction: **28.8%**
- Up-share: **0.513**

### Universe selection (measured over all 88 NIFTY-100 names with ≥95% coverage, 1,975 days)

CMIN selects purely by size — CMIN-US is the 110 largest U.S. stocks by market cap, CMIN-CN
is all 300 CSI-300 constituents. Neither controls for cross-sectional correlation, and the
paper reports 69,300 / 189,000 "training samples" as though the rows were independent. They
are not: at a typical US mega-cap ρ ≈ 0.5, CMIN-US's 69,300 rows are worth roughly 1,250
effective observations. We do not repeat that framing.

We select 50 names by round-robin across NSE sector classifications, taking the most liquid
unpicked name from the least-represented sector:

| Universe | ρ | n | n_eff | ≥80% agreement days |
|---|---|---|---|---|
| top-50 by liquidity (CMIN-style) | 0.312 | 98,700 | 6,067 | 24.3% |
| **sector-stratified (adopted)** | **0.289** | 98,700 | **6,512** | **20.9%** |

The gain is modest and honestly stated: ρ falls 7%, n_eff rises 7%. The agreement-day drop
(24.3% → 20.9%, a 14% relative reduction) matters more, because that is the axis along which
a model can score well by predicting the index rather than the stock. No sector exceeds 4 of
50; `data/raw/universe50.csv` carries the sector label for every name.

**We hold 3× the trading days CMIN trains on** (1,974 vs 630). On the axis that actually
binds effective sample size, this replication is better-powered than the source paper.

Stock-days are therefore not independent observations. Applying the Kish effective-sample
formula, `n_eff = n / (1 + (k-1)ρ)`:

| Universe | Nominal windows | **Effective** |
|---|---|---|
| 50 | 98,700 | ~5,775 |
| 200 | 394,800 | ~5,950 |
| 500 | 987,000 | ~5,987 |

**A 10× larger universe buys ~3.7% more information.** Each additional stock contributes
~0.22 of an independent sample, and the curve saturates almost immediately.

## What this means

1. **The binding constraint is 1,974 trading days, not stock count.** No collection effort
   creates more history. Universe expansion is not a route to statistical power.
2. **Expand to NIFTY 500 anyway — for a different reason.** The projection above holds ρ at
   the large-cap value, which is pessimistic. Mid- and small-caps are less index-driven, so a
   broad universe should exceed 5,987 effective. The gain comes from *lower correlation*, not
   from more rows. Verify by re-measuring ρ on the wider universe before assuming it.
3. **~6,000 effective observations against 1.25M parameters** is the real overfitting risk,
   and it drives spec 04's proposed depth reduction.
4. **Power.** With test-window effective N on the order of ~1,000, the standard error on
   accuracy is ≈1.6% and a 95% interval is roughly ±3%. Beating 51.3% significantly needs
   ~54.5%+.

## Gates

No result is reportable unless all of these hold. "Enforced" means the code makes the gate
automatic; the unenforced ones are process discipline and cannot be automated away.

| Gate | Requirement | Enforcement |
|---|---|---|
| **G1** | ≥5 seeds, reported mean ± std | `across_seeds`; `--seeds` defaults to five |
| **G2** | Intervals via block bootstrap over **trading days**, never stock-days | `block_bootstrap` — resamples days, not rows |
| **G3** | Every ablation claim rests on non-overlapping intervals | `intervals_overlap` — call it before writing any comparison down |
| **G4** | Constant-up and price-only baselines beside every headline number | constant-up in every results file; price-only is `--ablation price_only` |
| **G5** | Shuffled-text control lands at chance | `--ablation shuffled_text` |
| **G6** | ρ re-measured on the final universe, effective N restated | measured above for the adopted 50; **re-measure if the universe changes** |
| **G7** | Test set evaluated exactly once | *process only* — the code cannot tell an honest run from a fifth attempt |
| **G8** | Metrics split by market-agreement day | `agreement_days` + `split_by_agreement`, in every results file |

G3 is the one most likely to be violated by accident, because the temptation arrives *after*
the numbers are in hand and every ablation table invites a ranking. A 55.1% vs 53.8% table is
noise. G7 is the one that cannot be recovered from: once the test set has informed a choice,
no later discipline undoes it.

## The failure mode this spec exists to prevent

The architecture will produce a number. With 98,700 nominal samples it will *look* well
powered. Ablation tables will show differences of 1–2 points, and those differences will be
entirely within noise while appearing to confirm the paper.

Note the paper *does* report mean ± std over 10 seeds with paired t-tests — it is not
methodologically naive. What it does not do is discount for cross-sectional correlation, and
its reported spreads are strikingly tight (±0.21 accuracy across seeds). Our gates are
therefore stricter than the paper's, not a correction of an omission.

- [ ] **G8 — Beta-contamination gate.** Report test metrics split by market-agreement day
      (≥80% of the universe moving together vs the rest). If the model's edge is confined to
      high-agreement days it is timing the index, not predicting the stock, and the result
      does not clear this spec regardless of headline MCC. The source paper runs no such
      diagnostic. Agreement days are derived from the label column itself, so the split is
      exactly the data the model is scored against — no separate index series to misalign.
