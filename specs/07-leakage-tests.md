# Spec 07 — Leakage guards

**Status: implemented and passing (14 tests, ~20 s on CPU).** `tests/test_leakage.py`.

## Why this spec exists

Specs 05 and 06 protect against *reporting* a result badly. This one protects against
*computing* one badly. The two failures look nothing alike: a statistical mistake produces a
number with an honest interval around it, while a leak produces a number that is simply not
about the future. A leak also flatters — it raises accuracy, which means nothing in the run
itself will look wrong.

Every test here is a specific way this project could produce a good-looking wrong answer.
None of them assert that the model works; they assert that a number, if produced, means what
the write-up will claim it means.

## Contract

- **In:** `configs/india.yaml`, and `data/interim/prices.csv` where a test needs the real panel.
- **Out:** pass/fail. No artifacts.

```bash
python -m pytest tests/ -q
```

Panel-dependent tests skip cleanly when `prices.csv` is absent, so the suite runs before
spec 01 has ever been executed. They stop being skipped the moment there is real data to
check, which is exactly when they start mattering.

## What is guarded

| Guard | The failure it catches |
|---|---|
| Splits disjoint and ordered | a boundary edit that silently overlaps train and test |
| No window crosses a split boundary | a validation window reaching back into training days |
| Scaler fit on training rows only | test-period mean/variance leaking backwards into z-scores |
| `close_raw` left unscaled | labels computed from a standardised "price" |
| Label is strictly next-day direction | any peek beyond t+1; a magnitude threshold creeping in |
| Final day per symbol never a target | training on a fabricated label |
| **Window text never comes from the future** | the text pipeline handing the model tomorrow's news |
| Post-close news → next session | a 16:00 IST filing scored against the close it preceded |
| Session buckets ordered and bounded | an intraday bucketing bug that scrambles chronology |
| Patch count = Eq. (4) | an architecture edit that changes N without anyone noticing |
| Parameter count = 1,254,312 | any accidental capacity change at `d_align = 17` |
| Zero-text finite gradients | the price-only ablation dying on a degenerate softmax |
| Text stream reaches the output | the *opposite* bug — silently dropped text |
| Panel on disk matches the current `train_end` | a **stale rebuild** after the split boundary moved |

## Three of these deserve explanation

**Window text never comes from the future.** Each embedding is set to a one-hot of its own
date ordinal, so a window's text tensor becomes a readable record of which days it drew text
from. `argmax` over the text dimension recovers the day indices, and the test asserts they
are exactly `i-L+1 … i`. This is stronger than checking that the lookup key looks right: it
verifies the tensor that the model actually consumes, and it survives any refactor of how
embeddings are joined.

**Panel on disk matches the current `train_end`.** This guard was written because the failure
actually happened: `train_end` was corrected from 2023-06-30 to 2022-12-31 (spec 05 R1) while
`prices.csv` still held z-scores fitted through June 2023 — features silently encoding
statistics from what had just become validation data. Nothing in the file records which
boundary it was built with, so the test checks the arithmetic that must hold if the fit was
train-only: the training slice standardises to (0, 1) per symbol, and later periods do not.

**Text stream reaches the output.** The mirror image of a leak, and easier to miss. A wiring
bug that dropped the text pathway would make every text ablation report "no effect" — and
that would be written up as a finding about news rather than as a bug. Two forward passes on
identical prices with different text must not agree.

## What this spec deliberately does not do

- **It does not test model quality.** A model can pass every guard here and still be
  worthless. That is the correct division: spec 05 asks whether the number is good, spec 07
  asks whether it is real.
- **It cannot enforce spec 06's G7** (test set touched once). No test can distinguish an
  honest run from a fifth attempt.
- **It does not cover the summariser's output quality.** Whether mT5 produces a real summary
  or copies the headline verbatim is checked by spec 03's preflight, against the actual model.

## Acceptance criteria

- [x] All 14 guards pass on the real 50-symbol panel (98,690 rows, up-share 0.510).
- [x] Suite runs on CPU in under a minute; no network, no model download.
- [ ] Re-run after every change to `dataset.py`, `prices.py`, `hierarchical.py` or `model.py`.
      These guards are worthless if they only ever ran once.
- [ ] Extend with a text-coverage guard once real summaries exist: assert the fraction of
      stock-days with non-empty text, so a silently-empty embedding file cannot masquerade as
      a price-only run.
