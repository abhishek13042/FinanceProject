# GPU session runbook (Legion)

Everything in this project that runs on a CPU has been run on the CPU. What is left needs a
GPU, and the GPU is borrowed — so this document assumes limited, interruptible time on a
machine that is not yours.

**Two jobs, in order:** summarise the news corpus (spec 03), then train the ablation matrix
(spec 05). Both resume after an interruption. Neither needs the internet if you bring the
model cache with you.

---

## 1. What to copy across

| What | Size | Why |
|---|---|---|
| the repo | ~1 MB | code, specs, configs |
| `data/interim/prices.csv` | 38 MB | spec 01 output, already built |
| `data/raw/news.csv` | 18 MB | 61,791 BSE filings, 2018-2025 — the summariser's input |
| `data/raw/bse_scrip_map.csv` | ~1 MB | symbol→scrip cache; only needed if you re-collect |
| `~/.cache/huggingface/` | ~2.3 GB | mT5 weights — **copy this** rather than re-downloading on someone else's connection |

Put the HF cache in the same place on the Legion, or point `HF_HOME` at wherever you put it.

```bash
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If that prints `False`, stop — the CPU-only wheel got installed. Install the CUDA build of
torch before going any further; everything below assumes a GPU and will silently run 20x
slower without one.

---

## 2. Preflight — 5 minutes, do not skip

```bash
python scripts/build_text.py preflight configs/india.yaml --n 8
```

This loads mT5, summarises eight stock-days, and reports throughput plus an **extrapolated
GPU-hour estimate for the full backfill**. Read three things before committing the session:

- **the hour estimate** — if it says 40 hours, you do not have a 40-hour session; reduce the
  corpus or the universe now, not after 12 hours of generation
- **the three IN/OUT samples** — mT5-XLSum sometimes copies the headline verbatim instead of
  summarising. The preflight warns when it detects this. Verbatim copies make spec 05's
  "no hierarchical summary" ablation meaningless, because both arms become the same thing
- **the all-zero-row count** — stock-days whose text is empty

The preflight prints a boilerplate line first: `dropped 26,008 of 61,791 (42.1%)`. That is
expected — it is administrative paperwork (lost share certificates, trading-window notices).
After it, text covers **23.4% of stock-days**, median 6 news days per 30-day window, 3.86% of
windows entirely text-free. Set `data.filter_boilerplate: false` if you want the unfiltered
arm, but do not expect the GPU hours back.

## 3. Summarise (spec 03)

```bash
python scripts/build_text.py summarize configs/india.yaml --chunk 200
python scripts/build_text.py summarize configs/india.yaml --raw     # ablation arm, no GPU
python scripts/build_text.py embed configs/india.yaml
python scripts/build_text.py embed configs/india.yaml --raw
```

Every 200 stock-days is fsync'd to `daily_summaries.partial.csv`. **Killing this is safe** —
restart the same command and it skips what is already on disk. A chunk is the most work a
crash can cost, so lower `--chunk` if the machine is unstable.

`--raw` needs no GPU at all; it concatenates headlines instead of summarising. Run it here
anyway so both text variants come home in one trip.

`embed` is the cheap step — text to vectors, fine on CPU. If the session ends after
`summarize`, bring `daily_summaries.csv` home and run `embed` on the laptop.

## 4. Train (spec 05)

```bash
python -m pytest tests/ -q          # 24 guards, ~30 s — run them on the new machine too
python scripts/run_all.py --only full,price_only,shuffled_text
```

`run_all.py` runs 8 ablations x 5 seeds and **skips any ablation already on disk**, so
re-running resumes the matrix. The order is deliberate:

1. `full`
2. `price_only`
3. `shuffled_text`

**Stop if the shuffled-text control does not land at chance.** It means the text stream is
being exploited through a leak, and every remaining row would be an ablation of a broken
pipeline — GPU hours spent producing numbers that cannot be reported.

### What it will cost

Measured on this laptop's CPU: 460 ms/step, 3,765 steps/epoch, **48 CPU-hours per ablation**
(20 epochs x 5 seeds). A laptop RTX should turn that into roughly 1–2 hours per ablation, so
the full matrix is most of a day. If the session is shorter, run the first three ablations —
they answer "does text help at all, and is the result real?", which is the whole question.
The remaining five are refinements.

`batch_size` is 16 for fidelity to the paper. Raising it will use the GPU better and change
the result; if you raise it, say so in the write-up rather than quietly.

## 5. What to bring home

```
data/interim/daily_summaries.csv          (+ _raw)
data/interim/daily_embeddings.npz         (+ _raw)
artifacts/results_*.json
```

The `.pt` checkpoints are optional — the results files contain every number that gets
reported. Then, on the laptop:

```bash
python scripts/run_all.py --table
```

which re-prints the comparison table and applies spec 06's G3: for each ablation it states
whether the difference from `full` is reportable or is noise. **Read that verdict before
writing anything down.** Two overlapping intervals are not a finding, however tidy the
ranking looks.

---

## Before quoting any number

Spec 06's gates, condensed:

- ≥5 seeds, mean ± std — automatic
- intervals from a bootstrap over **trading days**, not stock-days — automatic
- constant-up floor beside every headline — automatic
- differences must have disjoint intervals — `--table` tells you
- shuffled-text control at chance — check it yourself
- **the test set is evaluated once** — no code can enforce this one

Expected range on NSE data: **53–57% accuracy, MCC 0.05–0.15.** The paper's 67% / 70% is an
upper bound, not a target. Anything near 60% should be treated as a leak until proven
otherwise — start with `python -m pytest tests/ -q` and the shuffled-text control.
