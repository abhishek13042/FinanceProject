# Prompt to paste into Claude Code on the Legion

I'm continuing a research project on a borrowed GPU laptop. The repo is already copied here.
Read `HANDOFF.md` first — it is the runbook for exactly this session — then `specs/README.md`
for what each spec covers. Don't re-derive the design; it's settled and written down.

**What the project is:** a replication of a hierarchical-LLM-summarisation + patch-transformer
+ co-attention stock-direction model (Zhang, Dong & Xu, ESWA 302:130441), retargeted from the
paper's US/CN corpora to 50 NSE stocks, 2018-2025. Everything CPU-shaped is already done:
prices (`data/interim/prices.csv`), 61,791 BSE filings (`data/raw/news.csv`), the model, the
dataset, and 24 passing leakage/resume guards. What's left needs CUDA.

**Do these in order, and stop where I say stop:**

1. `pip install -r requirements.txt`, then verify CUDA:
   `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
   If `cuda.is_available()` is False, stop and fix the torch wheel — everything below silently
   runs ~20x slower on CPU and I'd rather lose ten minutes than a session.

2. `python scripts/build_text.py preflight configs/india.yaml --n 8`
   **This is a gate, not a formality.** Report back to me, before continuing:
   - the extrapolated GPU-hour estimate for the full summarisation run
   - the three IN/OUT samples — mT5-XLSum sometimes copies the headline verbatim instead of
     summarising it. If it does, spec 05's "no hierarchical summary" ablation collapses into
     the same thing as the main arm and the experiment loses its control. Say so plainly if
     you see verbatim copying; don't paper over it.
   - the all-zero-row count (stock-days with no text)
   You should also see `boilerplate filter: dropped ~26,008 of 61,791 (42.1%)` — that's
   expected, it's administrative paperwork, not a bug.

3. If the hour estimate fits the session, summarise:
   ```
   python scripts/build_text.py summarize configs/india.yaml --chunk 200
   python scripts/build_text.py summarize configs/india.yaml --raw
   python scripts/build_text.py embed configs/india.yaml
   python scripts/build_text.py embed configs/india.yaml --raw
   ```
   Checkpointed every 200 stock-days; killing it is safe, the same command resumes.

4. `python -m pytest tests/ -q` — 24 guards, ~30 s. All must pass on this machine too.

5. `python scripts/run_all.py --only full,price_only,shuffled_text`
   Three ablations, not the full eight. These three answer the entire question ("does text
   help, and is the result real?"); the other five are refinements and cost ~5x more GPU time.
   **Stop immediately if the `shuffled_text` control does not land near chance** — that means
   the text stream is being exploited through a leak, and every further run would be an
   ablation of a broken pipeline.

**Calibration, so you know what "working" looks like:** expected range on NSE data is 53-57%
accuracy, MCC 0.05-0.15. The paper's 67% is an upper bound on a denser corpus, not a target.
**Anything near 60% should be treated as a leak until proven otherwise.** After filtering,
text covers only 23.4% of stock-days (median 6 news days per 30-day window), so a null result
— `full` tying `price_only` — is a legitimate outcome and must be reported as one, not tuned
away.

**Bring home:** `data/interim/daily_summaries.csv` (+`_raw`),
`data/interim/daily_embeddings.npz` (+`_raw`), `artifacts/results_*.json`. The `.pt`
checkpoints are optional.

Tell me the preflight numbers before you start step 3. Don't skip ahead.
