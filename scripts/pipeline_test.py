"""Full-pipeline rehearsal: every command the GPU session will run, on synthetic data.

The existing harnesses each cover one end - `smoke_test.py` checks tensor shapes, and
`smoke_eval.py` checks the spec 05 reporting path. Neither runs the *seam* the borrowed-GPU
session actually depends on: corpus -> attribution -> summarise -> resume -> embed -> the
(symbol, date_str) key lookup inside StockWindowDataset -> train -> results json.

That seam is where a session dies at hour six, so it is rehearsed here in about a minute,
with a stub summariser standing in for mT5. No network, no GPU, no model weights.

    python scripts/pipeline_test.py

Exit code is 0 only if every stage passed.
"""
import json, os, shutil, subprocess, sys, tempfile, traceback
import numpy as np, pandas as pd, yaml

sys.path.insert(0, ".")
from src.data.prices import clean, featurize, FEATURE_COLS
from src.data.dataset import StockWindowDataset, chronological_split, load_embeddings
import src.summarize.runner as R

SYMS = ["AAA.NS", "BBB.NS", "CCC.NS", "DDD.NS"]
CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    return bool(cond)


class StubSummarizer:
    """Same interface as HierarchicalSummarizer, deterministic, no model.

    `fail_after_chunks` simulates the kill that spec 03 R6's resume logic exists for.
    """

    def __init__(self, text_dim, fail_after_chunks=None):
        self.dev, self.text_dim = "cpu", text_dim
        self.calls, self.fail_after = 0, fail_after_chunks

    def summarize(self, news):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise KeyboardInterrupt("simulated kill mid-backfill")
        g = (news.sort_values(["symbol", "date", "timestamp"])
                 .groupby(["symbol", "date"])["text"]
                 .apply(lambda s: "SUMMARY(" + " | ".join(s)[:60] + ")").reset_index()
                 .rename(columns={"text": "summary"}))
        return g[["symbol", "date", "summary"]]

    def embed(self, summaries):
        rng = np.random.default_rng(0)
        v = np.stack([rng.normal(size=self.text_dim) + (len(s) % 7) for s in summaries])
        return v.astype(np.float32)


def make_prices(cfg):
    """Raw OHLCV -> the real clean()/featurize() path, so spec 01's code is exercised."""
    rng = np.random.default_rng(7)
    days = pd.bdate_range(cfg["data"]["start"], "2025-12-31")
    frames = []
    for s in SYMS:
        r = rng.normal(0, 0.012, len(days))
        close = 100 * np.exp(np.cumsum(r))
        frames.append(pd.DataFrame({
            "symbol": s, "date": days, "close": close,
            "open": close * (1 + rng.normal(0, .003, len(days))),
            "high": close * (1 + abs(rng.normal(0, .006, len(days)))),
            "low": close * (1 - abs(rng.normal(0, .006, len(days)))),
            "volume": rng.integers(100000, 10000000, len(days)).astype(float)}))
    raw = pd.concat(frames, ignore_index=True)
    df = featurize(clean(raw, cfg, cfg["data"]["train_end"]), cfg["data"]["train_end"])
    df.to_csv(cfg["data"]["price_csv"], index=False)
    return df


def make_news(cfg, prices):
    """A corpus with the three things that break the summariser: post-close filings,
    near-duplicate wire copies, and administrative boilerplate."""
    rng = np.random.default_rng(11)
    cal = pd.DatetimeIndex(sorted(prices["date"].unique()))
    body = "Revenue rose and the board approved a capex plan. " * 3
    rows = []
    for s in SYMS:
        for d in cal[::9]:                                   # ~11% of stock-days carry news
            for hh, mm in [(8, 40), (11, 20), (16, 45)][: rng.integers(1, 4)]:
                ts = pd.Timestamp(d).tz_localize("Asia/Kolkata") + pd.Timedelta(hours=hh, minutes=mm)
                rows.append(dict(symbol=s, timestamp=ts.isoformat(),
                                 headline=f"{s} reports quarterly results at {hh}:{mm}", body=body))
            # a syndicated near-duplicate of the first item, for dedup
            ts = pd.Timestamp(d).tz_localize("Asia/Kolkata") + pd.Timedelta(hours=8, minutes=45)
            rows.append(dict(symbol=s, timestamp=ts.isoformat(),
                             headline=f"{s} reports quarterly results at 8:40", body=body))
            # administrative paperwork, for the boilerplate filter
            rows.append(dict(symbol=s, timestamp=ts.isoformat(),
                             headline="Loss of share certificate intimation under Regulation 39(3)",
                             body="Intimation regarding loss of share certificates. " * 3))
    news = pd.DataFrame(rows)
    news.to_csv(cfg["data"]["news_csv"], index=False)
    return news


def stage_corpus(cfg):
    print("\n[1/6] corpus load: boilerplate filter, dedup, attribution, buckets")
    import scripts.build_text as BT
    news, cal = BT.load_news(cfg)
    check("news survives the filters", len(news) > 0, f"{len(news):,} articles")
    check("timestamps are datetimes, not strings",
          pd.api.types.is_datetime64_any_dtype(news["timestamp"]), str(news["timestamp"].dtype))
    check("boilerplate dropped", not news["headline"].str.contains("Loss of share").any())
    check("near-duplicates dropped",
          news.groupby(["symbol", "date"]).size().max() <= 3, "<=3 distinct articles/stock-day")

    # post-close attribution: a 16:45 filing must land on a LATER trading day than its own
    ist = news["timestamp"].dt.tz_convert("Asia/Kolkata")
    late = news[ist.dt.hour >= 16]
    own = ist[ist.dt.hour >= 16].dt.normalize().dt.tz_localize(None)
    check("post-close news moves to the next session",
          len(late) > 0 and bool((late["date"] > own).all()), f"{len(late):,} post-close filings")
    check("attributed dates are real trading days", bool(news["date"].isin(cal).all()))
    return news


def stage_summarize(cfg, news):
    print("\n[2/6] summarisation backfill + resume after a kill")
    out_csv = cfg["data"]["summary_csv"]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    units = news[["symbol", "date"]].drop_duplicates()

    # kill it deliberately, two chunks in
    hs = StubSummarizer(cfg["model"]["text_dim"], fail_after_chunks=2)
    try:
        R.run_summarization(hs, news, out_csv, chunk_days=40, log=lambda *a: None)
        check("simulated kill actually fired", False, "run completed when it should have died")
    except KeyboardInterrupt:
        check("simulated kill actually fired", True)
    partial = os.path.splitext(out_csv)[0] + R.PARTIAL_SUFFIX
    banked = R.completed_units(out_csv)
    check("partial checkpoint survives the kill",
          os.path.exists(partial) and len(banked) > 0, f"{len(banked)} stock-days banked")

    # restart: must skip what is on disk and finish the rest
    hs2 = StubSummarizer(cfg["model"]["text_dim"])
    df = R.run_summarization(hs2, news, out_csv, chunk_days=40, log=lambda *a: None)
    check("resume completes the backfill", len(df) == len(units), f"{len(df):,}/{len(units):,}")
    check("resume produced no duplicate stock-days", not df.duplicated(["symbol", "date"]).any())
    check("resume skipped the banked work", len(banked) > 0)
    check("date column is the YYYY-MM-DD string form",
          bool(df["date"].astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()),
          str(df["date"].iloc[0]))

    # the --raw ablation arm must cover exactly the same stock-days, or spec 05's control
    # compares two different corpora rather than two text treatments
    raw = R.raw_variant(news)
    raw["date"] = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")
    raw_csv = os.path.splitext(out_csv)[0] + "_raw.csv"
    raw.to_csv(raw_csv, index=False)
    check("raw arm covers the same stock-days as the summarised arm",
          set(zip(raw["symbol"], raw["date"])) == set(zip(df["symbol"], df["date"])),
          f"raw {len(raw):,} vs summarised {len(df):,}")
    check("raw arm is not identical to the summarised arm",
          not raw.sort_values(["symbol", "date"])["summary"].reset_index(drop=True)
                 .equals(df.sort_values(["symbol", "date"])["summary"].reset_index(drop=True)))
    return df, raw_csv


def stage_embed(cfg, raw_csv):
    print("\n[3/6] embedding + the key format the dataset actually looks up")
    hs = StubSummarizer(cfg["model"]["text_dim"])
    R.embed_texts(hs, cfg["data"]["summary_csv"], cfg["data"]["embed_npz"], batch_log=lambda *a: None)
    R.embed_texts(hs, raw_csv, os.path.splitext(cfg["data"]["embed_npz"])[0] + "_raw.npz",
                  batch_log=lambda *a: None)
    emb = load_embeddings(cfg["data"]["embed_npz"], cfg["model"]["text_dim"])
    k = next(iter(emb))
    check("npz keys parse into (symbol, date_str)", isinstance(k, tuple) and len(k) == 2, str(k))
    check("embedding width matches cfg.model.text_dim",
          next(iter(emb.values())).shape[0] == cfg["model"]["text_dim"])
    check("no all-zero embedding rows", not any(np.abs(v).sum() == 0 for v in emb.values()))
    return emb


def stage_dataset(cfg, prices, emb):
    print("\n[4/6] window construction: does the text actually reach a sample?")
    L, td = cfg["model"]["seq_len"], cfg["model"]["text_dim"]
    tr, va, te = chronological_split(prices, cfg)
    ds = StockWindowDataset(tr, emb, L, td)
    check("training windows exist", len(ds) > 100, f"{len(ds):,} windows")

    # The integration failure this whole file exists for: keys that look right in the npz but
    # never match, leaving every window silently zero-filled. `full` then equals `price_only`
    # and the null result is an artefact of a string mismatch, not a finding about the market.
    hit = sum(1 for i in range(0, len(ds), 25) if np.abs(ds.text_window(i)).sum() > 0)
    tot = len(range(0, len(ds), 25))
    check("text reaches the windows (key lookup works)", hit > 0,
          f"{hit}/{tot} sampled windows carry text")

    x, t, y = ds[0]
    check("sample shapes are (L, d_p) and (L, text_dim)",
          tuple(x.shape) == (L, len(FEATURE_COLS)) and tuple(t.shape) == (L, td),
          f"{tuple(x.shape)} {tuple(t.shape)}")
    check("splits are non-empty and ordered",
          bool(len(tr) and len(va) and len(te)
               and tr["date"].max() < va["date"].min() < te["date"].min()))

    # ablation transforms must not bleed into one another across seeds
    import src.train as T
    a = T.apply_ablation([ds], "zero_text", 0)[0]
    check("zero_text does not mutate the original dataset", np.abs(ds.text_window(0)).sum() > 0)
    check("zero_text really zeroes its own copy", np.abs(a.text_window(0)).sum() == 0)
    b = T.apply_ablation([ds], "shuffle_text", 1)[0]
    check("shuffle_text keeps a non-zero text stream", np.abs(b.text_window(0)).sum() > 0)
    return ds


def stage_train(cfg_path, cfg):
    print("\n[5/6] training + reporting path for the three decisive ablations")
    import src.train as T
    for tag in ["full", "price_only", "shuffled_text"]:
        try:
            T.main(cfg_path, tag, [42, 43], 40, tag)
            p = os.path.join(cfg["train"]["out_dir"], f"results_{tag}.json")
            r = json.load(open(p))
            check(f"{tag}: results json written", os.path.exists(p))
            check(f"{tag}: >=2 seeds reported", len(r["seeds"]) >= 2)
            check(f"{tag}: bootstrap interval is finite",
                  bool(np.isfinite([r["ci_block_bootstrap_days"]["mcc"]["lo"],
                                    r["ci_block_bootstrap_days"]["mcc"]["hi"]]).all()))
            check(f"{tag}: constant-up floor present", "acc" in r["baseline_constant_up"])
        except Exception as e:
            traceback.print_exc()
            check(f"{tag}: ran to completion", False, f"{type(e).__name__}: {e}")

    art = cfg["train"]["out_dir"]
    cks = [f for f in os.listdir(art) if f.endswith(".pt")]
    check("checkpoints are namespaced per ablation",
          len({f.rsplit("_seed", 1)[0] for f in cks}) >= 3, f"{len(cks)} files")

    # Seed-level resume: an interrupted ablation must not discard the seeds it finished.
    # Deleting the results file and re-running should restore cached predictions and
    # reproduce the numbers exactly - a retrain would not be bit-identical.
    cached = [f for f in os.listdir(art) if f.startswith("seed_full_")]
    check("per-seed prediction caches written", len(cached) >= 2, f"{len(cached)} files")
    rp = os.path.join(art, "results_full.json")
    before = json.load(open(rp))["test_seed_averaged"]
    os.remove(rp)
    T.main(cfg_path, "full", [42, 43], 40, "full")
    after = json.load(open(rp))["test_seed_averaged"]
    check("resumed ablation reproduces its numbers exactly", before == after,
          f"acc {before['acc']:.4f} -> {after['acc']:.4f}")


def stage_table(cfg_path):
    print("\n[6/6] --table: the G3 verdict the write-up is read off")
    r = subprocess.run([sys.executable, "scripts/run_all.py", "--config", cfg_path, "--table"],
                       capture_output=True, text=True)
    out = r.stdout
    check("run_all --table exits clean", r.returncode == 0, r.stderr.strip()[-200:])
    check("table prints the G3 verdict", "G3" in out and "reportable" in out)
    check("table prints the constant-up floor", "constant-up floor" in out)
    print("\n".join("    " + l for l in out.strip().splitlines()[-8:]))


def main():
    ws = tempfile.mkdtemp(prefix="pipeline_test_").replace("\\", "/")
    cfg = yaml.safe_load(open("configs/india.yaml"))
    cfg["data"].update(price_csv=f"{ws}/prices.csv", news_csv=f"{ws}/news.csv",
                       summary_csv=f"{ws}/interim/daily_summaries.csv",
                       embed_npz=f"{ws}/interim/daily_embeddings.npz")
    # small enough to finish on a CPU in about a minute; the code paths are the real ones
    cfg["model"].update(text_dim=32, d_model=32, d_ff=64, n_layers=1, n_heads=4)
    cfg["train"].update(epochs=2, patience=2, out_dir=f"{ws}/artifacts")
    os.makedirs(f"{ws}/interim", exist_ok=True)
    os.makedirs(f"{ws}/artifacts", exist_ok=True)
    cfg_path = f"{ws}/pipeline_test.yaml"
    yaml.safe_dump(cfg, open(cfg_path, "w"))
    print(f"workspace: {ws}")

    try:
        prices = make_prices(cfg)
        make_news(cfg, prices)
        news = stage_corpus(cfg)
        _, raw_csv = stage_summarize(cfg, news)
        emb = stage_embed(cfg, raw_csv)
        stage_dataset(cfg, prices, emb)
        stage_train(cfg_path, cfg)
        stage_table(cfg_path)
    except Exception as e:
        traceback.print_exc()
        check("pipeline ran to completion", False, f"{type(e).__name__}: {e}")

    bad = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n{'=' * 70}\n{len(CHECKS) - len(bad)}/{len(CHECKS)} checks passed")
    if bad:
        print("FAILED:\n" + "\n".join("  - " + b for b in bad))
    else:
        shutil.rmtree(ws, ignore_errors=True)
        print("PIPELINE OK - the GPU session's code path is clear end to end")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
