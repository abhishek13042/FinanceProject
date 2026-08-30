"""Spec 03 R6 - resumable orchestration for the summarisation backfill.

A full backfill is many GPU-hours on borrowed hardware, so this module assumes the run
WILL be interrupted. Work is chunked by stock-day, every completed chunk is flushed to an
append-only checkpoint, and a restart skips whatever is already on disk.

Two text variants are produced in the same pass, deliberately:
  - hierarchical summaries (the paper's Algorithm 1)
  - raw concatenated headlines (spec 05's "no hierarchical summary" ablation)
Emitting both here costs one extra file and saves a second borrowed-GPU session.

Summaries are written as TEXT, never as vectors. Embedding is a separate, cheap step that
runs fine on CPU, so the scarce GPU time is spent only on generation.
"""
import os
import numpy as np
import pandas as pd

PARTIAL_SUFFIX = ".partial.csv"


def _partial_path(out_csv: str) -> str:
    return os.path.splitext(out_csv)[0] + PARTIAL_SUFFIX


def completed_units(out_csv: str) -> set:
    """(symbol, date_str) pairs already summarised in a previous run."""
    p = _partial_path(out_csv)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return set()
    df = pd.read_csv(p, usecols=["symbol", "date"], dtype=str)
    return set(zip(df["symbol"], df["date"]))


def _flush(rows: pd.DataFrame, path: str) -> None:
    """Append-only write; header only on first touch. Fsync so a hard kill cannot truncate."""
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        rows.to_csv(f, index=False, header=new)
        f.flush()
        os.fsync(f.fileno())


def raw_variant(news: pd.DataFrame, max_chars: int = 2000) -> pd.DataFrame:
    """Ablation control: the same stock-days, but with headlines concatenated in
    chronological order instead of summarised. No LLM generation involved."""
    g = (news.sort_values(["symbol", "date", "timestamp"])   # chronological; see hierarchical.py
             .groupby(["symbol", "date"])["text"].apply(lambda s: " ".join(s)[:max_chars])
             .reset_index().rename(columns={"text": "summary"}))
    return g[["symbol", "date", "summary"]]


def run_summarization(hs, news: pd.DataFrame, out_csv: str, chunk_days: int = 200,
                      log=print) -> pd.DataFrame:
    """Summarise every (symbol, date) in `news`, resuming from a partial checkpoint.

    `chunk_days` trades restart granularity against generation throughput: a chunk is the
    unit of lost work if the process dies, and also the batch the two passes see.
    """
    news = news.copy()
    news["date"] = pd.to_datetime(news["date"]).dt.strftime("%Y-%m-%d")
    units = news[["symbol", "date"]].drop_duplicates()

    done = completed_units(out_csv)
    todo = units[~units.apply(lambda r: (r["symbol"], r["date"]) in done, axis=1)]
    log(f"stock-days: {len(units):,} total | {len(done):,} already done | {len(todo):,} to do")
    if len(todo) == 0:
        log("nothing to do; assembling final output from checkpoint")
        return finalize(out_csv)

    partial = _partial_path(out_csv)
    keyed = news.set_index(["symbol", "date"]).sort_index()
    for i in range(0, len(todo), chunk_days):
        block = todo.iloc[i:i + chunk_days]
        sub = keyed.loc[list(map(tuple, block.values))].reset_index()
        out = hs.summarize(sub)                                   # passes 1 and 2
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        _flush(out[["symbol", "date", "summary"]], partial)
        log(f"  chunk {i // chunk_days + 1}/{-(-len(todo) // chunk_days)} "
            f"-> {len(out)} stock-days flushed ({i + len(block):,}/{len(todo):,})")
    return finalize(out_csv)


def finalize(out_csv: str) -> pd.DataFrame:
    """Promote the append-only checkpoint to the committed output, dropping any duplicate
    stock-days a resume may have produced (last write wins)."""
    df = pd.read_csv(_partial_path(out_csv), dtype={"symbol": str, "date": str})
    df = df.drop_duplicates(subset=["symbol", "date"], keep="last").sort_values(["symbol", "date"])
    df.to_csv(out_csv, index=False)
    return df


def embed_texts(hs, texts_csv: str, out_npz: str, batch_log=print) -> np.ndarray:
    """Text -> pooled encoder vectors. Cheap enough for CPU; keep it off the GPU budget."""
    df = pd.read_csv(texts_csv, dtype={"symbol": str, "date": str})
    df["summary"] = df["summary"].fillna("").astype(str)
    vecs = hs.embed(df["summary"].tolist())
    keys = np.array([f"{s}|{d}" for s, d in zip(df["symbol"], df["date"])])
    np.savez_compressed(out_npz, keys=keys, vecs=vecs.astype(np.float32))
    batch_log(f"wrote {out_npz}: {vecs.shape}")
    return vecs
