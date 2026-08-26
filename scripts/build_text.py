"""Stage 2 (spec 03): hierarchical summarisation + embedding extraction.

Modes:
  preflight   load the model, summarise a few stock-days, print samples and a time estimate.
              Run this BEFORE booking GPU time - it is what catches a broken corpus, a
              degenerate-filing summary, or a wrong key format while it is still cheap.
  summarize   the real backfill. Resumable: safe to kill and restart (spec 03 R6).
  embed       text -> vectors. CPU is fine; do not spend GPU time here.

  python scripts/build_text.py preflight [configs/india.yaml] [--n 8]
  python scripts/build_text.py summarize [configs/india.yaml] [--chunk 200] [--raw]
  python scripts/build_text.py embed     [configs/india.yaml] [--raw]

--raw selects the ablation variant (concatenated headlines, no LLM generation) which spec 05
needs for the "no hierarchical summary" row. Produce it in the same sitting as the summaries.
"""
import argparse, os, sys, time
import numpy as np, pandas as pd, yaml

sys.path.insert(0, ".")
from src.summarize.hierarchical import (HierarchicalSummarizer, attribution_date,
                                        bucket_by_session, dedup)
from src.summarize.runner import embed_texts, raw_variant, run_summarization


def load_news(cfg):
    """Corpus -> symbol, timestamp, text, date (attribution-corrected), bucket."""
    d = cfg["data"]
    news = pd.read_csv(d["news_csv"])
    body = news["body"] if "body" in news.columns else pd.Series("", index=news.index)
    news["text"] = (news["headline"].fillna("") + ". " + body.fillna("")).str.strip()
    news["symbol"] = news["symbol"].astype(str).str.upper().apply(
        lambda s: s if s.endswith((".NS", ".BO")) else s + d["suffix"])

    prices = pd.read_csv(d["price_csv"], parse_dates=["date"])
    cal = pd.DatetimeIndex(sorted(prices["date"].unique()))
    # post-close news belongs to the NEXT session; this is the project's core no-look-ahead rule
    news["date"] = attribution_date(news["timestamp"], cal, cfg)
    news["bucket"] = bucket_by_session(news["timestamp"], cfg)
    news = dedup(news, d["dedup_ratio"], d["min_chars"])
    return news, cal


def paths(cfg, raw):
    d = cfg["data"]
    if not raw:
        return d["summary_csv"], d["embed_npz"]
    stem = lambda p, s: os.path.splitext(p)[0] + s + os.path.splitext(p)[1]
    return stem(d["summary_csv"], "_raw"), stem(d["embed_npz"], "_raw")


def cmd_preflight(cfg, args):
    news, cal = load_news(cfg)
    units = news[["symbol", "date"]].drop_duplicates()
    print(f"corpus: {len(news):,} articles | {news.symbol.nunique()} symbols | "
          f"{len(units):,} stock-days | calendar {len(cal):,} trading days")
    cov = len(units) / (news.symbol.nunique() * len(cal))
    print(f"coverage: {cov:.1%} of stock-days carry news "
          f"(the rest resolve to the zero vector, spec 03 acceptance)")

    sample = units.head(args.n)
    sub = news.merge(sample, on=["symbol", "date"])
    print(f"\nloading {cfg['summarizer']['model']} ...")
    t0 = time.time()
    hs = HierarchicalSummarizer(cfg)
    print(f"loaded in {time.time() - t0:.1f}s on device={hs.dev}")

    t0 = time.time()
    out = hs.summarize(sub)
    per_day = (time.time() - t0) / max(len(out), 1)
    print(f"summarised {len(out)} stock-days in {time.time() - t0:.1f}s "
          f"({per_day:.2f} s/stock-day)")
    print(f"ESTIMATE for full backfill: {per_day * len(units) / 3600:.1f} GPU-hours "
          f"({len(units):,} stock-days)")

    print("\n--- sample outputs (check these are not boilerplate) ---")
    for _, r in out.head(3).iterrows():
        src = sub[(sub.symbol == r.symbol) & (sub.date == r.date)]["text"].str.cat(sep=" ")
        print(f"\n[{r.symbol} {pd.Timestamp(r.date).date()}]  {len(src)} chars in -> "
              f"{len(r.summary)} chars out")
        print(f"  IN : {src[:200]}...")
        print(f"  OUT: {r.summary[:200]}")
        if r.summary.strip()[:120] in src:
            print("  !! WARNING: summary is a verbatim slice of the input (spec 03 acceptance)")

    v = hs.embed(out["summary"].tolist())
    zero = int((np.abs(v).sum(1) == 0).sum())
    print(f"\nembeddings {v.shape}, all-zero rows: {zero} (must be 0 for covered stock-days)")
    key = f"{out.iloc[0].symbol}|{pd.Timestamp(out.iloc[0].date).strftime('%Y-%m-%d')}"
    print(f"key format: {key!r} -- must match StockWindowDataset's (symbol, date_str) lookup")
    print("\npreflight OK" if zero == 0 else "\npreflight FAILED: zero-vector rows present")


def cmd_summarize(cfg, args):
    news, _ = load_news(cfg)
    out_csv, _ = paths(cfg, args.raw)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if args.raw:                                    # no GPU needed for the ablation variant
        df = raw_variant(news)
        df.to_csv(out_csv, index=False)
        print(f"wrote {out_csv}: {len(df):,} stock-days (raw headlines, no generation)")
        return
    hs = HierarchicalSummarizer(cfg)
    print(f"device={hs.dev}")
    df = run_summarization(hs, news, out_csv, chunk_days=args.chunk)
    print(f"wrote {out_csv}: {len(df):,} stock-days")


def cmd_embed(cfg, args):
    text_csv, npz = paths(cfg, args.raw)
    hs = HierarchicalSummarizer(cfg)
    print(f"device={hs.dev}")
    embed_texts(hs, text_csv, npz)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["preflight", "summarize", "embed"])
    p.add_argument("config", nargs="?", default="configs/india.yaml")
    p.add_argument("--n", type=int, default=8, help="preflight sample size, in stock-days")
    p.add_argument("--chunk", type=int, default=200, help="stock-days per checkpoint flush")
    p.add_argument("--raw", action="store_true", help="ablation variant: raw headlines")
    a = p.parse_args()
    cfg = yaml.safe_load(open(a.config))
    {"preflight": cmd_preflight, "summarize": cmd_summarize, "embed": cmd_embed}[a.mode](cfg, a)
