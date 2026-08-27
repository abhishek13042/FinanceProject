"""Build data/raw/news.csv - the input to the hierarchical summariser.

  python scripts/collect_news.py bse            # historical backfill; resumable, re-run to continue
  python scripts/collect_news.py merge          # fold the checkpoint into news.csv, no network
  python scripts/collect_news.py rss            # live snapshot (run on a schedule)

BSE gives you real filing timestamps back to ~2018, which is what the intraday-bucket pass
needs. RSS only accumulates forward from the day you start collecting.
"""
import sys, yaml, pandas as pd
sys.path.insert(0, ".")
from src.data.news_sources import (FEEDS, append_csv, collect_bse, company_names,
                                   fetch_rss, read_partial, tag_symbols)
from src.data.prices import load_universe

# The backfill is long and the connection is unreliable, so progress has to be visible as it
# happens rather than at exit - block-buffered stdout would hide a working run for an hour.
sys.stdout.reconfigure(line_buffering=True)

mode = sys.argv[1] if len(sys.argv) > 1 else "bse"
cfg = yaml.safe_load(open(sys.argv[2] if len(sys.argv) > 2 else "configs/india.yaml"))
d = cfg["data"]
symbols = load_universe(d["universe_file"], d["suffix"])

if mode in ("bse", "merge"):
    # `merge` skips the network entirely and just folds whatever the last interrupted run
    # checkpointed into news.csv. Useful when the link is down but the partial file is not.
    df = (read_partial(d["news_csv"]) if mode == "merge"
          else collect_bse(symbols, d["start"], d["end"], d["suffix"], news_csv=d["news_csv"]))
    df["headline"] = df["headline"].fillna("")
    cat = df["category"].fillna("") if "category" in df else ""
    df["body"] = cat + ". " + df["body"].fillna("")
    n = append_csv(df, d["news_csv"])
    print(f"\n{len(df):,} announcements on disk -> {d['news_csv']} ({n:,} rows total)")
    print(f"covering {df['symbol'].nunique()} of {len(symbols)} symbols"
          + ("" if df["symbol"].nunique() >= len(symbols) else "; re-run to fetch the rest"))
    if len(df):
        post = (df["timestamp"].dt.hour >= 15).mean()
        print(f"post-close share: {post:.1%} (these are attributed to the NEXT session)")

elif mode == "rss":
    names = company_names(symbols, d["suffix"])
    print(f"tagging against {len(names)} company names")
    frames = []
    for tag, url in FEEDS.items():
        try:
            raw = fetch_rss(url)
            hit = tag_symbols(raw, names, d["suffix"])
            print(f"  {tag}: {len(raw)} items -> {len(hit)} symbol-tagged")
            frames.append(hit)
        except Exception as e:
            print(f"  {tag}: {type(e).__name__} {str(e)[:60]}")
    if frames:
        n = append_csv(pd.concat(frames, ignore_index=True), d["news_csv"])
        print(f"-> {d['news_csv']} ({n:,} rows total)")
else:
    sys.exit("mode must be 'bse', 'merge' or 'rss'")
