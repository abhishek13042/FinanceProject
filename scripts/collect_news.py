"""Build data/raw/news.csv - the input to the hierarchical summariser.

  python scripts/collect_news.py bse            # historical backfill (run once, then top up)
  python scripts/collect_news.py rss            # live snapshot (run on a schedule)

BSE gives you real filing timestamps back to ~2018, which is what the intraday-bucket pass
needs. RSS only accumulates forward from the day you start collecting.
"""
import sys, yaml, pandas as pd
sys.path.insert(0, ".")
from src.data.news_sources import (FEEDS, append_csv, collect_bse, company_names,
                                   fetch_rss, tag_symbols)
from src.data.prices import load_universe

mode = sys.argv[1] if len(sys.argv) > 1 else "bse"
cfg = yaml.safe_load(open(sys.argv[2] if len(sys.argv) > 2 else "configs/india.yaml"))
d = cfg["data"]
symbols = load_universe(d["universe_file"], d["suffix"])

if mode == "bse":
    df = collect_bse(symbols, d["start"], d["end"], d["suffix"])
    df["headline"] = df["headline"].fillna("")
    df["body"] = df.get("category", "").fillna("") + ". " + df["body"].fillna("")
    n = append_csv(df, d["news_csv"])
    print(f"\n{len(df):,} announcements collected -> {d['news_csv']} ({n:,} rows total)")
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
    sys.exit("mode must be 'bse' or 'rss'")
