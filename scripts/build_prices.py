"""Stage 1: download NSE prices, clean them, and write the featurised panel."""
import sys, yaml, pandas as pd
sys.path.insert(0, ".")
from src.data.prices import load_universe, download, clean, featurize

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "configs/india.yaml"))
d = cfg["data"]
syms = load_universe(d["universe_file"], d["suffix"])
print(f"universe: {len(syms)} NSE symbols")
raw = download(syms, d["start"], d["end"])
df = featurize(clean(raw, cfg, d["train_end"]), d["train_end"])
df.to_csv(d["price_csv"], index=False)
print(f"wrote {d['price_csv']}: {len(df):,} rows, {df.symbol.nunique()} symbols, "
      f"up-share {df.label.mean():.3f}")
