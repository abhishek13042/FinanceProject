"""Synthetic end-to-end check: no network, no LLM. Verifies shapes and the training loop."""
import sys, numpy as np, pandas as pd, torch, yaml
sys.path.insert(0, ".")
from src.data.dataset import StockWindowDataset, chronological_split
from src.models.model import StockTrendModel

cfg = yaml.safe_load(open("configs/india.yaml"))
cfg["model"]["text_dim"] = 64
rng = np.random.default_rng(0)
dates = pd.bdate_range("2023-01-02", periods=300)
rows = []
for s in ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]:
    px = 100 + np.cumsum(rng.normal(0, 1, len(dates)))
    for i, dt in enumerate(dates):
        rows.append(dict(symbol=s, date=dt, close=px[i], logvol=rng.normal(), ret=rng.normal(),
                         range=abs(rng.normal()), ma_gap=rng.normal()))
df = pd.DataFrame(rows)
df["label"] = (df.groupby("symbol")["close"].shift(-1) > df["close"]).astype(int)
emb = {(s, d.strftime("%Y-%m-%d")): rng.normal(size=64).astype(np.float32)
       for s in df.symbol.unique() for d in dates}

cfg["data"].update(train_end="2023-08-31", val_end="2023-10-31", test_end="2026-01-01")
tr, va, te = chronological_split(df, cfg)
ds = StockWindowDataset(tr, emb, cfg["model"]["seq_len"], 64)
x, t, y = ds[0]
print("split sizes:", len(tr), len(va), len(te), "| train windows:", len(ds))
print("price window", tuple(x.shape), "text window", tuple(t.shape), "label", y.item())
m = StockTrendModel(cfg["model"])
logits = m(x[None], t[None])
loss = torch.nn.BCEWithLogitsLoss()(logits, y[None])
loss.backward()
gn = sum(p.grad.norm().item() for p in m.parameters() if p.grad is not None)
print(f"logits {tuple(logits.shape)} loss {loss.item():.4f} grad-norm {gn:.3f}")
print("N patches =", m.align.N, "| OK")
