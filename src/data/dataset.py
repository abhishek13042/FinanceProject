"""Windowed samples: an L-day price window + the aligned daily text embeddings.

Windows are stored as *indices into per-symbol daily matrices*, not as materialised tensors.
Materialising them costs `n_windows x L x 768` floats — 5.8 GB for the training split alone,
because consecutive windows overlap in 29 of their 30 days and each copy is paid for in full.
Indexing costs `n_days x 768`, about 300 MB for the whole panel, and slices are free.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.prices import FEATURE_COLS as PRICE_COLS  # 17 channels (6 OHLCV + 11 TA)


class StockWindowDataset(Dataset):
    def __init__(self, prices: pd.DataFrame, emb: dict, seq_len: int, text_dim: int):
        """prices: cleaned+featurised long frame. emb: {(symbol, date_str): vector}."""
        self.L, self.text_dim = seq_len, text_dim
        self.px, self.tx = [], []          # per symbol: (n_days, d_p) and (n_days, text_dim)
        idx, y, dates, syms = [], [], [], []
        self._by_day = {}                  # date -> [(symbol_i, row), ...], for shuffle_text
        zero = np.zeros(text_dim, dtype=np.float32)

        for si, (sym, g) in enumerate(prices.sort_values("date").groupby("symbol")):
            g = g.reset_index(drop=True)
            ds = g["date"].dt.strftime("%Y-%m-%d").to_numpy()
            self.px.append(g[PRICE_COLS].to_numpy(np.float32))
            self.tx.append(np.stack([emb.get((sym, d), zero) for d in ds]).astype(np.float32))
            lab = g["label"].to_numpy(np.int64)
            for r, d in enumerate(ds):
                self._by_day.setdefault(d, []).append((si, r))
            for i in range(seq_len - 1, len(g) - 1):     # last day has no next-day label
                idx.append((si, i)); y.append(lab[i])
                # the window's last trading day: the day the prediction is *made* on
                dates.append(ds[i]); syms.append(sym)

        self.index = np.asarray(idx, dtype=np.int32)
        self.y = np.asarray(y, np.float32)
        # Kept per sample so evaluation can block-bootstrap over trading days (spec 05 R7)
        # and split metrics by market-agreement day (spec 06). Loaders for val/test run with
        # shuffle=False, so these stay aligned with prediction order.
        self.dates, self.syms = np.asarray(dates), np.asarray(syms)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, k):
        si, i = self.index[k]
        s = i - self.L + 1
        return (torch.from_numpy(self.px[si][s:i + 1]),
                torch.from_numpy(self.tx[si][s:i + 1]),
                torch.tensor(self.y[k]))

    def text_window(self, k) -> np.ndarray:
        """The text tensor sample `k` will be handed. Used by the leakage guards."""
        si, i = self.index[k]
        return self.tx[si][i - self.L + 1:i + 1]

    def zero_text(self) -> "StockWindowDataset":
        """price-only ablation: the text stream carries no information at all."""
        self.tx = [np.zeros_like(a) for a in self.tx]
        return self

    def shuffle_text(self, seed: int = 0) -> "StockWindowDataset":
        """Negative control (spec 05). Permute text *within* each trading day, so every window
        keeps a real, same-day summary belonging to a different stock. Date alignment survives;
        stock alignment does not. Accuracy must collapse to chance - if it does not, the text
        stream is being exploited through a leak rather than through signal."""
        rng = np.random.default_rng(seed)
        new = [a.copy() for a in self.tx]
        for entries in self._by_day.values():
            if len(entries) > 1:
                for (si, r), p in zip(entries, rng.permutation(len(entries))):
                    sj, r2 = entries[p]
                    new[si][r] = self.tx[sj][r2]
        self.tx = new
        return self


def chronological_split(df: pd.DataFrame, cfg: dict):
    """Strict time split - no overlap, no leakage across the boundaries."""
    d = cfg["data"]
    tr = df[df["date"] <= pd.Timestamp(d["train_end"])]
    va = df[(df["date"] > pd.Timestamp(d["train_end"])) & (df["date"] <= pd.Timestamp(d["val_end"]))]
    te = df[(df["date"] > pd.Timestamp(d["val_end"])) & (df["date"] <= pd.Timestamp(d["test_end"]))]
    return tr, va, te


def load_embeddings(npz_path: str, text_dim: int) -> dict:
    z = np.load(npz_path, allow_pickle=True)
    return {tuple(k.split("|")): v for k, v in zip(z["keys"], z["vecs"])}
