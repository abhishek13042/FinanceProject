"""End-to-end training and reporting. Only the encoder/fusion/classifier are trained; the
summarisation LLM stays frozen and its embeddings are precomputed (see scripts/).

A run here is a *reportable unit*, not a single fit: >=5 seeds (spec 05 R6), block-bootstrap
intervals over trading days (R7), the constant-up floor (R5), and the beta-contamination
split (spec 06). Nothing is written that could be quoted as a headline number without them.
"""
import argparse, copy, json, os, time
import numpy as np, pandas as pd, torch, yaml
from torch.utils.data import DataLoader

from src.data.dataset import StockWindowDataset, chronological_split, load_embeddings
from src.data.options import OPT_COLS, attach as attach_options
from src.data.prices import FEATURE_COLS as PRICE_COLS
from src.eval import (across_seeds, agreement_days, block_bootstrap, constant_up, metrics,
                      split_by_agreement)
from src.models.model import StockTrendModel

# Each ablation is a (config override, dataset transform) pair. Keeping them declarative
# means the ablation table in spec 05 maps one-to-one onto runnable names.
ABLATIONS = {
    "full":          ({}, None),
    "price_only":    ({}, "zero_text"),
    "shuffled_text": ({}, "shuffle_text"),          # negative control - must land at chance
    "concat":        ({"fusion": "concat"}, None),
    "oneway":        ({"fusion": "oneway"}, None),
    "layers2":       ({"n_layers": 2}, None),
    "align32":       ({"d_align": 32}, None),
    "align64":       ({"d_align": 64}, None),
    # Option-chain channels appended to the price stream (src/data/options.py). Compare
    # opt_price_only vs price_only (do options add signal?) and opt_full vs full.
    "opt_price_only": ({"use_options": True}, "zero_text"),
    "opt_full":       ({"use_options": True}, None),
}


def predict(model, loader, dev):
    model.eval(); P, Y = [], []
    with torch.no_grad():
        for x, t, y in loader:
            P.append(torch.sigmoid(model(x.to(dev), t.to(dev))).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)


def build_data(cfg, require_text=True):
    prices = pd.read_csv(cfg["data"]["price_csv"], parse_dates=["date"])
    npz = cfg["data"]["embed_npz"]
    if not require_text and not os.path.exists(npz):
        # The price-only baseline zeroes the text stream anyway, so it does not need the
        # embeddings to exist. This lets the R5 baseline be produced on a CPU laptop before
        # the GPU summarisation session has ever run - it is a real result, not a stand-in.
        print(f"note: {npz} absent; running with an empty text stream (price-only)")
        emb = {}
    else:
        emb = load_embeddings(npz, cfg["model"]["text_dim"])
    cols = list(PRICE_COLS)
    if cfg["model"].get("use_options"):
        feats = pd.read_csv(cfg["data"]["option_features"], parse_dates=["date"])
        prices = attach_options(prices, feats, cfg["data"]["train_end"])
        cols += OPT_COLS
        print(f"options: {len(OPT_COLS)} channels, has_opt on {prices['has_opt'].mean():.1%} "
              f"of stock-days")
    cfg["model"]["n_price_feats"] = len(cols)
    L, td = cfg["model"]["seq_len"], cfg["model"]["text_dim"]
    splits = [StockWindowDataset(d, emb, L, td, cols) for d in chronological_split(prices, cfg)]
    return prices, splits


def apply_ablation(splits, transform, seed):
    """Transforms apply to every split, so train and test see the same world. Shuffling only
    the test set would measure a distribution shift, not the absence of signal."""
    if transform is None:
        return splits
    out = []
    for ds in splits:
        ds = copy.copy(ds)                      # shallow: arrays are replaced, not mutated
        out.append(ds.shuffle_text(seed) if transform == "shuffle_text" else ds.zero_text())
    return out


def run_seed(cfg, splits, seed, dev, tag, log=print):
    torch.manual_seed(seed); np.random.seed(seed)
    t = cfg["train"]
    mk = lambda d, sh: DataLoader(d, batch_size=t["batch_size"], shuffle=sh, drop_last=sh)
    dl_tr, dl_va, dl_te = mk(splits[0], True), mk(splits[1], False), mk(splits[2], False)

    model = StockTrendModel(cfg["model"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"])
    sched = torch.optim.lr_scheduler.StepLR(opt, t["decay_every"], gamma=1 - t["lr_decay"])
    lossf = torch.nn.BCEWithLogitsLoss()
    # Tagged per ablation: a bare best_seed<N>.pt is shared by every ablation, so a run
    # that never improves would silently load the PREVIOUS ablation's weights (identical
    # shapes for full/price_only/shuffled_text) or die on a shape mismatch (align32/64).
    ckpt = os.path.join(t["out_dir"], f"best_{tag}_seed{seed}.pt")

    best, bad = -1e9, 0
    for ep in range(1, t["epochs"] + 1):
        model.train(); tot = 0.0
        for x, tx, y in dl_tr:
            opt.zero_grad()
            loss = lossf(model(x.to(dev), tx.to(dev)), y.to(dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item()
        sched.step()
        vp, vy = predict(model, dl_va, dev)
        m = metrics(vy, vp)
        log(f"  seed {seed} epoch {ep:02d} loss {tot/max(len(dl_tr),1):.4f} "
            f"val_acc {m['acc']:.2f} val_mcc {m['mcc']:.3f}")
        # `or not exists`: a degenerate first epoch (MCC nan) would otherwise leave no
        # checkpoint at all and crash the load below after a full training run.
        if m["mcc"] > best or not os.path.exists(ckpt):   # R2: selection on val MCC only
            best, bad = m["mcc"], 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= t["patience"]:
                log(f"  seed {seed} early stop at epoch {ep}")
                break

    model.load_state_dict(torch.load(ckpt))
    prob, y = predict(model, dl_te, dev)         # R3: test set touched once, after selection
    return prob, y, best


def main(cfg_path, ablation, seeds, n_boot, tag):
    cfg = yaml.safe_load(open(cfg_path))
    overrides, transform = ABLATIONS[ablation]
    cfg["model"].update(overrides)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = cfg["train"]["out_dir"]
    os.makedirs(out, exist_ok=True)
    tag = tag or ablation

    prices, splits = build_data(cfg, require_text=(transform != "zero_text"))
    dates = splits[2].dates
    hi_days = agreement_days(prices)
    print(f"[{tag}] device={dev} overrides={overrides or '-'} transform={transform or '-'} "
          f"| train/val/test windows {[len(s) for s in splits]}")

    per_seed, probs, t0 = [], [], time.time()
    for s in seeds:
        # Seed-level resume. An ablation is 5 seeds x up to 20 epochs; without this, a crash
        # on the last seed discards every earlier one. Test predictions are all that a seed
        # contributes to the reported numbers, so caching them is equivalent to caching the
        # run - and far smaller than the checkpoint. Cheap insurance on borrowed hardware.
        cache = os.path.join(out, f"seed_{tag}_{s}.npz")
        if os.path.exists(cache):
            z = np.load(cache)
            prob, y, val_mcc = z["prob"], z["y"], float(z["val_mcc"])
            print(f"[{tag}] seed {s} restored from {cache}")
        else:
            ds = apply_ablation(splits, transform, s)
            prob, y, val_mcc = run_seed(cfg, ds, s, dev, tag)
            np.savez(cache, prob=prob, y=y, val_mcc=val_mcc)
        m = metrics(y, prob)
        m["val_mcc"] = val_mcc
        per_seed.append(m); probs.append(prob)
        print(f"[{tag}] seed {s} TEST acc {m['acc']:.2f} mcc {m['mcc']:.3f} f1 {m['f1']:.3f}")

    # Intervals are computed on the seed-averaged probability. Seed variance is reported
    # separately (mean +/- std); pooling the two would double-count the same uncertainty.
    prob_bar = np.mean(probs, axis=0)
    cfg_note = {k: cfg["model"][k] for k in ("n_layers", "d_align", "n_price_feats")}
    cfg_note["fusion"] = cfg["model"].get("fusion", "coattention")
    res = {
        "tag": tag, "ablation": ablation, "overrides": overrides, "transform": transform,
        "seeds": list(seeds), "n_boot": n_boot, "minutes": round((time.time() - t0) / 60, 1),
        "config": cfg_note,
        "test_mean_over_seeds": across_seeds(per_seed),                           # R6 / G1
        "test_seed_averaged": metrics(y, prob_bar),
        "ci_block_bootstrap_days": block_bootstrap(y, prob_bar, dates, n_boot),   # R7 / G2
        "baseline_constant_up": constant_up(y),                                   # R5 / G4
        "beta_contamination": split_by_agreement(y, prob_bar, dates, hi_days),    # spec 06
        "n_test_days": int(len(np.unique(dates))),
        "high_agreement_day_frac": round(float(np.isin(dates, list(hi_days)).mean()), 3),
    }
    path = f"{out}/results_{tag}.json"
    json.dump(res, open(path, "w"), indent=2)

    ci, ms = res["ci_block_bootstrap_days"], res["test_mean_over_seeds"]
    print(f"\n[{tag}] acc {ms['acc']['mean']:.2f} +/- {ms['acc']['std']:.2f} "
          f"(95% CI {ci['acc']['lo']:.2f}-{ci['acc']['hi']:.2f})")
    print(f"[{tag}] mcc {ms['mcc']['mean']:.3f} +/- {ms['mcc']['std']:.3f} "
          f"(95% CI {ci['mcc']['lo']:.3f}-{ci['mcc']['hi']:.3f})")
    print(f"[{tag}] constant-up floor: acc {res['baseline_constant_up']['acc']:.2f}")
    b = res["beta_contamination"]
    if b.get("mcc_gap") is not None:
        print(f"[{tag}] beta gate: mcc {b['high_agreement']['mcc']:.3f} on agreement days vs "
              f"{b['low_agreement']['mcc']:.3f} elsewhere (gap {b['mcc_gap']:+.3f})")
    print(f"[{tag}] wrote {path}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/india.yaml")
    ap.add_argument("--ablation", default="full", choices=sorted(ABLATIONS))
    ap.add_argument("--seeds", default="42,43,44,45,46",
                    help="comma-separated; spec 05 R6 requires >=5 for a reportable number")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    main(a.config, a.ablation, [int(s) for s in a.seeds.split(",")], a.n_boot, a.tag)
