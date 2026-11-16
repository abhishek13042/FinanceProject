"""Option backtest of the model's next-day direction calls, with full Indian costs.

For every test-set prediction (stock s, day t) the model says UP or DOWN for t -> t+1.
Two ways to express that view with stock options, nearest expiry >= MIN_DAYS away:
  A  buy  : UP -> buy the ATM call,  DOWN -> buy the ATM put
  B  sell : UP -> sell the ATM put,  DOWN -> sell the ATM call
Entry at the close of t, exit at the close of t+1 (the next trading day), 1 lot.

The ATM strike is the one nearest the forward (same-expiry futures close) among strikes
where BOTH the call and the put traded >= min_contracts on day t. Picking the strike the
same way for both directions means the direction choice cannot change which days trade.
Liquidity is only checked at entry - filtering on exit-day volume would peek ahead.

Costs per round trip (discount-broker retail, rates by date):
  brokerage  Rs 20 per executed order
  STT        on sell-side premium: 0.05%, 0.0625% from 2024-10-01
  exchange   NSE txn charge on premium: 0.0503%, 0.03503% from 2024-10-01
  SEBI fee   Rs 10 per crore of premium turnover
  stamp duty 0.003% of buy-side premium
  GST        18% on brokerage + exchange + SEBI fee
  slippage   each fill pays slip_pct of the premium + one tick (Rs 0.05) against us

Capital for return figures: the premium paid (buy), or margin_pct x notional (sell; SPAN +
exposure on single-stock options is typically 15-25% of notional).

Usage: python -m src.backtest --tag opt_full [--slip-pct 0.005] [--min-contracts 50]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import yaml

from src.data.dataset import StockWindowDataset, chronological_split
from src.data.options import forward, pick_expiries

TICK = 0.05
COST_SWITCH = pd.Timestamp("2024-10-01")


# ---------------------------------------------------------------- costs
def charges(buy_val: np.ndarray, sell_val: np.ndarray, dates: pd.Series) -> np.ndarray:
    """Statutory + broker charges (Rs) for one buy order and one sell order."""
    new = (pd.to_datetime(dates) >= COST_SWITCH).to_numpy()
    stt = np.where(new, 0.000625, 0.0005) * sell_val
    exch = np.where(new, 0.0003503, 0.000503) * (buy_val + sell_val)
    sebi = 10 / 1e7 * (buy_val + sell_val)
    stamp = 0.00003 * buy_val
    brok = 2 * 20.0
    gst = 0.18 * (brok + exch + sebi)
    return brok + stt + exch + sebi + stamp + gst


def fill(price: np.ndarray, buying: bool, slip_pct: float) -> np.ndarray:
    adj = price * slip_pct + (TICK if slip_pct > 0 else 0.0)
    return price + adj if buying else np.maximum(price - adj, 0.0)


# ---------------------------------------------------------------- trade table
def build_legs(samples: pd.DataFrame, options: pd.DataFrame, cal: dict,
               min_contracts: int) -> pd.DataFrame:
    """One row per sample: the ATM call/put at entry and the same contracts' next-day closes.
    Direction-independent, so the model and every baseline trade the same universe of days."""
    opts = options.copy()
    opts["symbol"] = opts["symbol"] + ".NS"
    by_day = {k: g for k, g in opts.groupby(["date", "symbol"])}
    rows = []
    for sym, d in zip(samples["symbol"], samples["date"]):
        r = {"symbol": sym, "date": d, "tradable": False}
        rows.append(r)
        nxt = cal.get(d)
        g, g1 = by_day.get((d, sym)), by_day.get((nxt, sym))
        if g is None or g1 is None:
            r["why"] = "no chain"
            continue
        opt, fut = g[g["inst"] == "OPT"], g[g["inst"] == "FUT"]
        ex, _ = pick_expiries(d, opt["expiry"])
        if ex is None:
            r["why"] = "no expiry"
            continue
        chain = opt[opt["expiry"] == ex]
        T = max((ex - d).days, 1) / 365
        F = forward(fut, chain, ex, T)
        pv = chain.pivot_table(index="strike", columns="opt_type",
                               values=["close", "contracts"], aggfunc="first")
        if not F > 0 or ("close", "CE") not in pv.columns or ("close", "PE") not in pv.columns:
            r["why"] = "no forward"
            continue
        ok = pv[(pv[("contracts", "CE")] >= min_contracts) & (pv[("contracts", "PE")] >= min_contracts)
                & (pv[("close", "CE")] > 0) & (pv[("close", "PE")] > 0)]
        if ok.empty:
            r["why"] = "illiquid"
            continue
        K = ok.index[np.argmin(np.abs(ok.index.to_numpy() - F))]
        nx = g1[(g1["inst"] == "OPT") & (g1["expiry"] == ex) & (g1["strike"] == K)]
        nx = nx.set_index("opt_type")
        if not {"CE", "PE"} <= set(nx.index):
            r["why"] = "no exit row"
            continue
        r.update(tradable=True, why="", expiry=ex, strike=K, fwd=F,
                 lot=float(chain["lot"].iloc[0]),
                 ce0=float(ok.loc[K, ("close", "CE")]), pe0=float(ok.loc[K, ("close", "PE")]),
                 ce1=float(nx.loc["CE", "close"]), pe1=float(nx.loc["PE", "close"]),
                 ce1_vol=float(nx.loc["CE", "contracts"]), pe1_vol=float(nx.loc["PE", "contracts"]))
    return pd.DataFrame(rows)


def trade(legs: pd.DataFrame, up: np.ndarray, strategy: str, slip_pct: float,
          margin_pct: float) -> pd.DataFrame:
    """Net P&L (Rs, 1 lot) of each tradable sample for direction calls `up` (bool array)."""
    t = legs[legs["tradable"]].copy()
    u = up[legs["tradable"].to_numpy()]
    q = t["lot"].to_numpy()
    if strategy == "buy":                     # UP -> long call, DOWN -> long put
        p0 = np.where(u, t["ce0"], t["pe0"]); p1 = np.where(u, t["ce1"], t["pe1"])
        b, s = fill(p0, True, slip_pct), fill(p1, False, slip_pct)
        gross = (p1 - p0) * q
        net_px = (s - b) * q
        capital = b * q
    else:                                     # UP -> short put, DOWN -> short call
        p0 = np.where(u, t["pe0"], t["ce0"]); p1 = np.where(u, t["pe1"], t["ce1"])
        s, b = fill(p0, False, slip_pct), fill(p1, True, slip_pct)
        gross = (p0 - p1) * q
        net_px = (s - b) * q
        capital = margin_pct * t["fwd"].to_numpy() * q
    cost = charges(b * q, s * q, t["date"])
    t["gross"], t["cost"] = gross, cost + (gross - net_px)    # slippage counted as a cost
    t["net"] = net_px - cost
    t["capital"] = capital
    return t


def summarise(t: pd.DataFrame, y: np.ndarray, up: np.ndarray) -> dict:
    daily = t.groupby("date")["net"].sum()
    cum = daily.cumsum()
    dd = float((cum - cum.cummax()).min()) if len(cum) else 0.0
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    cap = t.groupby("date")["capital"].sum().mean()
    hit = (up == (y == 1))
    return {
        "n_trades": int(len(t)), "n_days": int(daily.size),
        "dir_acc_traded": round(float(hit.mean() * 100), 2) if len(hit) else None,
        "win_rate": round(float((t["net"] > 0).mean() * 100), 2),
        "gross_pnl": round(float(t["gross"].sum())),
        "costs": round(float(t["cost"].sum())),
        "net_pnl": round(float(t["net"].sum())),
        "net_per_trade": round(float(t["net"].mean()), 1),
        "avg_daily_capital": round(float(cap)),
        # net P&L per rupee committed per trade (premium for buys, margin for sells)
        "ret_per_trade_pct": round(float(t["net"].sum() / t["capital"].sum() * 100), 3),
        "sharpe_ann": round(sharpe, 2),
        "max_drawdown": round(dd),
    }


# ---------------------------------------------------------------- driver
def test_samples(cfg: dict) -> pd.DataFrame:
    prices = pd.read_csv(cfg["data"]["price_csv"], parse_dates=["date"])
    te = chronological_split(prices, cfg)[2]
    ds = StockWindowDataset(te, {}, cfg["model"]["seq_len"], 1)
    return pd.DataFrame({"symbol": ds.syms, "date": pd.to_datetime(ds.dates), "y": ds.y}), prices


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/india.yaml")
    ap.add_argument("--tag", required=True, help="uses artifacts/seed_<tag>_*.npz")
    ap.add_argument("--options", default="data/interim/options.csv.gz")
    ap.add_argument("--slip-pct", type=float, default=0.005)
    ap.add_argument("--margin-pct", type=float, default=0.20)
    ap.add_argument("--min-contracts", type=int, default=50)
    ap.add_argument("--n-random", type=int, default=200)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    out = cfg["train"]["out_dir"]

    seeds = sorted(glob.glob(f"{out}/seed_{a.tag}_*.npz"))
    if not seeds:
        raise SystemExit(f"no seed caches for tag {a.tag}")
    prob = np.mean([np.load(p)["prob"] for p in seeds], axis=0)
    samples, prices = test_samples(cfg)
    assert len(samples) == len(prob), (len(samples), len(prob))
    assert np.array_equal(samples["y"].to_numpy(), np.load(seeds[0])["y"]), "sample order drift"
    samples["prob"] = prob

    legs_path = f"{out}/backtest_legs_{a.min_contracts}.csv.gz"
    if os.path.exists(legs_path):
        legs = pd.read_csv(legs_path, parse_dates=["date", "expiry"])
    else:
        cal_days = sorted(prices["date"].unique())
        cal = {pd.Timestamp(d): pd.Timestamp(n) for d, n in zip(cal_days[:-1], cal_days[1:])}
        lo = samples["date"].min()
        opts = pd.read_csv(a.options, parse_dates=["date", "expiry"])
        opts = opts[opts["date"] >= lo]
        legs = build_legs(samples, opts, cal, a.min_contracts)
        legs.to_csv(legs_path, index=False)
    assert len(legs) == len(samples)
    m = legs["tradable"].to_numpy()
    print(f"[{a.tag}] {len(seeds)} seeds | {len(samples):,} test predictions, "
          f"{m.sum():,} tradable ({m.mean():.1%})")
    print("  untradable:", legs.loc[~m, "why"].value_counts().to_dict())

    y = samples["y"].to_numpy()
    model_up = samples["prob"].to_numpy() > 0.5
    rng = np.random.default_rng(0)
    conf = np.abs(samples["prob"].to_numpy() - 0.5)
    policies = {
        "model": model_up,
        "always_up": np.ones(len(y), bool),
        "always_down": np.zeros(len(y), bool),
        "oracle": y == 1,
    }
    res = {"tag": a.tag, "n_seeds": len(seeds), "slip_pct": a.slip_pct,
           "margin_pct": a.margin_pct, "min_contracts": a.min_contracts,
           "n_predictions": int(len(y)), "n_tradable": int(m.sum()),
           "model_up_frac": round(float(model_up.mean()), 3), "strategies": {}}
    for strat in ("buy", "sell"):
        R = {}
        for name, up in policies.items():
            R[name] = summarise(trade(legs, up, strat, a.slip_pct, a.margin_pct), y[m], up[m])
        # random direction: distribution of net P&L, so the model can be placed in it
        rand = [trade(legs, rng.random(len(y)) < 0.5, strat, a.slip_pct, a.margin_pct)["net"].sum()
                for _ in range(a.n_random)]
        R["random"] = {"net_pnl_mean": round(float(np.mean(rand))),
                       "net_pnl_p05": round(float(np.percentile(rand, 5))),
                       "net_pnl_p95": round(float(np.percentile(rand, 95))),
                       "model_percentile": round(float((np.array(rand) < R["model"]["net_pnl"]).mean() * 100), 1)}
        # trade only the most confident calls (exploratory: cut-offs chosen on test, not val)
        for q in (0.5, 0.8, 0.9):
            cut = np.quantile(conf, q)
            sel = legs.copy()
            sel["tradable"] = sel["tradable"] & (conf >= cut)
            k = sel["tradable"].to_numpy()
            R[f"model_top{int((1-q)*100)}pct"] = summarise(
                trade(sel, model_up, strat, a.slip_pct, a.margin_pct), y[k], model_up[k])
        res["strategies"][strat] = R

    path = f"{out}/backtest_{a.tag}.json"
    json.dump(res, open(path, "w"), indent=2, default=str)
    hdr = f"{'policy':<16}{'trades':>7}{'dir%':>7}{'win%':>7}{'gross Rs':>12}{'costs Rs':>12}" \
          f"{'net Rs':>12}{'ret/tr%':>9}{'Sharpe':>8}{'maxDD Rs':>12}"
    for strat, R in res["strategies"].items():
        print(f"\n== Strategy {'A: BUY ATM call/put' if strat == 'buy' else 'B: SELL ATM put/call'} ==")
        print(hdr)
        for name, s in R.items():
            if name == "random":
                continue
            print(f"{name:<16}{s['n_trades']:>7}{s['dir_acc_traded']:>7}{s['win_rate']:>7}"
                  f"{s['gross_pnl']:>12,}{s['costs']:>12,}{s['net_pnl']:>12,}"
                  f"{s['ret_per_trade_pct']:>9}{s['sharpe_ann']:>8}{s['max_drawdown']:>12,}")
        r = R["random"]
        print(f"random direction: net mean {r['net_pnl_mean']:,} (5-95%: {r['net_pnl_p05']:,} .. "
              f"{r['net_pnl_p95']:,}); model sits at the {r['model_percentile']}th percentile")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
