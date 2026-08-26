# Spec 01 — Price data

**Status: built and EXECUTED at full scale.** `src/data/prices.py`, `scripts/build_prices.py`.

## Contract

- **In:** `data/raw/universe50.csv` (columns `symbol`, `sector`; 50 names, sector-stratified — see spec 06), date range from config.
- **Out:** `data/interim/prices.csv` — long format, one row per stock-day:
  `symbol, date, close_raw, label`, plus the 17 z-scored feature columns
  (`FEATURE_COLS` in `src/data/prices.py` = 6 OHLCV-family + 11 technical indicators).
  `close_raw` is retained unscaled so labels are built from real prices, never z-scores.

## Requirements

1. **R1 — Corporate actions.** Fetch with `auto_adjust=True`. NSE bonus/split frequency is
   high enough that unadjusted series produce false return jumps indistinguishable from real
   moves.
2. **R2 — Calendar from data.** Derive the trading calendar from observed traded dates. Never
   hard-code a holiday list: the NSE calendar shifts annually and includes the Muhurat session.
3. **R3 — Gap policy.** Forward-fill a gap of ≤3 consecutive sessions (halts, data errors).
   Drop any symbol missing >5% of sessions over the full period — that signals delisting or
   illiquidity, not a data glitch.
4. **R4 — Circuit before sigma.** Clip returns to the ±20% exchange price band *first*, then
   apply the 3-sigma winsorisation with training-period statistics. A circuit-limit day is a
   genuine, informative event; treating it as an outlier destroys exactly the signal the text
   stream is supposed to explain.
5. **R5 — Seventeen channels (d_p = 17).** 6 OHLCV-family (`open, high, low, close, volume,
   turnover`) + 11 technical indicators (`ret1, rsi14, macd, macd_sig, bb_pctb, bb_width,
   atr14, sma20_gap, sma50_gap, obv, mom10`), each z-scored per symbol on **training-period**
   statistics only. This matches the paper's Table 4 "Price+TA+Text" row — its §4.2 describes
   closing prices only, but that is the *base* configuration, not the headline model.
6. **R6 — Labels.** `label = close.shift(-1) > close`, strictly greater. Flat days are 0.

## Acceptance criteria

- [x] Live download succeeds for `.NS` tickers (verified: RELIANCE, TCS — 240 rows, H1-2024).
- [x] No feature column contains NaN or ±inf after `featurize`.
- [x] Up-share lands in 0.50–0.53 (measured **0.513** over 2018–2025, 12 symbols).
- [x] Full 50-symbol, 2018–2025 build completes: **98,690 rows, 50 symbols, up-share 0.510**
      (50 × 1,974 = 98,700 — ten rows short, the tail days that have no t+1 close). The
      up-share matches the 0.513 measured on the earlier 12-name sample, so the sector
      stratification did not shift the class balance.
      (Universe fixed and coverage-checked: all 50 have ≥95% price coverage over 1,975 days.)

## Known issues

- **TATAMOTORS is a discontinuity.** The 2025 CV/PV demerger means the pre- and post-2025
  series are different entities. Currently kept; either exclude it or truncate at the
  demerger date before any published result.
- Delivery-percentage would be a natural 18th channel but requires the NSE bhavcopy, not
  yfinance. Deferred; bump `n_price_feats` if added.
- `turnover` stands in for the paper's unnamed 6th OHLCV column (it lists only five fields
  but counts six). Indian-market natural; flag as a deviation in any write-up.

## Resolved: universe selection

Universe composition is a spec-06 decision, not a spec-01 one, and it is now settled:
50 names, sector-stratified from the NIFTY 100, in `data/raw/universe50.csv`. Expanding
to NIFTY 500 costs nothing here — it is just a longer symbol list — but buys almost no
statistical power, because effective sample size saturates near 6,000 regardless of stock
count. See spec 06 for the measurement.
