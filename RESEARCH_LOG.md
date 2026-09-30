# RESEARCH LOG

Spec and hard rules: `MISSION.md` (verbatim user brief). Ledgers: `trials.csv` (every variant tested; row count feeds the Deflated
Sharpe Ratio), `holdout_ledger.csv` (one holdout look per finalist; empty = nothing has touched the holdout).
Tests: `python -m pytest -q` (38 pass). Network probe: `python scripts/check_network.py`. Data validation: `python scripts/validate_data.py` -> reports/data_validation.md.

---
## 0. STATUS / NEXT STEPS  (read this first)

**Last updated:** 2026-09-30 (session 1, continuing). **Phase 1 complete. Phase 2 IN PROGRESS.** The user opened network access mid-session (Yahoo, FRED(flaky), Coinbase,
Binance, Kalshi, Polymarket, NOAA, IEM, CBOE, ntfy all reachable; Stooq resets connections -> dropped; SEC returns 403 without a contact User-Agent; Massive API needs a key,
which is NOT needed). No trades, no order code, no broker tools. Holdout untouched (`holdout_ledger.csv` empty).

### Where Phase 2 stands
- **Batch 1 (equities/ETFs/crypto, 132 variants logged, `python -m research.families`, results in trials.csv, ranking via `python scripts/leaderboard.py`)**:
  families f01 SPY trend + related assets, f02 multi-asset trend / sector & asset-class momentum, f03 calendar (turn-of-month, day-of-week, overnight/intraday),
  f04 reversal (RSI2, IBS), f05 vol-managed, f06 short-vol (SVXY, capped), f07 crypto trend/weekend, f08 VIX-contango filter, f09 crypto trend x vol-target, f10 inverse-vol multi-asset trend.
  **0 of 132 reach DSR >= 0.95.** Deflation hurdle at N=132 is an annualised Sharpe of ~1.0. Best DSRs: crypto trend (0.84, but max DD 53-82% at full size), then
  inverse-vol multi-asset trend (Sharpe 0.81 vs SPY 0.51, max DD -9%, but CAGR ~7.5% < SPY) and the VIX-contango filter on SPY (Sharpe 0.6-0.67 vs 0.51).
- **Killed by costs / no edge:** overnight & intraday SPY/QQQ (daily round trip ~4bp x 252 = ~10%/yr), day-of-week, BTC weekend/weekday, IBS/RSI2 fade in validation (edge decayed: dev 0.8 -> val 0.3).
- **Batch 2 (in progress): Kalshi weather markets vs NBM forecasts** (design pre-registered in section 9). Data: `python scripts/fetch_kalshi_weather.py` (candles) +
  `python scripts/fetch_iem_weather.py` (forecasts); analysis module `research/weather.py`. NOT yet analysed: no price-vs-outcome result has been viewed.
- Generic Phase-3 evaluator: `research/phase3.py` (use only for candidates that clear DSR or as near-miss reporting; never loads the holdout).

### Blockers / asks for the user
1. **ntfy topic** (`<MY_TOPIC>`) for the Phase-4 heads-up (ntfy.sh is reachable now).
2. Confirm defaults in `lab/config.py` (account $5,000, tax 22%).
3. Massive/Polygon: only needed if an options idea proves worth buying quotes for (plan currently entitles ~recent equity data only; do not upgrade yet).

### Next steps
1. When weather downloads finish (`data/raw/kalshi/*_candles.parquet` for KXHIGHNY, CHI, MIA, AUS, LAX, DEN, PHIL and `data/raw/iem/NBS_*.parquet`): verify Chicago station (KMDW vs KORD) by
   forecast error only, run the pre-registered grid in `research/weather.py` (NOT yet written as a runner: build_city / add_rolling_bias / fit_sigma / p_yes / simulate exist), log every variant.
2. Model-free favourite-longshot rules on the same data. 3. If any variant clears DSR >= 0.95 -> Phase 3 (`research/phase3.py`) + one holdout look. 4. Otherwise near-miss report.
5. STRATEGY_REPORT.md, ntfy, STOP.

---
## 1. Pre-registration (frozen 2026-09-30, BEFORE any research backtest)

**Splits (`lab/config.py`).** Equities/ETFs/macro: Development <= 2013-12-31 (contains 2000-02 bear, 2008); Validation 2014-01-01..2023-12-31 (2015-16, 2018,
2020, 2022); FINAL HOLDOUT 2024-01-01..latest (~2.75y; Aug-2024 and Apr-2025 shocks). Crypto: dev <= 2019-12-31, val 2020-2023, holdout 2024+.
Prediction markets: dev <= 2022, val 2023, holdout 2024+. `run_backtest` raises `HoldoutViolation` for any panel extending past the validation end unless it
came from `open_holdout()`, which logs to `holdout_ledger.csv` and refuses a second look for the same `strategy_id`. Research code uses `research_panel()`.
Every result carries a split label. Regime reporting (2008, 2020, 2022, ...) is descriptive across the whole window, not tuned on.

**Timing / costs.** Signals use data <= t; default execution is the next bar's open (`next_open`); `next_close` (one more bar of delay) is used when opens are
unreliable; `same_close` is only for low-frequency signals. All strategies + the benchmark are scored on ONE common window that starts after the longest
look-back (no warm-up mismatch). Costs: half-spread + slippage + fees on actual traded notional after weight drift, sell-side regulatory fees, borrow on
shorts per calendar day, margin interest on gross long > 1, cash earns rf minus 25bp. Options: buy at ask / sell at bid, never mid, per-contract pass-through
fees. Prediction markets: buy at ask, fee formulas in `lab/prediction.py`. Every "2x cost" test uses `CostModel.scaled(2)`.

**Trial counting.** Every distinct strategy variant evaluated = 1 trial (`trial_id` = hash of family, variant, params, universe), including failures,
+-20% robustness nudges, and execution-timing variants. Re-running with 2x costs is the same trial (extra row). `needs_data` rows are not tested and do not count.

**Operational pass/fail definitions (Phase 3; all after costs, all must pass):**
- *Profitable in dev, val, holdout*: net CAGR > 0 in each split (holdout evaluated once).
- *Beats SPY buy&hold*: Sharpe (excess of rf) higher over the full common window, reported per split; also report whether it beats on absolute return.
- *DSR*: computed on the selection sample (dev+val) with N = all tested trials in `trials.csv` and the cross-trial variance of per-period Sharpe; require DSR >= 0.95.
  The holdout is then a one-shot test (PSR vs 0 and vs SPY), not part of selection.
- *Trades*: >= 100 out-of-sample (val+holdout) trades, or an explicit argument (e.g. block-bootstrap of daily returns) for why fewer is meaningful.
- *Robust*: +-20% on every parameter still profitable with DSR-comparable Sharpe; works on related assets; yearly table + named regimes shown.
- *Tail*: report max DD, worst day, worst month, rolling worst 1/5/21/63 bars, named stress windows, plus gap-shock tests. FAIL if any single event
  (named window, or a hypothetical gap for products with jump risk) could lose > 25% of the account.
- *Monte Carlo*: 10,000 trade resamples (and block bootstrap of returns); report 5th percentile final equity and worst-5% drawdown.
- *Account size*: tested at $2k/$5k/$10k: whole/fractional shares, minimum contract sizes and capital, PDT mode (both "none" and "legacy"), liquidity; survives 2x costs
  (still profitable in all splits with DSR >= 0.95).
- Skepticism rule: Sharpe > 3 or win rate > 90% with no big losses => assume a bug/bias and hunt for it before believing.

---
## 2. External facts (rules, fees). "2nd-hand" = from search summaries; primary pages were blocked. All are config values, not hard-coded assumptions.

| Fact | Value | Status |
|---|---|---|
| Pattern day trader rule | SEC approved FINRA Rule 4210 amendments 2026-04-14 eliminating the $25k minimum and PDT designation; effective 2026-06-04 (brokers may phase in to 2027-10-20). Robinhood reportedly rolled out day-one, removed PDT flags; $2,000 margin minimum still applies | 2nd-hand, multiple sources agree; verify in Robinhood app. `pdt_mode="none"` default, `"legacy"` (3 day trades / 5 days under $25k) kept as conservative check |
| Robinhood commissions | $0 stocks/ETFs/options; pass-through regulatory fees remain (options TAF $0.00329/contract on sells as of 2026-01-01; OCC/ORF amounts unconfirmed -> model $0.04/contract/side placeholder) | 2nd-hand; 2x-cost test covers it |
| Robinhood shorting | Reportedly allowed for eligible margin accounts with >= $2,000 | 2nd-hand |
| Robinhood options levels | L1 covered calls + cash-secured puts; L2 long calls/puts; L3 spreads, iron condors/butterflies. CSP needs cash = strike x 100 | 2nd-hand |
| Kalshi fees | taker = ceil_cent(0.07 x C x P x (1-P)); maker = ceil_cent(0.0175 x C x P x (1-P)); peaks 1.75c/contract at 50c; some series have multipliers | 2nd-hand, several sources agree; verify vs kalshi.com/docs/kalshi-fee-schedule.pdf |
| Polymarket US fees | one source: taker coeff 0.0695, maker rebate coeff -0.0125 (2026-09); older filing: 0.30% taker / 0.20% rebate | UNVERIFIED, conflicting; provisional |
| Crypto retail fees | Coinbase Advanced entry tier ~0.4/0.6% maker/taker (one source: 0.5/0.9% from 2026-09); Robinhood ~0.35-0.95% all-in | 2nd-hand; means only LOW-turnover crypto strategies can survive |
| SEC / FINRA TAF equity fees | ~$27.80 per $1M sold; $0.000166/share | rates unverified; economically negligible |
| Tax | short-term gains at ordinary rates; wash-sale rule; Section 1256 (60/40) applies to broad-index options (SPX/XSP), not SPY | standard; `lab/taxes.py` models yearly short-term tax only |

---
## 3. Lab inventory (all in `lab/`, all unit-tested with hand-computed answers; synthetic data is used ONLY in tests)

`config.py` splits+costs+account · `panel.py` aligned OHLCV + caveats travelling with every result · `engine.py` vectorised target-weight backtester
(next_open / next_close / same_close, drift-aware turnover, costs, borrow, financing, cash yield, trade extraction that reconciles to total P&L) ·
`lookahead.py` truncation test that catches shift(-1), centered windows, full-sample normalisation · `splits.py` holdout vault · `metrics.py`
Sharpe/Sortino/DD/worst day-month/yearly/stress windows/benchmark comparison (Jobson-Korkie-Memmel) · `stats.py` PSR, Deflated Sharpe, Sharpe-diff test,
trade-resample and block-bootstrap Monte Carlo · `trials.py` append-only ledger · `options.py` bid/ask fills, multi-leg payoffs, max loss, capital ·
`prediction.py` Kalshi/Polymarket fees, binary fills/settlement, Kelly · `constraints.py` fractional/whole shares, PDT, option capital · `taxes.py` ·
`report.py` plots · `data/{http,cache,loaders,quality}.py` loaders (UNTESTED LIVE; parsers tested on format fixtures), provenance manifest, data-error
and cross-source spot checks.

**Not built yet (needs live data/APIs):** Kalshi/Polymarket/NOAA loaders; option-chain replay; intraday engine (needed for seconds-minutes horizons and
true day-trade counting); survivorship-free stock universe (`lab/panel.py` caveats propagate; free stock data will be flagged survivorship-biased).

**Sanity evidence that the lab is not fooling us:** best-of-200 pure-noise strategies looks like Sharpe > 1 and passes a naive PSR test but fails the DSR
(test `test_noise_strategies_do_not_survive_deflation`); a one-bar-peeking strategy scores Sharpe > 5 without the detector and is rejected with it.

---
## 4. Data inventory

| Dataset | Source | Coverage | Status |
|---|---|---|---|
| S&P 500 index daily OHLCV | PyPI package `arch` (bundled Yahoo ^GSPC) | 1999-01-04..2018-12-31, price-only | REAL. Closes match 6 widely-published values exactly (676.53 on 2009-03-09; 2351.10 on 2018-12-24; 2506.85 on 2018-12-31; 1106.42 on 2008-09-29; 899.22 on 2008-10-10; 752.44 on 2008-11-20 — recalled from memory, so a sanity anchor, not an independent source). **Opens equal the prior close on 95-98% of days 1999-2005 (40% overall) => unusable; execution at closes.** |
| Nasdaq Composite daily | `arch` | 1999-2018 | REAL, unchecked (same open artifact expected) |
| VIX daily | `arch` | 2014-2018 only | REAL, too short |
| Ken French Mkt-RF/SMB/HML/RF monthly | `arch` | 1926-07..2018-11 | REAL (first row matches the published 1926-07 values). RF ends 2018-11, so the demo ends 2018-11-30 |
| SPY/ETFs/macro/crypto/prediction/weather | Stooq/Yahoo/FRED/Coinbase/Binance/Kalshi/Polymarket/NOAA | - | **BLOCKED (network policy)** |
| Recent equity bars (last ~2y) | Massive MCP | 2026-09 works; 2024-09 and earlier NOT entitled; rate-limited | usable only for spot checks |

---
## 5. Phase-1 demo (machinery check — NOT a finding). `python scripts/demo_trend.py` -> `reports/demo_sma200_sp500.png`

200-day SMA on the S&P 500 (price-only), long/cash, cash earns T-bill - 25bp, SPY-like costs, next_close execution, scored 1999-12-15..2018-11-30 after the 240-bar warm-up:

| Split | Strategy CAGR / Sharpe / MaxDD | Buy&hold CAGR / Sharpe / MaxDD |
|---|---|---|
| DEVELOPMENT (..2013) | 4.94% / 0.32 / -21.6% | 1.98% / 0.10 / -56.8% |
| VALIDATION (2014-18) | 3.80% / 0.35 / -18.1% | 8.50% / 0.66 / -14.2% |
| DEV+VAL | 4.64% / 0.33 / -21.6% | 3.64% / 0.20 / -56.8% |

Sharpe edge over buy&hold on dev+val: +0.13 but one-sided p = 0.28 (not significant). 72 trades. 2x costs: 4.47% / 0.31. +-20% (SMA160/240): 4.15% / 0.29 and 3.81% / 0.25.
Yearly: sidestepped 2001-02 and 2008 (+1.3% vs -38.5%), lost ground in every bull year (2009-10, 2015: -12.4% vs -0.7%). Deflated Sharpe with N=4: 0.885 (< 0.95).
MC (10k): block-bootstrap final-equity p05 1.17x, worst-5% max DD -39%, P(maxDD worse than 25%) 47%; trade-resample p05 0.99x (only 72 heterogeneous trades).
Reading: classic trend-following behaviour (lower vol and drawdown, similar-or-slightly-better return, worse in bull markets); NOT clear evidence of an
after-deflation edge on price-only data. Note the validation window is a bull run in which buy&hold wins outright: "beats SPY in every split" is a hard bar for any de-risking strategy.
The logged variants: sma200 (+2x costs row), sma200 same_close, sma160, sma240 (N = 4).

---
## 6. Idea backlog, prioritised by (likely edge x data availability x practicality at $2-10k). Economic rationale is written BEFORE testing.

*A = literature/practitioner, B = variants, C = own ideas. "DM" = data-mining (no rationale).* Ideas that cannot be tested with obtainable data are logged in trials.csv as `needs_data`.

1. **[A] Time-series trend on liquid ETFs (SPY, QQQ, IWM, EFA, EEM, TLT, IEF, GLD, DBC).** Why: slow diffusion of information and forced/behavioural flows create autocorrelation in macro-driven assets; the counterparty is the price-insensitive or late-reacting holder. Expect drawdown reduction more than extra return. Data: free. Practical: fractional ETF shares, ~$0 cost, monthly-ish turnover. Trade-count concern (<100).
2. **[A/B] Dual/sector/asset-class momentum (Antonacci GEM, sector rotation, 12-1 month).** Why: same as above cross-sectionally + institutional herding; free ETF data avoids single-stock survivorship. Practical at $2k.
3. **[A] Turn-of-month, overnight-vs-intraday, day-of-week/holiday effects on SPY.** Why: pension/fund flow timing; overnight premium attributed to intraday liquidity provision. Expect death after costs for daily round trips (~250 round trips/yr x ~2-4bp) — test, but likely fails at retail costs. Needs reliable ETF opens (check quality!).
4. **[A/C] Short-term reversal on ETFs/sectors; volatility-managed equity (de-lever when realised vol high).** Why: liquidity-provision premium; vol clustering with weakly compensated risk. Data free.
5. **[A] VIX term-structure / short-vol ETPs (SVXY etc.).** Why: insurance premium paid by hedgers. RISK: Feb-2018 one-day collapse => likely fails the 25% single-event rule; test unlevered/capped variants only.
6. **[A] Crypto trend and weekend/time-of-day effects (BTC/ETH spot).** Why: retail-driven trends, thin weekend liquidity. Retail fees ~0.5%/side kill anything with turnover; only monthly-ish trend is viable. Data: Coinbase/Binance public candles.
7. **[A] Prediction-market favourite-longshot bias (Kalshi/Polymarket US).** Why: retail overpays for low-probability outcomes (lottery preference); counterparty = longshot buyers. Kalshi fee peaks at 50c and shrinks at extremes, so extreme-price trades are cheap. Data: public histories (blocked now). Practical at $2k: yes, but liquidity per market is thin.
8. **[A] Kalshi weather vs NOAA/NBM forecasts.** Why: forecast skill differs from market-implied probability when retail anchors on headline temps. Needs archived point-in-time forecasts. Practical.
9. **[A] Options VRP (defined-risk credit spreads / iron condors; 0DTE), PEAD, single-stock momentum/reversal, funding-rate carry, latency arb** -> logged `needs_data` (see trials.csv). Structural notes: SPY-sized cash-secured puts need ~$59k; latency arb almost certainly dead for retail after fees; perps not US-accessible.
10. **[C] Regime combinations** (trend x vol filter, momentum x trend filter) — only after A-buckets produce something to combine; combinations multiply trials, so each must have a stated economic reason.

---
## 7. Decisions & discoveries log
- 2026-09-30 Network policy blocks all data hosts (403 CONNECT) -> Phase-2 blocked. Did not try to circumvent (proxy README forbids it); WebSearch (runs outside the container) used for rules/fees only.
- 2026-09-30 Massive plan probe: SPY daily 2026-09 OK; 2003, 2008, 2016, 2020, 2022, 2024-09 all `NOT_ENTITLED`; rapid calls -> `RATE_LIMIT`. Not usable for research history. Data returned through the MCP would also land in chat context rather than on disk, which makes bulk use impractical anyway.
- 2026-09-30 PDT rule was abolished effective 2026-06-04 (see facts). The brief's "under $25k" premise is out of date; legacy mode kept as a stress check.
- 2026-09-30 Found and fixed an engine bug via a known-answer test: open-to-close bars must compound overnight and intraday legs ((1+ov)(1+id)-1), not add them (was overstating buy&hold by ~0.1% over 300 bars).
- 2026-09-30 Found the S&P open-price artifact via `quality.check_ohlcv` (see data inventory) before it could contaminate a next-open backtest.
- 2026-09-30 Demo warm-up bug caught in review: strategy sat in cash during the SMA warm-up while buy&hold was invested. Fixed by scoring every variant + benchmark on one common window.
- 2026-09-30 Untested-live loaders are labelled as such in code and here; Kalshi/Polymarket/NOAA loaders deliberately deferred until real responses can be inspected.

## 8. Session log
- S1 (2026-09-30): brief received (message arrived truncated once, then in full); built and tested the lab; ran the Phase-1 demo; probed data access; wrote this log. Stopped at the Phase-1 checkpoint as the brief requests ("show me the lab working ... before scaling up") and because Phase 2 needs network access only the user can grant.

---
## 9. PRE-REGISTRATION: Kalshi weather markets vs NOAA/NBM forecasts (written 2026-09-30 BEFORE any Kalshi/IEM data was pulled)

**Split revision (disclosed):** prediction-market splits changed from dev<=2022 / val 2023 / holdout 2024+ to dev <= 2023-03-31, val 2023-04-01..2024-09-30,
HOLDOUT 2024-10-01..latest (~2.0y). Made before viewing any prediction-market outcome; equity/crypto splits unchanged.

**Why this is promising:** retail-heavy markets, a public model forecast with a published uncertainty (NBM `txn`/`xnd`, GFS MOS `n_x`), settlement on the NWS
CLI report, hourly bid/ask candles. Counterparty = bettors anchoring on headline temperatures / stale forecasts. **Why it may fail:** pros may already arbitrage it,
and fees peak at mid prices (1.75c/contract at 50c) vs edges of a few cents. Data verified reachable: Kalshi `/historical/markets` + `/markets` + batch candlesticks
(KXHIGHNY archive 2021-08-07..2026-07-30, plus live), IEM `cgi-bin/request/mos.py` bulk MOS (GFS `n_x`, NBS `txn`,`xnd`). API confirms `fee_type=quadratic`, multiplier 1.

**Design (fixed now):**
- Cities/series: highest-liquidity daily HIGH-temperature series (start: NYC KNYC, Chicago KMDW, Miami KMIA, Austin KAUS, LA KLAX, Denver KDEN, Philadelphia KPHL);
  station mapping verified by checking forecast-vs-settlement error size before use.
- Decision times (each a separate trial): DT1 = 20:00Z on D-1; DT2 = 10:00Z on D. Forecast = latest NBM run with runtime <= DT-3h (availability lag). Prices = the hourly
  candle ending at DT: BUY YES at yes_ask, BUY NO at 1 - yes_bid, plus 1c slippage; skip books with ask>=0.99 or bid<=0.01. Kalshi taker fee formula; hold to settlement.
- Model: outcome ~ Normal(mu = txn + city bias, sigma = c * xnd) with integer-degree continuity correction; city bias and scale c are fitted ONLY on dev.
- Rule: trade if edge = p - (price + fee/contract) >= theta, theta in {0.03, 0.05, 0.08}; fixed 1% of bankroll per trade capped by 10% of prior-hour volume; max trades/day capped.
- Also: favourite-longshot calibration by price bucket at DT (no model), rules "sell longshots" (buy NO when yes_ask<=0.10 / 0.05).
- Every (DT x theta x city-set) is a trial; 2x cost (slippage 2c, fee x2) and +-20% theta nudges logged. Holdout touched once per finalist.

**Addendum (2026-09-30, before any weather analysis): city bias.** Miami's Kalshi markets start 2023-05, Austin/others may too, so per-city bias cannot always be fitted on dev.
Replaced by a strictly point-in-time rolling bias: for event D and decision time DT, bias = mean(actual - txn) over the previous 60 events of that city that had SETTLED
before DT (DT1: events <= D-2; DT2: events <= D-1), requiring >= 20; events lacking that history are excluded from ALL evaluation (burn-in). The global sigma parameters
(a, c) are still fitted on dev events only. Station mapping (e.g. Chicago KMDW vs KORD) is decided from forecast-vs-settlement error size on dev events, never from P&L.
Data note: archived Kalshi candles use field names `close`/`volume` (live: `close_dollars`/`volume_fp`); loader fixed and verified on 2021, 2023, 2025 and 2026 markets.

## 10. Data validation log (2026-09-30, real data)
- Yahoo ETFs/indices, Coinbase BTC/ETH cached (33 datasets, `data/manifest.json`); `reports/data_validation.md`: 0 structural errors (only ^IRX yield-index zeros, expected).
- SPY vs S&P price-return correlation 0.983 overall / 0.997 since 2015. Explained, not data errors: ex-dividend Fridays (SPY price drops by its dividend, index does not), 4:15pm ETF close vs 4:00pm index close before ~2010, stress-day close-auction differences.
- **Independent check:** last six Yahoo SPY closes (2026-09-22..29) equal Massive's to the cent. SPY total-return CAGR 10.8% vs 8.9% price-only (dividend gap ~1.9%/yr as expected).
- ETF opens are reliable (fraction equal to prior close <5% except thin BIL/SHY), so `next_open` execution is valid for the traded ETFs; the old S&P index opens are not.
- Stooq unusable (bot-check page/connection resets). FRED intermittently times out; `^IRX` (13-week bill yield) used for the risk-free rate.
