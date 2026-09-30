# MISSION (verbatim brief from the user — this is the spec)

MISSION: Find a trading strategy with a real, historically supported edge, then stop and report to me. Do not build the live bot yet. We'll do that together after I review your findings.

WHO I AM
- US-based student with a small account (assume $2,000–$10,000; make it a config value). My broker is Robinhood. I also have access to Kalshi and US-legal prediction markets.
- Only consider markets and products I can legally trade from the US. Account for the pattern day trader rule (under $25k), options approval levels, and taxes on short-term gains.
- My risk tolerance: I can stomach a 25% drawdown. I can't stomach a strategy that can blow up the account in one bad day.

HARD RULES
- Research only. Do not write order-placement code, do not ask for broker credentials, and do not place any trades.
- Never fabricate data or results. If you can't get data for an idea, say so and move on.
- "No strategy passed" is an acceptable, valuable answer. Do not lower the bar to produce a winner.
- Keep RESEARCH_LOG.md and a trials ledger (trials.csv) updated as you go, so that if this session ends I can say "continue the mission" and you pick up where you left off.

PHASE 1: BUILD THE TEST LAB (before testing any idea)
- A backtesting framework with realistic costs: commissions, bid/ask spread, slippage, borrow costs for shorts, option fills at the bid when selling and the ask when buying (never mid), exchange and prediction-market fees.
- Data: use free sources first (Stooq/yfinance for daily equities and ETFs, FRED for macro, Coinbase/Binance public historical candles for crypto, Kalshi/Polymarket public history for prediction markets, NOAA for weather). If an idea needs data I don't have (e.g. historical options quotes or tick data), log it as "needs data: <source, cost>" and move on. I have a Massive (formerly Polygon) account and can provide an API key if you ask.
- Guard against the classic mistakes: look-ahead bias (only use data available at decision time), survivorship bias (use a universe that includes delisted stocks, or say clearly that you couldn't), and data errors (spot-check a sample of prices).
- Split the history into three parts and label them in every result:
  1. Development (oldest data): design and tune strategies here.
  2. Validation (middle): test tuned strategies here. Walk-forward where it makes sense.
  3. Final holdout (the most recent ~2–3 years): touch it ONCE per finalist, at the very end. Never tune on it.

PHASE 2: GENERATE AND TEST IDEAS
Test widely across three buckets:
A. Known strategies from academic research and practitioners, for example: time-series trend following, cross-sectional momentum, short-term reversal, overnight vs. intraday returns, turn-of-month and other calendar effects, post-earnings announcement drift, sector/dual momentum, pairs trading, the volatility risk premium (cash-secured puts, covered calls, iron condors, VIX term-structure strategies), 0DTE options, crypto funding-rate/basis carry, crypto weekend and time-of-day effects, prediction-market longshot bias, Kalshi weather markets vs. NOAA forecasts, and short-term latency arbitrage (confirm quickly whether it's dead after fees for a retail setup).
B. Variants and combinations of the above.
C. Your own new ideas. For each, write the economic reason it should work (who is on the other side of the trade and why they'd keep losing) BEFORE testing. Ideas with no plausible reason are data mining; mark them as such.
Cover the whole range: many small bets with occasional big wins, near-certain small wins, and everything in between. Holding periods from seconds to months, but only where you have data at that resolution.
Record EVERY variant you test in trials.csv, including the failures. The total count matters (see Phase 3).

PHASE 3: SEPARATE REAL EDGES FROM LUCK
A finalist must pass ALL of these, after costs:
- Profitable in development, validation, and the final holdout.
- Beats simply buying and holding SPY on a risk-adjusted basis (Sharpe), and report whether it beats it in absolute return too.
- Corrected for how many strategies you tried: compute the Deflated Sharpe Ratio using the total trial count from trials.csv, and require a probability of skill ≥ 95%.
- At least 100 trades out-of-sample (or explain why fewer is still meaningful).
- Robust: still works when parameters are nudged ±20%, on related assets, and in each regime (2008, 2020 crash, 2022 bear, etc.). Report yearly returns.
- Tail risk: report max drawdown, worst day, worst month, and a stress test for gap events (e.g. Feb 2018 volatility spike, March 2020). Any strategy that could lose more than 25% in a single event fails, however high its win rate.
- Monte Carlo: resample trades 10,000 times; report the 5th-percentile outcome.
- Works at my account size (PDT rule, minimum contract sizes, liquidity) and survives a 2× increase in assumed costs.

PHASE 4: REPORT AND STOP
When at least one strategy passes Phase 3, or when you've exhausted what you can test with the data available:
- Write STRATEGY_REPORT.md: the winning strategy (or the top 3 near-misses if none passed), the economic reason it works, exact rules, all Phase 3 metrics, equity curves (saved as PNGs), what could make it stop working, and how much I could realistically expect to make per year at my account size.
- Include a table of every strategy family tested and why each passed or failed.
- Send me a heads-up: POST a short summary to https://ntfy.sh/<MY_TOPIC> (I'll install the ntfy app and subscribe to that topic). Then STOP and wait for me.

WORKING STYLE
- Start with Phase 1 and show me the lab working on one simple strategy (e.g. SPY 200-day trend following) before scaling up.
- After that, work autonomously. Prioritize ideas by: likely edge × data availability × how practical it is at my account size.
- Be skeptical of your own results. If something looks too good (Sharpe above 3, win rate above 90% with no big losses), assume a bug or bias and hunt for it before believing it.

---
Notes added by Claude (not part of the user's brief):
- `<MY_TOPIC>` for ntfy has NOT been provided yet; ask for it before Phase 4. ntfy.sh is currently blocked by the container's network policy (see RESEARCH_LOG.md).
- Massive key: never ask the user to paste it into chat; it must be set as an environment secret (env var `MASSIVE_API_KEY`) in the cloud environment settings.
