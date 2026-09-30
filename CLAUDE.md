# Instructions for any Claude session in this repo

This repo is a **research-only** trading-strategy search. Before doing anything:

1. Read `MISSION.md` (the user's brief, verbatim — it is the spec and the hard rules).
2. Read `RESEARCH_LOG.md` — top section "STATUS / NEXT STEPS" says exactly where the last session stopped.
3. Read `trials.csv` (every variant ever tested; its row count feeds the Deflated Sharpe Ratio) and `holdout_ledger.csv` (which finalists already used their one holdout look).

When the user says **"continue the mission"**: do steps 1–3, then resume at NEXT STEPS.

Non-negotiables (details in MISSION.md): no order-placement code, no broker credentials, no trades
(never call any `mcp__robinhood__place_*` / cancel / exercise tool); never fabricate data or results; never tune on the
final holdout; log every variant to `trials.csv` (use `lab.trials.log_trial`); update `RESEARCH_LOG.md` as you go;
"no strategy passed" is an acceptable answer.

Run tests: `python -m pytest -q`.
