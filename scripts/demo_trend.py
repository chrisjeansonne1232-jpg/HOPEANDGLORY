"""Phase 1 demonstration: the lab running ONE simple strategy end-to-end (200-day SMA trend following on the S&P 500).

DATA: real, but bundled in the PyPI package `arch` (network access to Stooq/Yahoo was blocked in the research container):
  * S&P 500 daily OHLCV 1999-01-04 .. 2018-12-31 (Yahoo ^GSPC history), used through 2018-11-30 (last month with a real RF
    value; nothing is filled in). PRICE-ONLY: no dividends. Closes spot-checked against six
    widely published values (exact match). Opens are NOT independent prints before ~2006 -> execution uses closes.
  * Ken French monthly RF (T-bill) 1926-2018, spread over trading days as the cash rate.
This dataset ends in 2018, so the FINAL HOLDOUT (2024+) is not part of it and remains untouched.
THIS IS A DEMONSTRATION OF THE MACHINERY, NOT RESEARCH EVIDENCE OR A FINDING. It IS logged to trials.csv (every variant counts).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import arch.data.frenchdata as fd
import arch.data.sp500 as sp

from lab import metrics as M
from lab import stats as S
from lab import trials as T
from lab.config import DEFAULT, EQUITY_SPLITS, REPORTS
from lab.engine import buy_and_hold, run_backtest
from lab.lookahead import check_lookahead
from lab.panel import Panel, daily_rf_from_monthly_percent
from lab.report import equity_plot, fmt_summary_table
from lab.splits import research_panel
from lab.taxes import tax_drag_table

FAMILY = "trend_following_sma"
UNIVERSE = "SP500_index_price_only"
SRC = "arch-bundled Yahoo ^GSPC 1999-2018 (price only) + Ken French RF"
CAVEATS = ["PRICE-ONLY index (no dividends): benchmark and strategy both understate total return",
           "opens unusable before ~2006 -> execution at closes (next_close)",
           "index not directly tradable: SPY-like costs assumed", "DEMO ONLY - not a finding; ends 2018 (holdout untouched)"]


def load_panel() -> tuple[Panel, pd.Series]:
    df = sp.load()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    fr = fd.load()
    v = np.asarray(fr.index.astype("int64"))
    rf_m = pd.Series(fr["RF"].to_numpy(), index=pd.PeriodIndex([f"{x // 100}-{x % 100:02d}" for x in v], freq="M"))
    # the sample ends at the last month for which a REAL risk-free rate exists (no filling)
    end = rf_m.index.max().to_timestamp(how="end").normalize()
    panel = Panel.from_frames({"SPX": df.loc[:end]}, source=SRC, caveats=CAVEATS)
    rf = daily_rf_from_monthly_percent(rf_m, panel.index)
    return panel, rf


def sma_strategy(n: int):
    def f(panel: Panel) -> pd.DataFrame:
        c = panel.adj_close
        return (c > c.rolling(n).mean()).astype(float)          # decided at the close of t using closes <= t
    return f


def evaluate(panel, rf, n, mode="next_close", cost=DEFAULT.cost, label=None):
    w = sma_strategy(n)(panel)
    return run_backtest(panel, w, cost=cost, rf_period=rf, exec_mode=mode, label=label or f"SMA{n} trend ({mode})")


MAX_LOOKBACK = 240   # longest SMA among all variants; every variant AND the benchmark are scored from the same first bar


def window(panel, split):
    return EQUITY_SPLITS.bounds(split, panel.index[MAX_LOOKBACK], panel.index[-1])


def log_variant(panel, res, bench_dev_val_sharpe, n, mode, cost_mult, rationale, notes=""):
    dv = res.sliced(*window(panel, "DEV+VAL"), "DEV+VAL")
    T.log_trial(family=FAMILY, variant=f"sma{n}_{mode}", params={"n": n, "mode": mode}, universe=UNIVERSE, data_source=SRC,
                data_range=f"{dv.returns.index[0].date()}..{dv.returns.index[-1].date()}", split="DEV+VAL", metrics=dv.summary(),
                n_trades=len(dv.trades), cost_mult=cost_mult, bench_sharpe=bench_dev_val_sharpe, rationale=rationale, notes=notes)


def main():
    panel, rf = load_panel()
    panel = research_panel(panel)                                   # never let research see the holdout
    print("DATA:", SRC, f"({panel.index[0].date()}..{panel.index[-1].date()}, {len(panel.index)} bars)")
    print("CAVEATS:", *CAVEATS, sep="\n  - ")

    # 1. look-ahead check BEFORE the first backtest
    assert check_lookahead(sma_strategy(200), panel)
    print("\nlook-ahead truncation test: PASSED (weights on date t are unchanged when data after t is removed)")

    # 2. main strategy + benchmark under identical costs/timing
    strat = evaluate(panel, rf, 200)
    bench = buy_and_hold(panel, "SPX", rf_period=rf, exec_mode="next_close", label="buy&hold S&P 500 (price only)")
    stress2 = evaluate(panel, rf, 200, cost=DEFAULT.cost.scaled(2.0), label="SMA200 trend @ 2x costs")
    same_close = evaluate(panel, rf, 200, mode="same_close", label="SMA200 trend (same_close: trade at signal-bar close)")

    rat = "economic: trend persistence from slow-moving capital/behavioural underreaction; equity drawdowns are avoided when trend turns negative"
    b_dv = bench.sliced(*window(panel, "DEV+VAL"), "DEV+VAL").summary()["sharpe"]
    log_variant(panel, strat, b_dv, 200, "next_close", 1.0, rat, "Phase-1 demo; replication of a widely known rule")
    log_variant(panel, stress2, b_dv, 200, "next_close", 2.0, rat, "2x cost stress")
    log_variant(panel, same_close, b_dv, 200, "same_close", 1.0, rat, "execution-timing variant (trade at the signal bar's close)")
    for n in (160, 240):                                            # +-20% parameter nudges are trials too
        log_variant(panel, evaluate(panel, rf, n), b_dv, n, "next_close", 1.0, rat, "+-20% robustness variant")

    # 3. report by split, always labelled
    print(f"\n=== RESULTS BY SPLIT (net of costs; cash earns T-bill minus 25bp; all rows scored from {window(panel, 'DEV+VAL')[0].date()}, after the {MAX_LOOKBACK}-bar warm-up) ===")
    rows = {}
    for split in ("DEVELOPMENT", "VALIDATION", "DEV+VAL"):
        a, b = window(panel, split)
        for nm, r in (("SMA200 trend", strat), ("buy&hold", bench)):
            rows[f"[{split}] {nm}"] = r.sliced(a, b, split).summary()
    print(fmt_summary_table(rows))
    print("\nHOLDOUT: not present in this dataset (ends 2018) -> untouched. Holdout ledger empty.")

    print("\n=== STRATEGY vs BUY&HOLD (dev+val, same timing/costs) ===")
    a, b = window(panel, "DEV+VAL")
    sv, bv = strat.sliced(a, b, "DEV+VAL"), bench.sliced(a, b, "DEV+VAL")
    cmp = M.compare(sv.returns, bv.returns, sv.rf, panel.ppy)
    for k, v in cmp.items():
        print(f"  {k:<26} {v:.4f}" if isinstance(v, float) else f"  {k:<26} {v}")

    print("\n=== ROBUSTNESS ===")
    print(" 2x costs        :", stress2.sliced(a, b, "DEV+VAL").describe())
    print(" same_close exec :", same_close.sliced(a, b, "DEV+VAL").describe())
    for n in (160, 240):
        print(f" SMA{n} (+-20%)   :", evaluate(panel, rf, n).sliced(a, b, "DEV+VAL").describe())

    print("\n=== YEARLY RETURNS (strategy vs buy&hold) ===")
    yr = pd.concat([M.yearly_returns(sv.returns), M.yearly_returns(bv.returns)], axis=1, keys=["SMA200", "buy&hold"])
    print(yr.map(lambda x: f"{x:+.1%}").to_string())

    print("\n=== TAIL RISK / STRESS WINDOWS (strategy's own return inside window) ===")
    sw = M.stress_windows(sv.returns)
    sb = M.stress_windows(bv.returns).set_index("window")["strategy_return"].rename("buy&hold_return")
    print(sw.set_index("window").join(sb)[["strategy_return", "max_dd_in_window", "buy&hold_return"]].map(lambda x: f"{x:+.1%}").to_string())
    print("  rolling worst:", {k: f"{v:.1%}" for k, v in M.rolling_worst(sv.returns).items()})

    print("\n=== MONTE CARLO (10,000 resamples) ===")
    tr = S.mc_trades(sv.trades.pnl_frac.to_numpy(), n_sims=DEFAULT.mc_sims)
    print(f"  trades={len(sv.trades)}  trade resample: {tr}")
    print("  daily-return block bootstrap:", S.mc_block_bootstrap(sv.returns, n_sims=DEFAULT.mc_sims))

    print("\n=== DEFLATED SHARPE (selection sample = dev+val) ===")
    n_tr, var = T.n_trials(), T.sr_variance()
    d = S.dsr(M.excess(sv.returns, sv.rf), n_tr, var if var == var else 0.0)
    print(f"  N trials logged so far = {n_tr}; cross-trial SR variance = {var:.3e}")
    print("  " + ", ".join(f"{k}={float(v):.4g}" for k, v in d.items()))
    print("  NOTE: N is tiny here (a handful of nudges of one rule); the real bar is set once hundreds of variants are logged.")

    print("\n=== TAX SENSITIVITY (all gains short-term; tax paid yearly) ===")
    tt = tax_drag_table(sv.returns)
    tt["after_tax_cagr"] = tt["after_tax_cagr"].map("{:.2%}".format)
    tt["total_tax_paid_per_$1"] = tt["total_tax_paid_per_$1"].map("${:.3f}".format)
    print(tt.to_string())

    REPORTS.mkdir(exist_ok=True)
    a0 = window(panel, "DEV+VAL")[0]
    equity_plot({"SMA200 trend (net of costs)": strat.returns.loc[a0:], "buy&hold S&P 500 (price only)": bench.returns.loc[a0:]},
                REPORTS / "demo_sma200_sp500.png", "PHASE-1 DEMO: 200-day SMA trend vs buy&hold, S&P 500 1999-2018",
                caveats=CAVEATS, subtitle="dev | val boundaries dashed; holdout not in this dataset; NOT a research finding")
    print("\nsaved reports/demo_sma200_sp500.png")


if __name__ == "__main__":
    main()
