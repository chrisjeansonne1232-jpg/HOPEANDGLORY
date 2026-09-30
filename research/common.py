"""Shared Phase-2 research runner. Every call to Runner.run() (a) runs the look-ahead truncation test the first time a
strategy function is seen, (b) backtests with realistic costs on the RESEARCH panel only (dev+val; the holdout is never
loaded here), (c) logs the variant to trials.csv (DEV+VAL row drives N and the Sharpe variance; DEV and VAL rows are
informational), (d) returns a summary dict. Nothing here can see 2024+ data."""
from __future__ import annotations

import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from lab import metrics as M
from lab import trials as T
from lab.config import CRYPTO_COST, CRYPTO_SPLITS, EQUITY_SPLITS, CostModel, SplitConfig
from lab.data import cache
from lab.engine import buy_and_hold, run_backtest, run_session_only
from lab.lookahead import check_lookahead
from lab.panel import Panel, rf_from_annual
from lab.splits import research_panel

# Half-spread assumptions in bps (deliberately generous vs typical quotes; SPY/QQQ are ~0.1-0.5bp in reality).
SPREADS = {"SPY": 0.5, "QQQ": 0.7, "IWM": 1.0, "EFA": 1.0, "EEM": 2.0, "TLT": 1.0, "IEF": 1.5, "SHY": 1.5, "LQD": 2.0, "HYG": 2.0,
           "AGG": 1.5, "TIP": 2.0, "GLD": 1.0, "SLV": 2.0, "DBC": 5.0, "VNQ": 2.0, "SVXY": 8.0, "VXX": 8.0, "UVXY": 10.0,
           **{f"XL{c}": 2.0 for c in "BEFIKPUVY"}}
COST = CostModel(half_spread_bps=2.0, half_spread_bps_by_asset=SPREADS, slippage_bps=1.0)
SRC_YAHOO = "yahoo chart API (adj close = dividend+split adjusted), fetched 2026-09-30"


def load_panel(symbols, src="yahoo", ppy=252, caveats=None) -> Panel:
    frames = {s: cache.load(src, s) for s in symbols}
    miss = [s for s, f in frames.items() if f is None]
    if miss:
        raise FileNotFoundError(f"not cached: {miss}")
    cav = ["ETF universe = funds that exist TODAY (closed/merged ETFs absent): mild survivorship bias",
           "adj_close is Yahoo's dividend/split adjustment (unofficial)"] + (caveats or [])
    return Panel.from_frames(frames, ppy=ppy, source=SRC_YAHOO if src == "yahoo" else src, caveats=cav)


def rf_series(index: pd.DatetimeIndex) -> pd.Series:
    irx = cache.load("yahoo", "^IRX")["close"].ffill()      # 13-week T-bill discount yield, percent
    return rf_from_annual(irx / 100.0, index)


class Runner:
    def __init__(self, symbols, start: str, universe: str, splits: SplitConfig = EQUITY_SPLITS, bench: str = "SPY",
                 cost: CostModel = COST, ppy: int = 252, src: str = "yahoo", exec_mode: str = "next_open", verbose=True):
        need = list(dict.fromkeys(list(symbols) + [bench]))
        self.full = load_panel(need, src=src, ppy=ppy)
        self.panel = research_panel(self.full, splits)          # holdout is physically removed from research
        self.splits, self.universe, self.cost, self.exec_mode, self.bench_sym = splits, universe, cost, exec_mode, bench
        self.start = pd.Timestamp(start)
        self.rf = rf_series(self.panel.index)
        self.checked: set[str] = set()
        self.verbose = verbose
        self.bench = buy_and_hold(self.panel, bench, cost=cost, rf_period=self.rf, splits=splits, exec_mode=exec_mode)
        self.results: list[dict] = []

    def _slice(self, res, split):
        a, b = self.splits.bounds(split, self.start, self.panel.index[-1])
        return res.sliced(max(a, self.start), b, split)

    def run(self, family: str, variant: str, params: dict, fn, rationale: str, exec_mode: str | None = None, cost=None,
            cost_mult: float = 1.0, allow_short=False, max_gross=1.0, notes: str = "", log: bool = True, session: str | None = None) -> dict:
        key = f"{family}:{getattr(fn, '__qualname__', str(fn))}"
        if key not in self.checked:
            check_lookahead(fn, self.panel)
            self.checked.add(key)
        w = fn(self.panel)
        c = (cost or self.cost).scaled(cost_mult) if cost_mult != 1.0 else (cost or self.cost)
        if session:
            res = run_session_only(self.panel, w, session, cost=c, rf_period=self.rf, label=f"{family}/{variant}",
                                   allow_short=allow_short, splits=self.splits)
            exec_mode = f"session_{session}"
        else:
            res = run_backtest(self.panel, w, cost=c, rf_period=self.rf, exec_mode=exec_mode or self.exec_mode,
                               label=f"{family}/{variant}", allow_short=allow_short, max_gross=max_gross, splits=self.splits,
                               account_size=5000.0)
        views = {s: self._slice(res, s) for s in ("DEVELOPMENT", "VALIDATION", "DEV+VAL")}
        bviews = {s: self._slice(self.bench, s) for s in views}
        summ = {s: v.summary() for s, v in views.items()}
        bsum = {s: v.summary() for s, v in bviews.items()}
        out = {"family": family, "variant": variant, "params": params, "cost_mult": cost_mult, "res": res}
        for s, k in (("DEVELOPMENT", "dev"), ("VALIDATION", "val"), ("DEV+VAL", "dv")):
            sm = summ[s]
            out.update({f"{k}_cagr": sm.get("cagr"), f"{k}_sharpe": sm.get("sharpe"), f"{k}_maxdd": sm.get("max_dd"),
                        f"{k}_trades": sm.get("n_trades"), f"{k}_bench_sharpe": bsum[s].get("sharpe"), f"{k}_bench_cagr": bsum[s].get("cagr")})
        out["worst_month"] = summ["DEV+VAL"].get("worst_month")
        out["exposure"] = summ["DEV+VAL"].get("exposure")
        if log:
            rng = f"{views['DEV+VAL'].returns.index[0].date()}..{views['DEV+VAL'].returns.index[-1].date()}"
            for s, v in views.items():
                T.log_trial(family=family, variant=variant, params={**params, "exec": exec_mode or self.exec_mode}, universe=self.universe,
                            data_source=self.full.source, data_range=rng, split=s, metrics=summ[s], n_trades=summ[s].get("n_trades"),
                            cost_mult=cost_mult, bench_sharpe=bsum[s].get("sharpe"), rationale=rationale, notes=notes)
        self.results.append(out)
        if self.verbose:
            print(f"  {family}/{variant:<34} DEV sh {out['dev_sharpe']:+.2f} cagr {out['dev_cagr']:+.1%} | VAL sh {out['val_sharpe']:+.2f} "
                  f"cagr {out['val_cagr']:+.1%} | DV sh {out['dv_sharpe']:+.2f} (SPY {out['dv_bench_sharpe']:+.2f}) dd {out['dv_maxdd']:.0%} "
                  f"trades {out['dv_trades']}")
        return out
