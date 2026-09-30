"""Generic Phase-3 evaluator. A candidate is described by daily net-return series (already produced by the lab with realistic costs) and
callbacks' outputs; this module applies the pre-registered pass/fail definitions (RESEARCH_LOG.md section 1) and returns a verdict table.
It never loads the holdout: holdout results, if any, are passed in by the caller after lab.splits.open_holdout()."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lab import metrics as M
from lab import stats as S
from lab import trials as T
from lab.config import DEFAULT


def _cagr_sharpe(r, rf, ppy):
    r = r.dropna()
    return (M.cagr(r, ppy), M.sharpe(r, rf, ppy)) if len(r) > 10 else (np.nan, np.nan)


def evaluate(name: str, ret: dict[str, pd.Series], bench: pd.Series, rf: pd.Series | None, ppy: int, trade_pnl: np.ndarray | None,
             oos_trades: int | None, nudges: dict[str, dict[str, pd.Series]] | None = None, cost2x: dict[str, pd.Series] | None = None,
             related: dict[str, dict[str, pd.Series]] | None = None, holdout: pd.Series | None = None, max_dd_limit: float = 0.25,
             windows: dict | None = None) -> dict:
    """ret: {'DEVELOPMENT','VALIDATION','DEV+VAL'} -> net return series. nudges/related: {label: {split: series}}."""
    v = {"name": name, "checks": {}}
    dv = ret["DEV+VAL"]
    rows = {}
    for s in ("DEVELOPMENT", "VALIDATION", "DEV+VAL"):
        c, sh = _cagr_sharpe(ret[s], rf, ppy)
        bc, bsh = _cagr_sharpe(bench.reindex(ret[s].index).dropna(), rf, ppy)
        rows[s] = {"cagr": c, "sharpe": sh, "bench_cagr": bc, "bench_sharpe": bsh, "max_dd": M.max_drawdown(ret[s])}
    v["by_split"] = pd.DataFrame(rows).T
    chk = v["checks"]
    chk["profitable_dev_and_val"] = bool(rows["DEVELOPMENT"]["cagr"] > 0 and rows["VALIDATION"]["cagr"] > 0)
    chk["beats_bench_sharpe_dev+val"] = bool(rows["DEV+VAL"]["sharpe"] > rows["DEV+VAL"]["bench_sharpe"])
    v["beats_bench_absolute_return"] = bool(rows["DEV+VAL"]["cagr"] > rows["DEV+VAL"]["bench_cagr"])
    n, var = T.n_trials(), T.sr_variance()
    d = S.dsr(M.excess(dv, rf), n, var if var == var else 0.0)
    v["dsr"] = d
    chk["DSR>=0.95"] = bool(d["dsr"] >= 0.95)
    chk[">=100_oos_trades"] = bool(oos_trades is not None and oos_trades >= 100)
    v["oos_trades"] = oos_trades
    if nudges:
        ok = all(_cagr_sharpe(s["DEVELOPMENT"], rf, ppy)[0] > 0 and _cagr_sharpe(s["VALIDATION"], rf, ppy)[0] > 0 for s in nudges.values())
        v["nudges"] = pd.DataFrame({k: {"dev_sharpe": _cagr_sharpe(s["DEVELOPMENT"], rf, ppy)[1], "val_sharpe": _cagr_sharpe(s["VALIDATION"], rf, ppy)[1],
                                        "dv_sharpe": _cagr_sharpe(s["DEV+VAL"], rf, ppy)[1]} for k, s in nudges.items()}).T
        chk["params_+-20%_profitable"] = bool(ok)
    if related:
        ok = all(_cagr_sharpe(s["DEV+VAL"], rf, ppy)[0] > 0 for s in related.values())
        v["related"] = pd.DataFrame({k: {"dv_cagr": _cagr_sharpe(s["DEV+VAL"], rf, ppy)[0], "dv_sharpe": _cagr_sharpe(s["DEV+VAL"], rf, ppy)[1]} for k, s in related.items()}).T
        chk["related_assets_profitable"] = bool(ok)
    v["yearly"] = M.yearly_returns(dv).to_frame("strategy").join(M.yearly_returns(bench.reindex(dv.index).dropna()).rename("bench"))
    v["tail"] = {"max_dd": M.max_drawdown(dv), "worst_day": M.worst_day(dv), "worst_month": M.worst_month(dv), **M.rolling_worst(dv, (1, 5, 21, 63))}
    sw = M.stress_windows(dv, windows)
    v["stress"] = sw
    worst_event = min([sw.strategy_return.min() if len(sw) else 0.0, v["tail"].get("worst_21p", 0.0)])
    v["worst_event"] = worst_event
    chk["max_dd<=25%"] = bool(abs(v["tail"]["max_dd"]) <= max_dd_limit)
    chk["worst_event_loss<=25%"] = bool(worst_event >= -max_dd_limit)
    if trade_pnl is not None and len(trade_pnl) >= 5:
        v["mc_trades"] = S.mc_trades(trade_pnl, n_sims=DEFAULT.mc_sims)
    v["mc_block"] = S.mc_block_bootstrap(dv, n_sims=DEFAULT.mc_sims, block=21 if ppy > 100 else 3)
    mcb = v["mc_block"]
    chk["MC_p05_final_equity>1"] = bool(mcb.get("final_equity_p05", 0) > 1.0)
    if cost2x:
        s2 = cost2x
        c2 = {sp: _cagr_sharpe(s2[sp], rf, ppy) for sp in ("DEVELOPMENT", "VALIDATION", "DEV+VAL")}
        d2 = S.dsr(M.excess(s2["DEV+VAL"], rf), n, var if var == var else 0.0)
        v["cost2x"] = {"by_split": {k: {"cagr": a, "sharpe": b} for k, (a, b) in c2.items()}, "dsr": d2["dsr"]}
        chk["2x_costs_profitable_and_DSR>=0.95"] = bool(c2["DEVELOPMENT"][0] > 0 and c2["VALIDATION"][0] > 0 and d2["dsr"] >= 0.95)
    if holdout is not None:
        hc, hs = _cagr_sharpe(holdout, rf, ppy)
        v["holdout"] = {"cagr": hc, "sharpe": hs, "max_dd": M.max_drawdown(holdout), "n": len(holdout)}
        chk["profitable_in_holdout"] = bool(hc > 0)
    v["passed_all"] = all(chk.values())
    return v


def to_markdown(v: dict) -> str:
    out = [f"### {v['name']}", "", "| split | CAGR | Sharpe | bench CAGR | bench Sharpe | max DD |", "|---|---|---|---|---|---|"]
    for s, r in v["by_split"].iterrows():
        out.append(f"| {s} | {r.cagr:.1%} | {r.sharpe:.2f} | {r.bench_cagr:.1%} | {r.bench_sharpe:.2f} | {r.max_dd:.1%} |")
    d = v["dsr"]
    out += ["", f"DSR = **{d['dsr']:.3f}** (N={d['n_trials']}, hurdle SR0 {d['sr0_annualized_equiv']:.2f}/yr equiv, PSR vs 0 = {d['psr_vs_zero']:.3f}); "
            f"beats bench on absolute return: {v['beats_bench_absolute_return']}; OOS trades: {v['oos_trades']}", "", "| check | result |", "|---|---|"]
    out += [f"| {k} | {'PASS' if ok else 'FAIL'} |" for k, ok in v["checks"].items()]
    out += ["", f"**Overall: {'PASSED ALL' if v['passed_all'] else 'DID NOT PASS'}**"]
    return "\n".join(out)
