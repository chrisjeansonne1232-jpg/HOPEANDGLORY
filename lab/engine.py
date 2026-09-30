"""Vectorised backtest engine for target-weight strategies (spot equities / ETFs / spot crypto).

TIMING CONVENTION (the anti-look-ahead contract)
    weights[t] = target portfolio weights DECIDED AT THE CLOSE OF BAR t using data <= t only.
    exec_mode="next_open"  (default, realistic): trade at the OPEN of bar t+1.
    exec_mode="next_close" (conservative)      : trade at the CLOSE of bar t+1 (one full extra bar of delay).
    exec_mode="same_close" (optimistic)        : trade at the close of bar t. Only defensible for low-frequency
                                                 signals executed in the last minutes (e.g. month-end rules).
COSTS (all charged on the bar the trade executes, on the actual traded notional after weight drift):
    half spread + slippage + proportional fee, regulatory sell fees, per-trade commission, short borrow (per calendar
    day held), margin interest on gross long > 1, and cash earns (rf - haircut). `cost_multiplier` scales frictions.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics as M
from .config import DEFAULT, CostModel, EQUITY_SPLITS, SplitConfig
from .panel import Panel
from .splits import guard_panel

EPS = 1e-9
MODES = ("next_open", "next_close", "same_close")


@dataclass
class BacktestResult:
    label: str
    exec_mode: str
    returns: pd.Series          # net of every cost
    gross_returns: pd.Series    # before trading costs, borrow, financing (after cash yield)
    costs: pd.DataFrame         # per-period cost drag by component (fractions of equity)
    pos: pd.DataFrame           # position (weight) after the trade executed on each bar
    turnover: pd.Series         # sum |trade| / equity on each bar
    trades: pd.DataFrame        # position episodes per asset (see _extract_trades)
    rf: pd.Series
    ppy: int
    cost_model: CostModel
    caveats: list
    data_source: str = ""
    split: str = "FULL"

    def sliced(self, start, end, split: str) -> "BacktestResult":
        sl = slice(pd.Timestamp(start), pd.Timestamp(end))
        tr = self.trades
        if len(tr):
            tr = tr[(tr.entry >= pd.Timestamp(start)) & (tr.entry <= pd.Timestamp(end))].reset_index(drop=True)
        return dataclasses.replace(self, returns=self.returns.loc[sl], gross_returns=self.gross_returns.loc[sl],
                                   costs=self.costs.loc[sl], pos=self.pos.loc[sl], turnover=self.turnover.loc[sl],
                                   trades=tr, rf=self.rf.loc[sl], split=split)

    def summary(self) -> dict:
        s = M.summarize(self.returns, self.rf, self.ppy)
        if not s:
            return {"split": self.split, "label": self.label}
        from .stats import sr_period

        s["sr_period"] = sr_period(M.excess(self.returns, self.rf))
        s.update({"split": self.split, "label": self.label, "n_trades": int(len(self.trades)),
                  "ann_turnover": float(self.turnover.mean() * self.ppy),
                  "ann_cost_drag": float(self.costs.sum(axis=1).mean() * self.ppy),
                  "exposure": float(self.pos.abs().sum(axis=1).mean())})
        return s

    def describe(self) -> str:
        s = self.summary()
        if "cagr" not in s:
            return f"[{self.split}] {self.label}: no data"
        return (f"[{self.split}] {self.label} ({s['start']}..{s['end']}, n={s['n_periods']}, trades={s['n_trades']}) "
                f"CAGR {s['cagr']:.2%} | Sharpe {s['sharpe']:.2f} | maxDD {s['max_dd']:.1%} | worst period {s['worst_period']:.1%} | "
                f"worst month {s['worst_month']:.1%} | turnover {s['ann_turnover']:.1f}x/yr | cost drag {s['ann_cost_drag']:.2%}/yr")


def _check_weights(w: pd.DataFrame, max_gross: float, allow_short: bool) -> None:
    if not allow_short and (w < -EPS).any().any():
        raise ValueError("strategy produced short weights but allow_short=False (Robinhood shorting needs margin >= $2,000)")
    gross = w.abs().sum(axis=1)
    if (gross > max_gross + 1e-6).any():
        d = gross.idxmax()
        raise ValueError(f"gross exposure {gross.max():.2f} on {d.date()} exceeds max_gross={max_gross}; "
                         "leverage must be requested explicitly")
    if not np.isfinite(w.to_numpy()).all():
        raise ValueError("non-finite weights")


def run_backtest(panel: Panel, weights: pd.DataFrame, cost: CostModel = DEFAULT.cost, rf_period: pd.Series | None = None,
                 exec_mode: str = "next_open", label: str = "", max_gross: float = 1.0, allow_short: bool = False,
                 account_size: float | None = DEFAULT.account_size, splits: SplitConfig = EQUITY_SPLITS) -> BacktestResult:
    if exec_mode not in MODES:
        raise ValueError(f"exec_mode must be one of {MODES}")
    guard_panel(panel, splits)
    idx = panel.index
    assets = list(weights.columns)
    missing = [a for a in assets if a not in panel.assets]
    if missing:
        raise KeyError(f"weights reference assets not in panel: {missing}")
    extra = weights.index.difference(idx)
    if len(extra):
        raise ValueError(f"weights have {len(extra)} dates not in the price panel (first {extra[0].date()})")

    c = panel.adj_close[assets]
    w = weights.reindex(idx).fillna(0.0)
    w = w.where(c.notna(), 0.0)                  # cannot hold an asset before/after it has prices
    _check_weights(w, max_gross, allow_short)

    dt_days = pd.Series(idx, index=idx).diff().dt.days
    dt_days = dt_days.fillna(dt_days.median() if dt_days.notna().any() else 1).clip(lower=1)
    rf = (rf_period.reindex(idx).fillna(0.0) if rf_period is not None else pd.Series(0.0, index=idx))

    # ---- positions and per-bar returns by execution mode ---------------------------------------------
    if exec_mode == "next_open":
        o = panel.adj_open[assets]
        bad = o.isna() & c.notna()
        if bad.any().any():
            raise ValueError("next_open execution needs an open price wherever a close exists")
        pos = w.shift(1).fillna(0.0)                       # weight held after trading at bar t's open
        prev = pos.shift(1).fillna(0.0)                    # weight carried into bar t (held over the prior close)
        ov = (o / c.shift(1) - 1.0).fillna(0.0)            # prior close -> open
        intr = (c / o - 1.0).fillna(0.0)                   # open -> close
        part_prev = prev * ov
        # the intraday leg is earned on equity that already includes the overnight move: (1+ov_p)(1+id_p)-1
        part_new = (pos * intr).mul(1.0 + part_prev.sum(axis=1), axis=0)
        pre_pos, r_pre = prev, ov                          # drift between last trade and this trade
        held = 0.5 * (prev + pos)
    else:
        pos = w if exec_mode == "same_close" else w.shift(1).fillna(0.0)
        prev = pos.shift(1).fillna(0.0)
        cc = (c / c.shift(1) - 1.0).fillna(0.0)
        part_prev, part_new = prev * cc, pd.DataFrame(0.0, index=idx, columns=assets)
        pre_pos, r_pre = prev, cc
        held = prev

    # ---- cash, drift and turnover ------------------------------------------------------------------
    long_held = held.clip(lower=0.0).sum(axis=1)
    cash_w = 1.0 - long_held
    cash_ret = cash_w.clip(lower=0.0) * (rf - cost.cash_haircut_annual * dt_days / 365.0)
    cash_pre = 1.0 - pre_pos.clip(lower=0.0).sum(axis=1)
    r_p = (pre_pos * r_pre).sum(axis=1) + cash_pre.clip(lower=0.0) * rf
    drifted = pre_pos.mul(1.0 + r_pre).div((1.0 + r_p).clip(lower=1e-6), axis=0)
    trade = (pos - drifted)
    turnover_a = trade.abs()
    turnover_a = turnover_a.where(turnover_a > EPS, 0.0)
    sells = (-trade).clip(lower=0.0).where(turnover_a > 0, 0.0)   # any reduction of position is a sale (incl. opening a short)

    # ---- costs (fractions of equity) ---------------------------------------------------------------
    k = cost.cost_multiplier
    unit_bps = pd.Series({a: cost.half_spread_for(a) + cost.slippage_bps + cost.fee_bps for a in assets}) * k
    trading_a = turnover_a.mul(unit_bps / 1e4, axis=1)
    px_raw = panel.close[assets].replace(0.0, np.nan)
    reg_a = sells * ((cost.sec_fee_per_million_usd / 1e6) + (cost.taf_per_share_usd / px_raw).fillna(0.0)) * k
    comm_a = (turnover_a > 0).astype(float) * ((cost.commission_per_trade_usd * k / account_size) if account_size else 0.0)
    shorts_a = (-held).clip(lower=0.0)
    borrow_a = shorts_a.mul(dt_days / 365.0, axis=0) * (cost.borrow_bps_annual / 1e4) * k
    borrowed = (long_held - 1.0).clip(lower=0.0)
    fin = borrowed * (cost.margin_rate_annual * k) * dt_days / 365.0

    costs = pd.DataFrame({"trading": trading_a.sum(axis=1), "regulatory": reg_a.sum(axis=1), "commission": comm_a.sum(axis=1),
                          "borrow": borrow_a.sum(axis=1), "financing": fin})
    gross = (part_prev + part_new).sum(axis=1) + cash_ret
    net = gross - costs.sum(axis=1)

    long_share = held.clip(lower=0.0).div(long_held.replace(0.0, np.nan), axis=0).fillna(0.0)
    cost_a = trading_a + reg_a + comm_a + borrow_a + long_share.mul(fin, axis=0)
    trades = _extract_trades(pos, part_prev, part_new, cost_a)

    return BacktestResult(label=label, exec_mode=exec_mode, returns=net.rename("net"), gross_returns=gross.rename("gross"),
                          costs=costs, pos=pos, turnover=turnover_a.sum(axis=1), trades=trades, rf=rf, ppy=panel.ppy,
                          cost_model=cost, caveats=list(panel.caveats), data_source=panel.source)


def _extract_trades(pos: pd.DataFrame, part_prev: pd.DataFrame, part_new: pd.DataFrame, cost_a: pd.DataFrame) -> pd.DataFrame:
    """One row per position episode (maximal run of constant-sign non-zero position) per asset.

    pnl_frac is additive P&L as a fraction of account equity (gross return earned by the position that was actually
    held, minus the costs caused by entering/exiting it). ret_on_alloc = pnl_frac / average |weight|.
    """
    idx = pos.index
    rows = []
    for a in pos.columns:
        p = pos[a].to_numpy()
        sgn = np.sign(np.where(np.abs(p) < EPS, 0.0, p))
        prev_sgn = np.concatenate([[0.0], sgn[:-1]])
        run = np.cumsum(sgn != prev_sgn)                       # run id at t (for pos after trade at t)
        prev_run = np.concatenate([[0], run[:-1]])             # run id of the position carried into t
        cost_key = np.where(sgn != 0, run, prev_run)
        n = int(run.max()) + 1
        pnl = (np.bincount(prev_run, weights=part_prev[a].to_numpy(), minlength=n)
               + np.bincount(run, weights=part_new[a].to_numpy(), minlength=n)
               - np.bincount(cost_key, weights=cost_a[a].to_numpy(), minlength=n))
        for r in np.unique(run):
            m = run == r
            s = sgn[m][0]
            if s == 0:
                continue
            e = int(np.argmax(m))
            last = int(len(m) - 1 - np.argmax(m[::-1]))
            is_open = last == len(m) - 1
            x = last + 1 if not is_open else last
            alloc = float(np.abs(p[m]).mean())
            rows.append({"asset": a, "side": "long" if s > 0 else "short", "entry": idx[e], "exit": idx[x],
                         "periods": int(m.sum()), "avg_alloc": alloc, "pnl_frac": float(pnl[r]),
                         "ret_on_alloc": float(pnl[r] / alloc) if alloc > 0 else np.nan, "still_open": bool(is_open)})
    cols = ["asset", "side", "entry", "exit", "periods", "avg_alloc", "pnl_frac", "ret_on_alloc", "still_open"]
    return pd.DataFrame(rows, columns=cols).sort_values("entry").reset_index(drop=True) if rows else pd.DataFrame(columns=cols)


def buy_and_hold(panel: Panel, asset: str, cost: CostModel = DEFAULT.cost, rf_period=None, label: str | None = None,
                 splits: SplitConfig = EQUITY_SPLITS, exec_mode: str = "next_open") -> BacktestResult:
    w = pd.DataFrame(1.0, index=panel.index, columns=[asset])
    return run_backtest(panel, w, cost=cost, rf_period=rf_period, exec_mode=exec_mode, label=label or f"buy&hold {asset}", splits=splits)


def run_session_only(panel: Panel, weights: pd.DataFrame, session: str, cost: CostModel = DEFAULT.cost,
                     rf_period: pd.Series | None = None, label: str = "", allow_short: bool = False,
                     splits: SplitConfig = EQUITY_SPLITS) -> BacktestResult:
    """Hold only part of each day and pay BOTH legs' costs every time.
    session="overnight": buy at close t, sell at open t+1  (weights[t] is a same-close decision: an approximation of a MOC order)
    session="intraday" : buy at open t+1, sell at close t+1 (weights[t] decided at close t, executed next open, exit same close)
    Cash earns rf minus haircut for the periods not held. The whole position is closed each day (no drift)."""
    if session not in ("overnight", "intraday"):
        raise ValueError("session must be 'overnight' or 'intraday'")
    guard_panel(panel, splits)
    idx, assets = panel.index, list(weights.columns)
    c, o = panel.adj_close[assets], panel.adj_open[assets]
    w = weights.reindex(idx).fillna(0.0).where(c.notna(), 0.0)
    _check_weights(w, 1.0, allow_short)
    ov = (o / c.shift(1) - 1.0).fillna(0.0)          # ov[t]: close t-1 -> open t
    intr = (c / o - 1.0).fillna(0.0)
    if session == "overnight":
        pos = w.shift(1).fillna(0.0)                 # pos[t] held from close t-1 to open t
        gross_a = pos * ov
    else:
        pos = w.shift(1).fillna(0.0)                 # pos[t] held from open t to close t
        gross_a = pos * intr
    k = cost.cost_multiplier
    unit = pd.Series({a: cost.half_spread_for(a) + cost.slippage_bps + cost.fee_bps for a in assets}) * k
    trading_a = pos.abs().mul(2.0 * unit / 1e4, axis=1)                        # in AND out, every day
    px_raw = panel.close[assets].replace(0.0, np.nan)
    reg_a = pos.abs() * ((cost.sec_fee_per_million_usd / 1e6) + (cost.taf_per_share_usd / px_raw).fillna(0.0)) * k
    held_frac = pos.clip(lower=0.0).sum(axis=1)
    rf = rf_period.reindex(idx).fillna(0.0) if rf_period is not None else pd.Series(0.0, index=idx)
    dt_days = pd.Series(idx, index=idx).diff().dt.days.fillna(1).clip(lower=1)
    cash_ret = (1.0 - held_frac).clip(lower=0.0) * (rf - cost.cash_haircut_annual * dt_days / 365.0)
    costs = pd.DataFrame({"trading": trading_a.sum(axis=1), "regulatory": reg_a.sum(axis=1), "commission": 0.0, "borrow": 0.0, "financing": 0.0})
    gross = gross_a.sum(axis=1) + cash_ret
    net = gross - costs.sum(axis=1)
    rows = []
    for a in assets:
        m = pos[a].abs() > EPS
        for t in idx[m]:
            pnl = float(gross_a.loc[t, a] - trading_a.loc[t, a] - reg_a.loc[t, a])
            rows.append({"asset": a, "side": "long" if pos.loc[t, a] > 0 else "short", "entry": t, "exit": t, "periods": 1,
                         "avg_alloc": float(abs(pos.loc[t, a])), "pnl_frac": pnl, "ret_on_alloc": pnl / float(abs(pos.loc[t, a])),
                         "still_open": False})
    trades = pd.DataFrame(rows, columns=["asset", "side", "entry", "exit", "periods", "avg_alloc", "pnl_frac", "ret_on_alloc", "still_open"])
    return BacktestResult(label=label, exec_mode=f"session_{session}", returns=net.rename("net"), gross_returns=gross.rename("gross"),
                          costs=costs, pos=pos, turnover=2.0 * pos.abs().sum(axis=1), trades=trades, rf=rf, ppy=panel.ppy,
                          cost_model=cost, caveats=list(panel.caveats), data_source=panel.source)
