"""Kalshi daily-high-temperature markets vs NBM forecasts (design pre-registered in RESEARCH_LOG.md section 9)."""
from __future__ import annotations

import datetime as dt
import math
import re
import sys

sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from lab import metrics as M
from lab import trials as T
from lab.config import DATA_RAW, PREDICTION_SPLITS
from lab.prediction import kalshi_fee

CITY_STATION = {"KXHIGHNY": "KNYC", "KXHIGHCHI": "KMDW", "KXHIGHMIA": "KMIA", "KXHIGHAUS": "KAUS", "KXHIGHLAX": "KLAX",
                "KXHIGHDEN": "KDEN", "KXHIGHPHIL": "KPHL"}
MON = {m: i for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split(), 1)}
DT_SPEC = {"DT1": (-1, 20), "DT2": (0, 10)}     # (day offset relative to event day D, hour UTC)
AVAIL_LAG_H = 3
SLIP = 0.01


def event_date(ev: str) -> pd.Timestamp:
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})$", ev)
    return pd.Timestamp(2000 + int(m.group(1)), MON[m.group(2)], int(m.group(3)))


def build_city(series: str, station: str | None = None, nbm_model: str = "NBS") -> pd.DataFrame:
    """One row per (market, decision time) with point-in-time forecast, quotes at the decision time and the outcome."""
    station = station or CITY_STATION[series]
    mk = pd.read_parquet(DATA_RAW / "kalshi" / f"{series}_markets.parquet")
    cd = pd.read_parquet(DATA_RAW / "kalshi" / f"{series}_candles.parquet")
    fc = pd.read_parquet(DATA_RAW / "iem" / f"{nbm_model}_{station}.parquet")
    fc = fc[fc.ftime.dt.hour == 0][["runtime", "ftime", "txn", "xnd"]].dropna(subset=["txn"]).sort_values("runtime")
    mk = mk[mk.result.isin(["yes", "no"]) & mk.expiration_value.notna()].copy()
    mk["D"] = mk.event_ticker.map(event_date)
    mk["y"] = (mk.result == "yes").astype(int)
    cd = cd.sort_values("end_ts")
    out = []
    for name, (off, hr) in DT_SPEC.items():
        e = mk.copy()
        e["dt_name"] = name
        e["dt"] = (e.D + pd.to_timedelta(off, "D") + pd.to_timedelta(hr, "h")).dt.tz_localize("UTC")
        e["dt_ts"] = e.dt.map(lambda x: int(x.timestamp()))
        e["ftime"] = (e.D + pd.Timedelta(days=1)).dt.tz_localize("UTC")
        e["avail"] = e.dt - pd.Timedelta(hours=AVAIL_LAG_H)
        e = e.sort_values("avail")
        e = pd.merge_asof(e, fc, left_on="avail", right_on="runtime", by="ftime", direction="backward", tolerance=pd.Timedelta(hours=12))
        q = pd.merge_asof(e.sort_values("dt_ts")[["ticker", "dt_ts"]], cd[["ticker", "end_ts", "bid_close", "ask_close", "volume", "oi"]],
                          left_on="dt_ts", right_on="end_ts", by="ticker", direction="backward", tolerance=7200)
        e = e.merge(q.drop(columns="dt_ts"), on="ticker", how="left")
        out.append(e)
    df = pd.concat(out, ignore_index=True)
    df["series"], df["station"] = series, station
    return df


def outcome_prob(row_type, floor, cap, mu, sigma):
    """P(YES) for integer-degree max temperature ~ discretised Normal(mu, sigma)."""
    if row_type == "greater":
        return 1.0 - norm.cdf((floor + 0.5 - mu) / sigma)
    if row_type == "less":
        return norm.cdf((cap - 0.5 - mu) / sigma)
    return norm.cdf((cap + 0.5 - mu) / sigma) - norm.cdf((floor - 0.5 - mu) / sigma)


def p_yes(df: pd.DataFrame, a: float, c: float) -> np.ndarray:
    mu = df.txn.to_numpy() + df.bias.to_numpy()
    sig = np.sqrt(a**2 + (c * df.xnd.fillna(2.0).to_numpy()) ** 2)
    st, fl, cp = df.strike_type.to_numpy(), df.floor_strike.to_numpy(), df.cap_strike.to_numpy()
    out = np.empty(len(df))
    g, l, b = st == "greater", st == "less", st == "between"
    out[g] = 1 - norm.cdf((fl[g] + 0.5 - mu[g]) / sig[g])
    out[l] = norm.cdf((cp[l] - 0.5 - mu[l]) / sig[l])
    out[b] = norm.cdf((cp[b] + 0.5 - mu[b]) / sig[b]) - norm.cdf((fl[b] - 0.5 - mu[b]) / sig[b])
    return np.clip(out, 1e-4, 1 - 1e-4)


def add_rolling_bias(df: pd.DataFrame, window: int = 60, min_hist: int = 20) -> pd.DataFrame:
    """Point-in-time city bias: mean residual (actual - txn) of the previous `window` settled events, per series and decision time.
    DT1 (evening before) cannot yet know event D-1's result; DT2 (morning of D) knows D-1's."""
    out = []
    for (ser, name), g in df.groupby(["series", "dt_name"]):
        ev = g.drop_duplicates("D").sort_values("D").copy()
        ev["res"] = ev.expiration_value - ev.txn
        lag = 2 if name == "DT1" else 1
        ev["bias"] = ev.res.shift(lag).rolling(window, min_periods=min_hist).mean()
        out.append(g.merge(ev[["D", "bias"]], on="D", how="left"))
    return pd.concat(out, ignore_index=True)


def fit_sigma(df: pd.DataFrame, dt_name: str, splits=PREDICTION_SPLITS) -> dict:
    """Fit sigma = sqrt(a^2 + (c*xnd)^2) on DEVELOPMENT events only (one row per event), residuals net of the rolling bias."""
    ev = df[(df.dt_name == dt_name) & (df.D <= pd.Timestamp(splits.dev_end))].drop_duplicates(["series", "D"]).dropna(subset=["txn", "bias"])
    e = (ev.expiration_value - ev.txn - ev.bias).to_numpy()
    xnd = ev.xnd.fillna(2.0).to_numpy()

    def nll(p):
        a, c = abs(p[0]) + 1e-3, abs(p[1])
        s = np.sqrt(a**2 + (c * xnd) ** 2)
        pr = norm.cdf((e + 0.5) / s) - norm.cdf((e - 0.5) / s)
        return -np.log(np.clip(pr, 1e-12, None)).sum()

    r = minimize(nll, x0=[1.5, 0.5], method="Nelder-Mead")
    return {"a": abs(r.x[0]) + 1e-3, "c": abs(r.x[1]), "n_dev_events": len(ev), "mae_dev": float(np.abs(e).mean()), "sd_dev": float(e.std())}


def forecast_error_report(df: pd.DataFrame) -> pd.DataFrame:
    ev = df[df.dt_name == "DT1"].drop_duplicates(["series", "D"]).dropna(subset=["txn"]).copy()
    ev["err"] = ev.expiration_value - ev.txn
    ev["year"] = ev.D.dt.year
    return ev.groupby(["series", "year"]).err.agg(["count", "mean", "std", lambda x: x.abs().mean()]).rename(columns={"<lambda_0>": "mae"}).round(2)


def simulate(df: pd.DataFrame, pyes: np.ndarray, theta: float, slip: float = SLIP, fee_mult: float = 1.0, bankroll: float = 5000.0,
             frac: float = 0.01, topk: int = 8, vol_frac: float = 0.25, mode: str = "model", longshot: float | None = None) -> pd.DataFrame:
    """Returns a trades DataFrame (one row per trade) with pnl in USD. Fills: YES at ask + slip; NO at (1 - bid) + slip."""
    d = df.copy()
    d["p"] = pyes
    ask, bid = d.ask_close.to_numpy(), d.bid_close.to_numpy()
    ok = np.isfinite(ask) & np.isfinite(bid) & (d.volume.fillna(0).to_numpy() > 0)
    py = np.where(ok & (ask < 0.99) & (ask > 0.0), np.minimum(ask + slip, 0.99), np.nan)          # price to buy YES
    pn = np.where(ok & (bid > 0.01) & (bid < 1.0), np.minimum(1.0 - bid + slip, 0.99), np.nan)    # price to buy NO
    fee_c = lambda p: 0.07 * fee_mult * p * (1 - p)
    if mode == "model":
        ey, en = d.p.to_numpy() - py - fee_c(py), (1 - d.p.to_numpy()) - pn - fee_c(pn)
        side_yes = np.nan_to_num(ey, nan=-9) >= np.nan_to_num(en, nan=-9)
        edge = np.where(side_yes, ey, en)
        sel = np.isfinite(edge) & (edge >= theta)
    else:                                   # model-free favourite rule: buy whichever side is the favourite priced within `longshot` of 1
        fy, fn = py >= 1 - longshot, pn >= 1 - longshot
        side_yes = fy
        edge = np.where(fy, 1.0, np.where(fn, 1.0, np.nan))
        sel = (fy | fn) & np.isfinite(np.where(fy, py, pn))
    d["side_yes"], d["edge"], d["price"] = side_yes, edge, np.where(side_yes, py, pn)
    d = d[sel].copy()
    if mode == "model" and topk:
        d = d.sort_values("edge", ascending=False).groupby(["dt_name", "D"]).head(topk)
    n = np.floor(frac * bankroll / d.price).clip(lower=1)
    n = np.minimum(n, np.maximum(1, np.floor(vol_frac * d.volume.fillna(0))))
    d["n"] = n.astype(int)
    win = np.where(d.side_yes, d.y == 1, d.y == 0)
    d["fee"] = [kalshi_fee(int(k), float(p)) * fee_mult for k, p in zip(d.n, d.price)]
    d["pnl"] = d.n * (win.astype(float) - d.price) - d.fee
    d["win"] = win
    return d


def daily_returns(trades: pd.DataFrame, first: pd.Timestamp, last: pd.Timestamp, bankroll: float = 5000.0) -> pd.Series:
    idx = pd.date_range(first, last, freq="D")
    s = trades.groupby("D").pnl.sum() / bankroll if len(trades) else pd.Series(dtype=float)
    return s.reindex(idx).fillna(0.0)


def summarize_trades(trades: pd.DataFrame, first, last, bankroll=5000.0) -> dict:
    r = daily_returns(trades, first, last, bankroll)
    sm = M.summarize(r, None, 365)
    if not sm:
        return {}
    from lab.stats import sr_period
    sm["sr_period"] = sr_period(r)
    sm["n_trades"] = int(len(trades))
    sm["win_rate"] = float(trades.win.mean()) if len(trades) else float("nan")
    sm["avg_pnl_per_trade"] = float(trades.pnl.mean()) if len(trades) else float("nan")
    return sm
