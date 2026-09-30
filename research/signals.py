"""Pure signal builders: each returns fn(panel) -> DataFrame of target weights decided at the close of bar t from data <= t.
Every builder is exercised by lab.lookahead.check_lookahead (inside Runner.run) before its first backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd
import exchange_calendars as xc

from lab.data import cache

_CAL = None


def _cal():
    global _CAL
    if _CAL is None:
        _CAL = xc.get_calendar("XNYS", start="1990-01-02", end="2027-12-31")
    return _CAL


def first_bar_of_month(index: pd.DatetimeIndex) -> np.ndarray:
    p = index.to_period("M").to_numpy()
    return np.r_[True, p[1:] != p[:-1]]


def monthly_hold(w: pd.DataFrame) -> pd.DataFrame:
    """Only re-decide on the first bar of each month (truncation-safe: needs no knowledge of the future calendar)."""
    reb = pd.Series(first_bar_of_month(w.index), index=w.index)
    return w.where(reb, np.nan).ffill().fillna(0.0)


def price_above_sma(assets, n):
    def f(panel):
        c = panel.adj_close[assets]
        m = c.rolling(n).mean()
        return (c > m).where(m.notna(), False).astype(float)
    return f


def momentum_sign(assets, n):
    def f(panel):
        c = panel.adj_close[assets]
        r = c / c.shift(n) - 1.0
        return (r > 0).where(r.notna(), False).astype(float)
    return f


def sma_cross(assets, fast, slow):
    def f(panel):
        c = panel.adj_close[assets]
        a, b = c.rolling(fast).mean(), c.rolling(slow).mean()
        return (a > b).where(b.notna(), False).astype(float)
    return f


def equal_slot_trend(assets, n, monthly=True):
    """Each asset owns a 1/N slot, held only while its price is above its n-day SMA; else that slot is cash."""
    base = price_above_sma(assets, n)
    def f(panel):
        w = base(panel) / len(assets)
        return monthly_hold(w) if monthly else w
    return f


def xs_momentum(assets, lookback, skip, top_k, abs_filter, hold_monthly=True):
    def f(panel):
        c = panel.adj_close[assets]
        mom = c.shift(skip) / c.shift(lookback) - 1.0
        rank = mom.rank(axis=1, ascending=False, method="first")
        sel = (rank <= top_k) & mom.notna()
        if abs_filter:
            sel = sel & (mom > 0)
        w = sel.astype(float) / top_k
        return monthly_hold(w) if hold_monthly else w
    return f


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up, dn = d.clip(lower=0), (-d).clip(lower=0)
    au, ad = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean(), dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + au / ad.replace(0, np.nan))


def rsi_reversal(assets, rsi_n, entry, exit_sma, trend_sma):
    """Connors-style: enter when RSI(n) < entry while price > trend_sma; hold until close > SMA(exit_sma)."""
    def f(panel):
        out = {}
        for a in assets:
            c = panel.adj_close[a]
            rsi, ex, tr = _rsi(c, rsi_n).to_numpy(), c.rolling(exit_sma).mean().to_numpy(), c.rolling(trend_sma).mean().to_numpy()
            cv = c.to_numpy()
            pos, w = 0.0, np.zeros(len(c))
            for i in range(len(c)):
                if pos == 0.0:
                    if np.isfinite(rsi[i]) and np.isfinite(tr[i]) and rsi[i] < entry and cv[i] > tr[i]:
                        pos = 1.0
                elif np.isfinite(ex[i]) and cv[i] > ex[i]:
                    pos = 0.0
                w[i] = pos
            out[a] = w
        return pd.DataFrame(out, index=panel.index)
    return f


def ibs_reversal(assets, thr, trend_sma=None):
    """Internal bar strength: long for the next holding period when close is in the bottom `thr` of the day's range."""
    def f(panel):
        c, h, l = panel.close[assets], panel.high[assets], panel.low[assets]
        ibs = (c - l) / (h - l).replace(0, np.nan)
        sig = (ibs < thr)
        if trend_sma:
            a = panel.adj_close[assets]
            sig = sig & (a > a.rolling(trend_sma).mean())
        return sig.fillna(False).astype(float)
    return f


def vol_target(assets, target, lookback, monthly=True):
    def f(panel):
        r = panel.adj_close[assets].pct_change()
        vol = r.rolling(lookback).std() * np.sqrt(panel.ppy)
        w = (target / vol).clip(upper=1.0).where(vol.notna(), 0.0)
        return monthly_hold(w) if monthly else w
    return f


def trend_and_voltarget(assets, n, target, lookback):
    t, v = price_above_sma(assets, n), vol_target(assets, target, lookback)
    def f(panel):
        return t(panel) * v(panel)
    return f


def short_vol_term_structure(asset, cap, threshold, vix_max=None):
    """Long the inverse-VIX ETP (weight `cap`) only while VIX/VIX3M < threshold (contango); else cash."""
    def f(panel):
        vix = cache.load("yahoo", "^VIX")["close"].reindex(panel.index).ffill()
        v3 = cache.load("yahoo", "^VIX3M")["close"].reindex(panel.index).ffill()
        ok = (vix / v3) < threshold
        if vix_max:
            ok = ok & (vix < vix_max)
        return pd.DataFrame({asset: cap * ok.fillna(False).astype(float)}, index=panel.index)
    return f


def _sessions():
    s = _cal().sessions_in_range("1990-01-02", "2027-12-31")
    df = pd.DataFrame(index=s)
    ym = s.to_period("M")
    df["dom"] = pd.Series(1, index=s).groupby(ym).cumsum().to_numpy()
    df["dom_end"] = pd.Series(1, index=s).groupby(ym).transform("count").to_numpy() - df["dom"].to_numpy()   # 0 = last session of month
    return df


def turn_of_month(asset, days_after):
    """Hold across the last session of the month and the first `days_after` sessions (calendar known in advance)."""
    def f(panel):
        s = _sessions().reindex(panel.index)
        on = (s["dom_end"] == 0) | (s["dom"] <= days_after - 1)
        return pd.DataFrame({asset: on.fillna(False).astype(float)}, index=panel.index)
    return f


def day_of_week(asset, dow):
    def f(panel):
        return pd.DataFrame({asset: (panel.index.dayofweek == dow).astype(float)}, index=panel.index)
    return f


def always_long(asset):
    def f(panel):
        return pd.DataFrame({asset: 1.0}, index=panel.index)
    return f


def vix_contango_filter(asset, threshold):
    """Hold `asset` only while VIX/VIX3M < threshold (term structure in contango = calm), else cash."""
    def f(panel):
        vix = cache.load("yahoo", "^VIX")["close"].reindex(panel.index).ffill()
        v3 = cache.load("yahoo", "^VIX3M")["close"].reindex(panel.index).ffill()
        ok = ((vix / v3) < threshold).fillna(False)
        return pd.DataFrame({asset: ok.astype(float)}, index=panel.index)
    return f


def trend_voltarget_crypto(asset, n, target, lookback=30):
    """Trend filter x inverse-vol sizing (weight = min(1, target / realised annualised vol))."""
    def f(panel):
        c = panel.adj_close[[asset]]
        trend = (c > c.rolling(n).mean()).astype(float)
        vol = c.pct_change().rolling(lookback).std() * np.sqrt(panel.ppy)
        w = (target / vol).clip(upper=1.0).where(vol.notna(), 0.0)
        return trend * w
    return f


def inverse_vol_slots(assets, n, vol_lb=60, monthly=True):
    """Trend flags x inverse-vol weights, gross exposure capped at 1 (a risk-parity flavoured GTAA)."""
    base = price_above_sma(assets, n)
    def f(panel):
        vol = panel.adj_close[assets].pct_change().rolling(vol_lb).std()
        iv = (1.0 / vol).where(vol > 0)
        w = base(panel) * iv
        w = w.div(iv.sum(axis=1), axis=0).fillna(0.0)      # weights among ALL assets sum to 1; off-trend slots stay in cash
        return monthly_hold(w) if monthly else w
    return f


def pairs_zscore(a, b, lookback, entry, exit_z=0.5):
    """Dollar-neutral pair: z-score of log(P_a/P_b) vs its rolling mean/std. z > entry -> short a / long b; z < -entry -> long a / short b;
    close when |z| < exit_z. Each leg carries 0.5 of equity (gross 1)."""
    def f(panel):
        la, lb = np.log(panel.adj_close[a]), np.log(panel.adj_close[b])
        sp = la - lb
        z = ((sp - sp.rolling(lookback).mean()) / sp.rolling(lookback).std()).to_numpy()
        pos, out = 0.0, np.zeros(len(z))
        for i in range(len(z)):
            if not np.isfinite(z[i]):
                out[i] = 0.0
                continue
            if pos == 0.0:
                if z[i] > entry:
                    pos = -1.0
                elif z[i] < -entry:
                    pos = 1.0
            elif (pos < 0 and z[i] < exit_z) or (pos > 0 and z[i] > -exit_z):
                pos = 0.0
            out[i] = pos
        return pd.DataFrame({a: 0.5 * out, b: -0.5 * out}, index=panel.index)
    return f
