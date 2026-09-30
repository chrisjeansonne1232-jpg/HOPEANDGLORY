"""Performance and tail-risk metrics. All functions take per-period simple returns (decimal)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

# Approximate market-stress windows (peak -> trough of the S&P 500; dates are well-documented events, the
# metric reported is the STRATEGY's own return in the window, not the index's).
STRESS_WINDOWS = {
    "1929 crash (Sep29-Jun32, monthly data only)": ("1929-09-30", "1932-06-30"),
    "1973-74 bear (monthly data only)": ("1973-01-31", "1974-09-30"),
    "1987 crash month (monthly data only)": ("1987-09-30", "1987-11-30"),
    "2000-02 dotcom bear": ("2000-03-24", "2002-10-09"),
    "2008 GFC panic (Sep15-Nov20)": ("2008-09-15", "2008-11-20"),
    "Aug 2015 flash crash": ("2015-08-17", "2015-08-25"),
    "Feb 2018 Volmageddon": ("2018-01-26", "2018-02-08"),
    "Q4 2018 selloff": ("2018-09-20", "2018-12-24"),
    "Mar 2020 COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 bear": ("2022-01-03", "2022-10-12"),
    "Aug 2024 vol spike": ("2024-07-16", "2024-08-05"),
    "Apr 2025 tariff shock": ("2025-04-02", "2025-04-08"),
}


def excess(r: pd.Series, rf: pd.Series | None) -> pd.Series:
    if rf is None:
        return r
    return r - rf.reindex(r.index).fillna(0.0)


def sharpe(r: pd.Series, rf: pd.Series | None = None, ppy: int = 252) -> float:
    x = excess(r, rf).dropna()
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1) * np.sqrt(ppy))


def sortino(r: pd.Series, rf: pd.Series | None = None, ppy: int = 252) -> float:
    x = excess(r, rf).dropna()
    dd = np.sqrt((np.minimum(x, 0.0) ** 2).mean())
    return float(x.mean() / dd * np.sqrt(ppy)) if dd > 0 else float("nan")


def equity_curve(r: pd.Series, start: float = 1.0) -> pd.Series:
    return start * (1.0 + r).cumprod()


def cagr(r: pd.Series, ppy: int = 252) -> float:
    n = len(r)
    if n == 0:
        return float("nan")
    eq = float((1.0 + r).prod())
    return eq ** (ppy / n) - 1.0 if eq > 0 else -1.0


def ann_vol(r: pd.Series, ppy: int = 252) -> float:
    return float(r.std(ddof=1) * np.sqrt(ppy)) if len(r) > 2 else float("nan")


def drawdown_series(r: pd.Series) -> pd.Series:
    eq = (1.0 + r).cumprod()
    peak = np.maximum(eq.cummax(), 1.0)
    return eq / peak - 1.0


def max_drawdown(r: pd.Series) -> float:
    return float(drawdown_series(r).min()) if len(r) else float("nan")


def max_dd_duration(r: pd.Series) -> int:
    """Longest run of consecutive periods spent below a previous equity peak."""
    dd = drawdown_series(r).to_numpy()
    longest = cur = 0
    for v in dd:
        cur = cur + 1 if v < 0 else 0
        longest = max(longest, cur)
    return longest


def worst_day(r: pd.Series) -> float:
    return float(r.min()) if len(r) else float("nan")


def monthly_returns(r: pd.Series) -> pd.Series:
    return (1.0 + r).groupby(r.index.to_period("M")).prod() - 1.0


def worst_month(r: pd.Series) -> float:
    m = monthly_returns(r)
    return float(m.min()) if len(m) else float("nan")


def yearly_returns(r: pd.Series) -> pd.Series:
    y = (1.0 + r).groupby(r.index.year).prod() - 1.0
    y.index.name = "year"
    return y


def cvar(r: pd.Series, q: float = 0.05) -> float:
    if len(r) < 20:
        return float("nan")
    cut = r.quantile(q)
    return float(r[r <= cut].mean())


def rolling_worst(r: pd.Series, windows=(1, 5, 21, 63)) -> dict:
    out = {}
    lr = np.log1p(r)
    for w in windows:
        if len(r) >= w:
            out[f"worst_{w}p"] = float(np.expm1(lr.rolling(w).sum().min()))
    return out


def stress_windows(r: pd.Series, windows: dict | None = None) -> pd.DataFrame:
    """Strategy return and max intra-window drawdown in named stress windows that lie inside the data."""
    rows = []
    for name, (a, b) in (windows or STRESS_WINDOWS).items():
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if r.index.min() > a or r.index.max() < b:
            continue
        seg = r.loc[a:b]
        if len(seg) < 2:
            continue
        rows.append({"window": name, "start": a.date(), "end": b.date(), "strategy_return": float((1 + seg).prod() - 1),
                     "max_dd_in_window": float(drawdown_series(seg).min())})
    return pd.DataFrame(rows)


def summarize(r: pd.Series, rf: pd.Series | None = None, ppy: int = 252) -> dict:
    r = r.dropna()
    if len(r) == 0:
        return {}
    return {
        "start": r.index[0].date(), "end": r.index[-1].date(), "n_periods": len(r),
        "cagr": cagr(r, ppy), "ann_vol": ann_vol(r, ppy), "sharpe": sharpe(r, rf, ppy), "sortino": sortino(r, rf, ppy),
        "max_dd": max_drawdown(r), "max_dd_len_periods": max_dd_duration(r),
        "calmar": (cagr(r, ppy) / abs(max_drawdown(r))) if max_drawdown(r) < 0 else float("nan"),
        "worst_period": worst_day(r), "worst_month": worst_month(r), "cvar95": cvar(r),
        "total_return": float((1 + r).prod() - 1), "skew": float(sps.skew(r)), "kurt": float(sps.kurtosis(r, fisher=False)),
        "pct_positive": float((r > 0).mean()),
    }


def compare(r: pd.Series, bench: pd.Series, rf: pd.Series | None = None, ppy: int = 252) -> dict:
    """Strategy vs benchmark on the common window: Sharpe (with Jobson-Korkie/Memmel p-value), CAGR, beta/alpha."""
    from .stats import sharpe_diff_test

    df = pd.concat([r, bench], axis=1, join="inner").dropna()
    df.columns = ["s", "b"]
    s, b = df["s"], df["b"]
    cov = np.cov(excess(s, rf).reindex(df.index), excess(b, rf).reindex(df.index))
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else float("nan")
    alpha = (excess(s, rf).reindex(df.index).mean() - beta * excess(b, rf).reindex(df.index).mean()) * ppy
    z, p = sharpe_diff_test(excess(s, rf).reindex(df.index), excess(b, rf).reindex(df.index))
    return {
        "sharpe_strategy": sharpe(s, rf, ppy), "sharpe_benchmark": sharpe(b, rf, ppy),
        "sharpe_diff": sharpe(s, rf, ppy) - sharpe(b, rf, ppy), "sharpe_diff_p_one_sided": p,
        "cagr_strategy": cagr(s, ppy), "cagr_benchmark": cagr(b, ppy),
        "beats_on_sharpe": sharpe(s, rf, ppy) > sharpe(b, rf, ppy), "beats_on_cagr": cagr(s, ppy) > cagr(b, ppy),
        "maxdd_strategy": max_drawdown(s), "maxdd_benchmark": max_drawdown(b),
        "beta": float(beta), "alpha_ann": float(alpha), "corr": float(df.corr().iloc[0, 1]),
    }
