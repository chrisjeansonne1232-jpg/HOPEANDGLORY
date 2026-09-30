"""Simple US tax drag for a taxable account: all gains treated as short-term (ordinary income), tax paid at each year
end from the account, losses carried forward against later gains (ignores the $3k ordinary-income offset, wash-sale
rules and state specifics). Conservative for trend/short-term strategies; buy-and-hold would defer, so the
after-tax benchmark comparison is shown alongside, not assumed. Section 1256 index options (60/40) are NOT modelled."""
from __future__ import annotations

import pandas as pd


def after_tax_curve(returns: pd.Series, rate: float, start_equity: float = 1.0) -> pd.DataFrame:
    eq, carry, rows = start_equity, 0.0, []
    for year, r in returns.groupby(returns.index.year):
        e0 = eq
        e1 = e0 * float((1 + r).prod())
        gain = e1 - e0 + carry
        tax = rate * gain if gain > 0 else 0.0
        carry = gain if gain < 0 else 0.0
        eq = e1 - tax
        rows.append({"year": year, "pretax_return": e1 / e0 - 1, "tax": tax, "after_tax_return": eq / e0 - 1, "equity_end": eq})
    return pd.DataFrame(rows).set_index("year")


def tax_drag_table(returns: pd.Series, rates=(0.0, 0.12, 0.22, 0.32), ppy: int = 252) -> pd.DataFrame:
    n_years = len(returns) / ppy
    out = []
    for rt in rates:
        c = after_tax_curve(returns, rt)
        out.append({"tax_rate": rt, "after_tax_cagr": c.equity_end.iloc[-1] ** (1 / n_years) - 1, "total_tax_paid_per_$1": c.tax.sum()})
    return pd.DataFrame(out).set_index("tax_rate")
