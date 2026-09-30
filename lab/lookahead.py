"""Generic look-ahead detector (truncation test).

A strategy is a callable `f(panel) -> DataFrame of target weights indexed by decision date`. If it only uses data
available at each decision time, then its weights on dates <= t must be IDENTICAL whether it is given the full panel or
the panel truncated at t. Anything that peeks (shift(-1), centered rolling windows, full-sample normalisation,
fitting on the whole history, ...) changes its past output when the future is removed and is caught here.

Run this on every new strategy before its first backtest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .panel import Panel


class LookAheadError(AssertionError):
    pass


def check_lookahead(strategy, panel: Panel, n_checks: int = 15, seed: int = 0, min_frac: float = 0.3, atol: float = 1e-9) -> bool:
    w_full = strategy(panel)
    n = len(panel.index)
    if n < 50:
        raise ValueError("panel too short for a meaningful look-ahead check")
    rng = np.random.default_rng(seed)
    cuts = np.sort(rng.choice(np.arange(int(n * min_frac), n - 1), size=min(n_checks, n - 1 - int(n * min_frac)), replace=False))
    for k in cuts:
        t = panel.index[k]
        w_cut = strategy(panel.truncate(t))
        a = w_full.reindex(w_cut.index)[w_cut.columns]
        na_mismatch = a.isna().to_numpy() != w_cut.isna().to_numpy()
        val_mismatch = ~np.isclose(a.fillna(0.0).to_numpy(), w_cut.fillna(0.0).to_numpy(), atol=atol, rtol=0.0)
        bad = na_mismatch | val_mismatch
        if bad.any():
            r, c = np.argwhere(bad)[0]
            raise LookAheadError(f"weights on {w_cut.index[r].date()} for {w_cut.columns[c]} change when data after "
                                 f"{t.date()} is removed: full={a.iloc[r, c]!r} vs truncated={w_cut.iloc[r, c]!r}")
    return True
