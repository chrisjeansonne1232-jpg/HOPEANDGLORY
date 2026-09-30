"""Plots and text tables. Every plot carries its data caveats and split boundaries in the picture itself."""
from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import metrics as M
from .config import EQUITY_SPLITS, SplitConfig


def equity_plot(curves: dict[str, pd.Series], path, title: str, splits: SplitConfig | None = EQUITY_SPLITS,
                caveats: list[str] | None = None, log: bool = True, subtitle: str = "") -> None:
    """curves: name -> net period returns. Draws growth of $1 (log) and drawdowns, with dev/val/holdout boundaries."""
    fig, (ax, ad) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for name, r in curves.items():
        r = r.dropna()
        ax.plot(M.equity_curve(r), label=name, lw=1.4)
        ad.plot(M.drawdown_series(r), lw=1.0)
    if log:
        ax.set_yscale("log")
    ax.set_ylabel("growth of $1" + (" (log)" if log else ""))
    ad.set_ylabel("drawdown")
    ad.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    lo = min(r.index.min() for r in curves.values())
    hi = max(r.index.max() for r in curves.values())
    if splits is not None:
        for d, lab in ((splits.dev_end, "DEV | VAL"), (splits.val_end, "VAL | HOLDOUT")):
            ts = pd.Timestamp(d)
            if lo < ts < hi:
                for a in (ax, ad):
                    a.axvline(ts, color="k", ls="--", lw=0.8)
                ax.text(ts, ax.get_ylim()[0], " " + lab, va="bottom", fontsize=8)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    ad.grid(alpha=0.3)
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""), fontsize=11)
    if caveats:
        fig.text(0.01, 0.005, textwrap.fill("CAVEATS: " + " | ".join(caveats), 170), fontsize=6.5, va="bottom", color="firebrick")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fmt_summary_table(rows: dict[str, dict]) -> str:
    """rows: label -> summary dict; returns a fixed-width text table."""
    cols = [("cagr", "CAGR", "{:.2%}"), ("ann_vol", "Vol", "{:.1%}"), ("sharpe", "Sharpe", "{:.2f}"), ("max_dd", "MaxDD", "{:.1%}"),
            ("worst_period", "WorstPer", "{:.1%}"), ("worst_month", "WorstMo", "{:.1%}"), ("n_trades", "Trades", "{:.0f}")]
    out = [f"{'':<46}" + "".join(f"{h:>10}" for _, h, _ in cols)]
    for lab, s in rows.items():
        cells = []
        for k, _, f in cols:
            v = s.get(k)
            cells.append(f"{f.format(v):>10}" if v is not None and v == v else f"{'-':>10}")
        out.append(f"{lab:<46}" + "".join(cells))
    return "\n".join(out)
