"""Development / validation / holdout separation, enforced in code.

* `run_backtest` refuses any panel that extends past the validation end unless it was opened through
  `open_holdout`, which appends to an append-only ledger and refuses a second look for the same strategy_id.
* This is an audit trail, not a cryptographic lock: whoever runs the code could edit the ledger. The ledger
  is committed to git so any tampering is visible in history.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os

import pandas as pd

from .config import EQUITY_SPLITS, HOLDOUT_LEDGER, SplitConfig
from .panel import Panel

LEDGER_COLUMNS = ["timestamp_utc", "strategy_id", "params_json", "splits", "data_source", "holdout_first", "holdout_last", "reason"]


class HoldoutViolation(RuntimeError):
    pass


def research_panel(panel: Panel, splits: SplitConfig = EQUITY_SPLITS) -> Panel:
    """What a researcher may look at: everything up to and including the validation end date."""
    return panel.slice(end=splits.val_end)


def guard_panel(panel: Panel, splits: SplitConfig) -> None:
    if not panel.holdout_open and panel.index.max() > pd.Timestamp(splits.val_end):
        raise HoldoutViolation(
            f"panel ends {panel.index.max().date()} which is inside the FINAL HOLDOUT (> {splits.val_end}). "
            "Pass lab.splits.research_panel(panel) while researching; use open_holdout() once per finalist."
        )


def load_ledger(path=HOLDOUT_LEDGER) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    return pd.read_csv(path)


def open_holdout(panel_full: Panel, strategy_id: str, params: dict, reason: str, splits: SplitConfig = EQUITY_SPLITS,
                 ledger_path=HOLDOUT_LEDGER) -> Panel:
    """Return the FULL panel (holdout included), flagged as open. One look per strategy_id, ever."""
    led = load_ledger(ledger_path)
    if len(led) and (led.strategy_id == strategy_id).any():
        raise HoldoutViolation(f"holdout already used for strategy_id={strategy_id!r} on "
                               f"{led[led.strategy_id == strategy_id].timestamp_utc.iloc[0]}. One look per finalist.")
    ho = panel_full.slice(start=pd.Timestamp(splits.val_end) + pd.Timedelta(days=1))
    if len(ho.index) == 0:
        raise HoldoutViolation("no data after the validation end; nothing to open")
    new = not os.path.exists(ledger_path)
    with open(ledger_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
        if new:
            w.writeheader()
        w.writerow({"timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "strategy_id": strategy_id,
                    "params_json": json.dumps(params, sort_keys=True, default=str), "splits": splits.name,
                    "data_source": panel_full.source, "holdout_first": str(ho.index[0].date()),
                    "holdout_last": str(ho.index[-1].date()), "reason": reason})
    out = panel_full.slice()
    out.holdout_open = True
    return out
