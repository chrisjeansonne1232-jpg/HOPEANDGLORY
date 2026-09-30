"""Trials ledger. EVERY strategy variant evaluated must be logged here, failures included: the row count is the
`n_trials` used by the Deflated Sharpe Ratio. Append-only CSV so a crashed/ended session loses nothing."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os

import numpy as np
import pandas as pd

from .config import TRIALS_CSV

COLUMNS = ["trial_id", "timestamp_utc", "family", "variant", "params_json", "universe", "data_source", "data_range",
           "split", "cost_mult", "n_periods", "n_trades", "cagr", "sharpe_ann", "sr_period", "skew", "kurt", "max_dd",
           "bench_sharpe", "rationale", "status", "notes"]
# status: tested | needs_data | infrastructure   (only 'tested' rows count toward N)
# rationale: "economic:<reason written BEFORE testing>" | "data-mining" | "replication:<citation>"


def trial_id(family: str, variant: str, params: dict, universe: str) -> str:
    blob = json.dumps([family, variant, params, universe], sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def load(path=TRIALS_CSV) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path)


def log_trial(*, family: str, variant: str, params: dict, universe: str, data_source: str, data_range: str, split: str,
              metrics: dict | None = None, n_trades: int | None = None, cost_mult: float = 1.0, bench_sharpe: float | None = None,
              rationale: str = "data-mining", status: str = "tested", notes: str = "", path=TRIALS_CSV,
              skip_duplicates: bool = True) -> str:
    """Append one evaluation. `metrics` is the dict from lab.metrics.summarize (per-period Sharpe is recomputed here)."""
    tid = trial_id(family, variant, params, universe)
    df = load(path)
    if skip_duplicates and len(df):
        dup = df[(df.trial_id == tid) & (df.split == split) & (df.cost_mult == cost_mult) & (df.data_range == data_range)]
        if len(dup):
            return tid
    m = metrics or {}
    n_periods = m.get("n_periods")
    sr_ann = m.get("sharpe")
    row = {
        "trial_id": tid, "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "family": family,
        "variant": variant, "params_json": json.dumps(params, sort_keys=True, default=str), "universe": universe,
        "data_source": data_source, "data_range": data_range, "split": split, "cost_mult": cost_mult, "n_periods": n_periods,
        "n_trades": n_trades, "cagr": m.get("cagr"), "sharpe_ann": sr_ann, "sr_period": m.get("sr_period"),
        "skew": m.get("skew"), "kurt": m.get("kurt"), "max_dd": m.get("max_dd"), "bench_sharpe": bench_sharpe,
        "rationale": rationale, "status": status, "notes": notes.replace("\n", " "),
    }
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
    return tid


def n_trials(path=TRIALS_CSV) -> int:
    """Number of distinct strategy variants tested (the N for the Deflated Sharpe Ratio)."""
    df = load(path)
    return int(df[df.status == "tested"].trial_id.nunique()) if len(df) else 0


def sr_variance(path=TRIALS_CSV, selection_split: str = "DEV+VAL") -> float:
    """Cross-trial variance of per-period Sharpe ratios measured on the selection sample (dev+val)."""
    df = load(path)
    if not len(df):
        return float("nan")
    df = df[(df.status == "tested") & (df.split == selection_split)].drop_duplicates("trial_id")
    v = df.sr_period.astype(float).dropna()
    return float(v.var(ddof=1)) if len(v) > 2 else float("nan")
