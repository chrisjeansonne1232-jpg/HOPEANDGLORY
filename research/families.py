"""Phase-2 families with PRE-REGISTERED grids (this file is committed BEFORE the first run of each family). Every variant is a
trial. Each family lists the economic rationale (who is on the other side and why they keep losing) or marks itself data-mining."""
from __future__ import annotations

import itertools
import numpy as np
import pandas as pd

from lab.config import CRYPTO_COST, CRYPTO_SPLITS
from research import signals as S
from research.common import Runner

R_TREND = ("economic: slow-moving institutional/behavioural under-reaction and forced de-risking create multi-month autocorrelation; "
           "counterparty = late/price-insensitive holders. Literature: Moskowitz-Ooi-Pedersen 2012, Faber 2007.")
R_XSMOM = ("economic: herding and slow diffusion of information across sectors/asset classes; losers keep being sold by "
           "constrained institutions. Literature: Jegadeesh-Titman 1993, Moskowitz-Grinblatt 1999, Antonacci dual momentum.")
R_CAL = ("economic: predictable fund/pension/payroll flows at month turns; overnight premium = intraday liquidity provision "
         "compensated at the open. Literature: Lakonishok-Smidt 1988, Ritter-Kelly overnight (2011), Cliff-Cooper-Gulen 2008.")
R_REV = ("economic: short-horizon reversal = compensation for liquidity provision to forced/impatient sellers "
         "(Nagel 2012); counterparty = sellers who must trade now. Connors RSI2 / IBS are practitioner formulations.")
R_VOL = ("economic: risk is more predictable than return, so scaling exposure inversely to recent vol raises Sharpe "
         "(Moreira-Muir 2017); counterparty = leveraged/benchmarked investors who cannot de-risk.")
R_SVOL = ("economic: variance risk premium - hedgers overpay for insurance (VIX futures roll down in contango); counterparty = "
          "hedgers. TAIL RISK: Feb-2018 one-day -83% in SVXY => weight capped so a repeat cannot lose >25%.")
R_CRYPTO = ("economic: retail/flow-driven trends in a market with high frictions and few arbitrageurs; counterparty = "
            "late retail momentum chasers. Retail fees ~0.6%/side => only low-turnover rules can survive.")
R_DM = "data-mining"


def f01_trend_spy(verbose=True):
    r = Runner(["SPY"], "1994-04-15", "SPY", verbose=verbose)
    for n in (100, 150, 200, 250, 300):
        r.run("trend_sma", f"SPY_sma{n}", {"n": n}, S.price_above_sma(["SPY"], n), R_TREND)
    for n in (126, 189, 252):
        r.run("trend_mom", f"SPY_mom{n}", {"n": n}, S.momentum_sign(["SPY"], n), R_TREND)
    for fast, slow in ((50, 200), (20, 100), (40, 160)):
        r.run("trend_cross", f"SPY_sma{fast}x{slow}", {"fast": fast, "slow": slow}, S.sma_cross(["SPY"], fast, slow), R_TREND)
    return r


def f01_trend_related(verbose=True):
    starts = {"QQQ": "2000-06-01", "IWM": "2001-06-01", "EFA": "2002-09-01", "EEM": "2004-06-01", "TLT": "2003-09-01",
              "IEF": "2003-09-01", "GLD": "2005-12-01", "VNQ": "2005-12-01"}
    out = []
    for a, st in starts.items():
        r = Runner([a], st, a, verbose=verbose)
        for n in (160, 200, 240):
            r.run("trend_sma_related", f"{a}_sma{n}", {"n": n, "asset": a}, S.price_above_sma([a], n), R_TREND,
                  notes="related-asset robustness of the SMA rule (benchmark is still SPY buy&hold)")
        out.append(r)
    return out


def f02_multiasset(verbose=True):
    rs = []
    a5 = ["SPY", "EFA", "IEF", "GLD", "VNQ"]
    r = Runner(a5, "2006-01-03", "GTAA5", verbose=verbose)
    for n in (170, 210, 250):
        r.run("gtaa_trend", f"gtaa5_sma{n}", {"n": n, "assets": a5}, S.equal_slot_trend(a5, n), R_TREND)
    rs.append(r)
    a10 = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "GLD", "VNQ"]
    r = Runner(a10, "2006-01-03", "GTAA10", verbose=verbose)
    for n in (170, 210, 250):
        r.run("gtaa_trend", f"gtaa10_sma{n}", {"n": n, "assets": a10}, S.equal_slot_trend(a10, n), R_TREND)
    rs.append(r)
    sect = [f"XL{c}" for c in "BEFIKPUVY"]
    r = Runner(sect, "2000-01-03", "SECTORS9", verbose=verbose)
    for lb, k, ab in itertools.product((63, 126, 252), (1, 3), (False, True)):
        skip = 21 if lb == 252 else 0
        r.run("sector_momentum", f"sect_L{lb}s{skip}_k{k}_abs{int(ab)}", {"lookback": lb, "skip": skip, "k": k, "abs": ab},
              S.xs_momentum(sect, lb, skip, k, ab), R_XSMOM)
    rs.append(r)
    ac = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG", "GLD", "DBC", "VNQ"]
    r = Runner(ac, "2008-06-02", "ASSETCLASS12", verbose=verbose)
    for lb, k, ab in itertools.product((126, 252), (2, 3), (False, True)):
        skip = 21 if lb == 252 else 0
        r.run("assetclass_momentum", f"ac_L{lb}s{skip}_k{k}_abs{int(ab)}", {"lookback": lb, "skip": skip, "k": k, "abs": ab},
              S.xs_momentum(ac, lb, skip, k, ab), R_XSMOM)
    rs.append(r)
    return rs


def f03_calendar(verbose=True):
    r = Runner(["SPY"], "1993-03-01", "SPY", verbose=verbose)
    for d in (2, 3, 4):
        r.run("turn_of_month", f"SPY_tom{d}", {"days_after": d}, S.turn_of_month("SPY", d), R_CAL)
    for dow, nm in enumerate(("mon", "tue", "wed", "thu", "fri")):
        r.run("day_of_week", f"SPY_{nm}", {"dow": dow}, S.day_of_week("SPY", dow), R_DM + " (weak: documented weekday effects have decayed)")
    r.run("overnight", "SPY_overnight", {}, S.always_long("SPY"), R_CAL, session="overnight")
    r.run("intraday", "SPY_intraday", {}, S.always_long("SPY"), R_CAL, session="intraday")
    q = Runner(["QQQ"], "1999-06-01", "QQQ", verbose=verbose)
    q.run("overnight", "QQQ_overnight", {}, S.always_long("QQQ"), R_CAL, session="overnight")
    q.run("intraday", "QQQ_intraday", {}, S.always_long("QQQ"), R_CAL, session="intraday")
    return [r, q]


def f04_reversal(verbose=True):
    r = Runner(["SPY"], "1994-04-15", "SPY", verbose=verbose)
    for entry in (5, 10, 15):
        r.run("rsi2", f"SPY_rsi2_e{entry}", {"entry": entry, "exit_sma": 5, "trend_sma": 200}, S.rsi_reversal(["SPY"], 2, entry, 5, 200), R_REV)
    for thr in (0.1, 0.2, 0.3):
        r.run("ibs", f"SPY_ibs{thr}", {"thr": thr}, S.ibs_reversal(["SPY"], thr), R_REV)
    r.run("ibs", "SPY_ibs0.2_trend200", {"thr": 0.2, "trend_sma": 200}, S.ibs_reversal(["SPY"], 0.2, 200), R_REV)
    out = [r]
    for a, st in (("QQQ", "2000-06-01"), ("IWM", "2001-06-01")):
        q = Runner([a], st, a, verbose=verbose)
        q.run("rsi2", f"{a}_rsi2_e10", {"entry": 10, "exit_sma": 5, "trend_sma": 200}, S.rsi_reversal([a], 2, 10, 5, 200), R_REV)
        q.run("ibs", f"{a}_ibs0.2", {"thr": 0.2}, S.ibs_reversal([a], 0.2), R_REV)
        out.append(q)
    return out


def f05_volmanaged(verbose=True):
    r = Runner(["SPY"], "1993-08-01", "SPY", verbose=verbose)
    for tgt, lb in itertools.product((0.08, 0.10, 0.12, 0.15), (20, 60)):
        r.run("vol_target", f"SPY_vt{int(tgt*100)}_L{lb}", {"target": tgt, "lookback": lb}, S.vol_target(["SPY"], tgt, lb), R_VOL)
    for tgt in (0.10, 0.15):
        r.run("trend_x_voltarget", f"SPY_sma200_vt{int(tgt*100)}", {"n": 200, "target": tgt, "lookback": 20},
              S.trend_and_voltarget(["SPY"], 200, tgt, 20), R_VOL + " + " + R_TREND)
    return r


def f06_shortvol(verbose=True):
    r = Runner(["SVXY"], "2011-11-01", "SVXY", verbose=verbose)
    for cap, th in itertools.product((0.15, 0.25), (0.90, 0.95, 1.00)):
        r.run("short_vol_ts", f"SVXY_cap{int(cap*100)}_th{th}", {"cap": cap, "threshold": th}, S.short_vol_term_structure("SVXY", cap, th), R_SVOL)
    return r


def f07_crypto(verbose=True):
    out = []
    for sym, st in (("BTC-USD_86400", "2016-02-15"), ("ETH-USD_86400", "2017-01-01")):
        r = Runner([sym], st, sym.split("_")[0], splits=CRYPTO_SPLITS, bench=sym, cost=CRYPTO_COST, ppy=365, src="coinbase", verbose=verbose)
        for n in (20, 50, 100, 150, 200):
            r.run("crypto_trend", f"{sym[:3]}_sma{n}", {"n": n}, S.price_above_sma([sym], n), R_CRYPTO)
        if sym.startswith("BTC"):
            for n in (90, 180):
                r.run("crypto_trend", f"BTC_mom{n}", {"n": n}, S.momentum_sign([sym], n), R_CRYPTO)
            r.run("crypto_weekend", "BTC_weekend_hold", {}, lambda p: pd.DataFrame({sym: np.isin(p.index.dayofweek, (4, 5)).astype(float)}, index=p.index), R_DM + " (weak; documented weekend effects are inconsistent)")
            r.run("crypto_weekend", "BTC_weekday_hold", {}, lambda p: pd.DataFrame({sym: (~np.isin(p.index.dayofweek, (4, 5))).astype(float)}, index=p.index), R_DM)
        out.append(r)
    return out


FAMILIES = {"f01a": f01_trend_spy, "f01b": f01_trend_related, "f02": f02_multiasset, "f03": f03_calendar, "f04": f04_reversal,
            "f05": f05_volmanaged, "f06": f06_shortvol, "f07": f07_crypto}

if __name__ == "__main__":
    import sys
    for k in (sys.argv[1:] or FAMILIES):
        print(f"\n##### {k} #####")
        FAMILIES[k]()
