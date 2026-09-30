# Data validation report

| dataset | rows | first | last | errors | notable warnings |
|---|---|---|---|---|---|
| coinbase/BTC-USD_86400 | 4091 | 2015-07-20 | 2026-09-30 | 0 | extreme_returns; open_equals_prev_close |
| coinbase/ETH-USD_86400 | 3786 | 2016-05-18 | 2026-09-30 | 0 | extreme_returns; open_equals_prev_close |
| yahoo/AGG | 5787 | 2003-09-29 | 2026-09-29 | 0 | - |
| yahoo/BIL | 4865 | 2007-05-30 | 2026-09-29 | 0 | open_equals_prev_close |
| yahoo/DBC | 5194 | 2006-02-06 | 2026-09-29 | 0 | - |
| yahoo/EEM | 5903 | 2003-04-14 | 2026-09-29 | 0 | - |
| yahoo/EFA | 6309 | 2001-08-27 | 2026-09-29 | 0 | - |
| yahoo/GLD | 5499 | 2004-11-18 | 2026-09-29 | 0 | - |
| yahoo/HYG | 4899 | 2007-04-11 | 2026-09-29 | 0 | - |
| yahoo/IEF | 6081 | 2002-07-30 | 2026-09-29 | 0 | - |
| yahoo/IWM | 6624 | 2000-05-26 | 2026-09-29 | 0 | - |
| yahoo/LQD | 6081 | 2002-07-30 | 2026-09-29 | 0 | - |
| yahoo/QQQ | 6932 | 1999-03-10 | 2026-09-29 | 0 | - |
| yahoo/SHY | 6081 | 2002-07-30 | 2026-09-29 | 0 | - |
| yahoo/SLV | 5137 | 2006-04-28 | 2026-09-29 | 0 | - |
| yahoo/SPY | 8474 | 1993-01-29 | 2026-09-29 | 0 | - |
| yahoo/SVXY | 3768 | 2011-10-04 | 2026-09-29 | 0 | extreme_returns |
| yahoo/TIP | 5739 | 2003-12-05 | 2026-09-29 | 0 | - |
| yahoo/TLT | 6081 | 2002-07-30 | 2026-09-29 | 0 | - |
| yahoo/UVXY | 3768 | 2011-10-04 | 2026-09-29 | 0 | extreme_returns |
| yahoo/VNQ | 5535 | 2004-09-29 | 2026-09-29 | 0 | - |
| yahoo/VXX | 2181 | 2018-01-25 | 2026-09-29 | 0 | extreme_returns |
| yahoo/XLB | 6984 | 1998-12-22 | 2026-09-29 | 0 | - |
| yahoo/XLE | 6984 | 1998-12-22 | 2026-09-29 | 0 | - |
| yahoo/XLF | 6984 | 1998-12-22 | 2026-09-29 | 0 | - |
| yahoo/XLI | 6984 | 1998-12-22 | 2026-09-29 | 0 | - |
| yahoo/XLK | 6984 | 1998-12-22 | 2026-09-29 | 0 | - |
| yahoo/XLP | 6984 | 1998-12-22 | 2026-09-29 | 0 | - |
| yahoo/XLU | 6984 | 1998-12-22 | 2026-09-29 | 0 | - |
| yahoo/XLV | 6984 | 1998-12-22 | 2026-09-29 | 0 | - |
| yahoo/XLY | 6984 | 1998-12-22 | 2026-09-29 | 0 | - |
| yahoo/^GSPC | 9253 | 1990-01-02 | 2026-09-29 | 0 | open_equals_prev_close |
| yahoo/^IRX | 9222 | 1990-01-02 | 2026-09-29 | 6 | extreme_returns; stale_prices; open_equals_prev_close |
|  | | | | ERROR | [ERROR] nonpositive_open: 8 rows <= 0 |
|  | | | | ERROR | [ERROR] nonpositive_high: 5 rows <= 0 |
|  | | | | ERROR | [ERROR] nonpositive_low: 9 rows <= 0 |
|  | | | | ERROR | [ERROR] nonpositive_close: 7 rows <= 0 |
|  | | | | ERROR | [ERROR] high_below_ohlc: 1 rows, first 2020-03-27 |
|  | | | | ERROR | [ERROR] low_above_ohlc: 3 rows, first 2020-03-20 |
| yahoo/^SKEW | 9179 | 1990-01-02 | 2026-09-29 | 0 | - |
| yahoo/^TNX | 9222 | 1990-01-02 | 2026-09-29 | 0 | extreme_returns |
| yahoo/^VIX | 9255 | 1990-01-02 | 2026-09-29 | 0 | extreme_returns |
| yahoo/^VIX3M | 5083 | 2006-07-17 | 2026-09-29 | 0 | extreme_returns |
| yahoo/^VVIX | 4957 | 2007-01-03 | 2026-09-29 | 0 | extreme_returns |

## Cross-checks
- SPY vs ^GSPC daily price-return correlation 0.98348 over 8473 days; max |diff| 0.0310 on 2000-01-07
- SPY total-return (adj close) CAGR 10.82% vs price-only CAGR 8.85% (gap = dividends, ~1.5-2% expected)
- SPY adj-return minus index price-return: mean 1.81%/yr (dividend yield less fee, ~+1.3-1.9% expected)

## Fraction of opens equal to prior close (unreliable-open indicator)

| symbol | fraction |
|---|---|
| ^GSPC | 0.40 |
| BIL | 0.39 |
| BTC-USD_86400 | 0.33 |
| ETH-USD_86400 | 0.28 |
| ^IRX | 0.24 |
| SHY | 0.11 |
| XLP | 0.05 |
| ^TNX | 0.04 |
| HYG | 0.04 |
| XLU | 0.04 |
| XLF | 0.04 |
| XLV | 0.04 |