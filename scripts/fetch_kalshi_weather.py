"""Queue: Kalshi daily-high markets + hourly candles for the pre-registered city set, and IEM NBM/GFS forecasts for their stations."""
import sys
sys.path.insert(0, ".")
from lab.data import kalshi, iem

SERIES = sys.argv[1].split(",") if len(sys.argv) > 1 else ["KXHIGHCHI", "KXHIGHMIA", "KXHIGHAUS", "KXHIGHLAX", "KXHIGHDEN", "KXHIGHPHIL"]
for s in SERIES:
    kalshi.fetch_series(s, workers=6)
