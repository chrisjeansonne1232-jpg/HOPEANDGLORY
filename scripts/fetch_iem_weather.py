import sys
sys.path.insert(0, ".")
from lab.data import iem
for st in sys.argv[1].split(","):
    for m in ("NBS", "GFS"):
        d = iem.fetch_mos(st, m, "2021-06", "2026-09")
        print(st, m, len(d), d.runtime.min(), d.runtime.max(), flush=True)
