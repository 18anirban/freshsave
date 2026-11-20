"""
STEP 2 of 3 - CREATE THE 14 DAILY FILES   (run once:  python src/make_daily_files.py)
=====================================================================================
In a real store the inventory system exports a closing-stock file every evening.
We do not have that system, so THIS SCRIPT plays its role for the demo store.

Each file = what is on the shelf at closing time that evening (one row per batch).
  REAL (copied from Kaggle): product, category, quantity received, prices, storage/handling scores,
                             and the final total sold (the simulated path ends exactly on it).
  SIMULATED:                 delivery day, compressed shelf life, the quantity left on each evening.
There are NO result columns (waste %, profit) - a real store would not know them that evening.
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src import features as F
from src.simulate import build_long

DELIVERIES = sorted(list(C.DELIVERY_SUNDAYS) + list(C.DELIVERY_WEDNESDAYS))   # [1, 4, 8, 11]


def build_demo_batches() -> pd.DataFrame:
    """Take the demo store's 2024 batches and give each one a delivery day and a demo shelf life."""
    b = F.prepare_batches(F.load_raw())
    b = b[(b["store_id"] == C.STORE_ID) & (b["transaction_date"] >= C.SPLIT_DATE)].copy()
    b = b.sort_values(["transaction_date", "batch_id"]).reset_index(drop=True)
    rng = np.random.default_rng(C.SEED + 7)
    short = b["category"].isin(C.SHORT_LIFE_CATEGORIES).to_numpy()
    # Long-life categories arrive on Sundays only; short-life categories get a Wednesday top-up as well.
    b["delivery_day"] = np.where(short, rng.choice(DELIVERIES, size=len(b)),
                                 rng.choice(list(C.DELIVERY_SUNDAYS), size=len(b)))
    # Every batch must finish inside the 14-day window -> cap its shelf life (waste fraction is unchanged).
    b["life_demo"] = np.minimum(b["life"], C.N_DAYS - b["delivery_day"] + 1)
    return b


def make_files(out_dirs):
    b = build_demo_batches()
    long = build_long(b.assign(life=b["life_demo"]), C.SEED + 2)       # one row per batch per selling day
    start = pd.Timestamp(C.DEMO_START)
    long["day_no"] = long["delivery_day"] + long["selling_days_elapsed"] - 1
    long["date"] = start + pd.to_timedelta(long["day_no"] - 1, unit="D")
    long["delivered_date"] = start + pd.to_timedelta(long["delivery_day"] - 1, unit="D")
    long["expiry_date"] = long["delivered_date"] + pd.to_timedelta(long["life_demo"] - 1, unit="D")
    long["current_price"] = long["base_price"]               # baseline file: no markdown applied yet
    cols = ["date", "day_no", "batch_id", "store_id", "category", "product_name", "delivered_date", "expiry_date",
            "selling_days_elapsed", "days_left", "qty_received", "qty_left", "units_sold_today", "units_sold_total",
            "base_price", "current_price", "cost_price"] + [c for c in F.STATIC_COLS if c not in ("base_price", "cost_price")]
    written = []
    for d in range(1, C.N_DAYS + 1):
        day = long[long["day_no"] == d][cols].sort_values(["days_left", "batch_id"])
        name = f"Day{d:02d}_{(start + pd.Timedelta(days=d - 1)).strftime('%a')}.csv"
        for folder in out_dirs:
            folder.mkdir(parents=True, exist_ok=True)
            day.to_csv(folder / name, index=False, date_format="%Y-%m-%d")
        written.append((name, len(day), int((day["days_left"] == 0).sum())))
    return b, written


def main():
    out_dirs = [C.DAILY_DIR]
    if os.name == "nt":                                   # on Windows also write to Downloads\DailyFiles-Retail
        out_dirs.append(C.WINDOWS_DAILY_DIR)
    b, written = make_files(out_dirs)
    print(f"Demo store {C.STORE_ID}: {len(b)} batches, deliveries on days {DELIVERIES}")
    print(b.groupby("delivery_day").size().rename("batches").to_string())
    print("\nFile              rows on shelf   expiring tonight")
    for n, r, e in written:
        print(f"{n:<16}  {r:>8}   {e:>8}")
    print("\nSaved to:", ", ".join(str(d) for d in out_dirs))


if __name__ == "__main__":
    main()
