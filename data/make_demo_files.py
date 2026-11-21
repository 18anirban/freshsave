# ====================================================================
# DEMO FILES for the Demo tab  (one store, 8 upload files)
# Cut from the HIDDEN TEST data, so the models have never seen it.
#   day_0_opening_stock.csv   stock on the shelf at the end of Day 0
#   day_1_sales.csv ... day_7_sales.csv   each evening's sales report
#   answer_key_hidden.csv     what really happened (used only by the
#                             final dashboard, never by the prediction)
# Run:  python data/make_demo_files.py
# ====================================================================
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
STORE = "STORE_01"          # large urban store
DAY0 = "2024-11-03"         # Sunday; Day 1 = Monday 4 Nov
DAYS = 7
OUTCOMES = ["units_wasted_final", "waste_cost", "at_risk"]

# STEP 1: take the hidden test data of ONE store
test = pd.read_csv(ROOT / "store" / "store_test_30.csv")
test = test[test["store_id"] == STORE]

# STEP 2: one file per day. Outcome columns are removed: tomorrow is unknown.
dates = pd.date_range(DAY0, periods=DAYS + 1).strftime("%Y-%m-%d")
names = ["day_0_opening_stock.csv"] + [f"day_{i}_sales.csv" for i in range(1, DAYS + 1)]
out = ROOT / "demo"
for date, name in zip(dates, names):
    day = test[test["date"] == date].sort_values(["category", "batch_id"])
    day.drop(columns=OUTCOMES).to_csv(out / name, index=False)
    print(f"{name:26s} {date}  rows {len(day)}")

# STEP 3: answer key = the real outcome of every batch seen in the week
week = test[test["date"].isin(dates)]
key = (week.groupby("batch_id")
       .agg(store_id=("store_id", "first"), category=("category", "first"),
            sub_category=("sub_category", "first"), shelf_life_days=("shelf_life_days", "first"),
            first_date=("date", "min"), last_date_in_week=("date", "max"),
            units_received=("units_received", "first"), units_wasted_final=("units_wasted_final", "first"),
            at_risk=("at_risk", "first"))
       .reset_index())
# a batch is "finished" if its last evening (expiry eve) falls inside the week
last_day = test.groupby("batch_id")["date"].max()
key["batch_ends_in_week"] = key["batch_id"].map(last_day).isin(dates)
key.to_csv(out / "answer_key_hidden.csv", index=False)
print(f"answer_key_hidden.csv      {len(key)} batches, {int(key['batch_ends_in_week'].sum())} end inside the week")
