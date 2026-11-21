# ====================================================================
# DEMO FILES for the Demo tab  (one store, 8 upload files)
# Cut from the HIDDEN TEST data, so the models have never seen it.
#
#   Day 0  = Saturday night: inventory loaded for the coming week
#   Day 1  = Sunday evening closing ... Day 7 = Saturday evening closing
#
#   day_0_opening_stock.csv   shelf at Saturday night (before Sunday's sales)
#   day_1_sales.csv ... day_7_sales.csv   each evening's closing report
#   answer_key_hidden.csv     what really happened (used only by the
#                             final dashboard, never by the prediction)
# Run:  python data/make_demo_files.py
# ====================================================================
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
STORE = "STORE_01"          # large urban store
DAY0 = "2024-11-02"         # Saturday night; Day 1 = Sunday 3 Nov, Day 7 = Saturday 9 Nov
DAYS = 7
OUTCOMES = ["units_wasted_final", "waste_cost", "at_risk"]

# STEP 1: take the hidden test data of ONE store
test = pd.read_csv(ROOT / "store" / "store_test_30.csv")
test = test[test["store_id"] == STORE]
dates = pd.date_range(DAY0, periods=DAYS + 1).strftime("%Y-%m-%d")
by_day = {d: test[test["date"] == d] for d in dates}

# STEP 2: Day 0 = Saturday night inventory
#   old batches   -> their real Saturday closing count (what is left on the shelf)
#   new batches   -> delivered for Sunday: full delivery on the shelf, nothing sold yet
sat, sun = by_day[dates[0]], by_day[dates[1]]
new = sun[~sun["batch_id"].isin(sat["batch_id"])].copy()
new["date"] = dates[0]
new["days_left"] = new["shelf_life_days"]          # full shelf life ahead
new["units_left"] = new["units_received"]          # nothing sold yet
new["units_sold_today"] = 0
day0 = pd.concat([sat, new], ignore_index=True)

# STEP 3: write the 8 upload files. Outcome columns are removed: tomorrow is unknown.
names = ["day_0_opening_stock.csv"] + [f"day_{i}_sales.csv" for i in range(1, DAYS + 1)]
out = ROOT / "demo"
for i, (date, name) in enumerate(zip(dates, names)):
    day = (day0 if i == 0 else by_day[date]).sort_values(["category", "batch_id"])
    day.drop(columns=OUTCOMES).to_csv(out / name, index=False)
    print(f"{name:26s} {date} {pd.Timestamp(date).day_name():9s} rows {len(day)}")

# STEP 4: answer key = the real outcome of every batch seen in the week
week = test[test["date"].isin(dates)]
key = (week.groupby("batch_id")
       .agg(store_id=("store_id", "first"), category=("category", "first"),
            sub_category=("sub_category", "first"), shelf_life_days=("shelf_life_days", "first"),
            first_date=("date", "min"), last_date_in_week=("date", "max"),
            units_received=("units_received", "first"), units_wasted_final=("units_wasted_final", "first"),
            at_risk=("at_risk", "first"))
       .reset_index())
last_day = test.groupby("batch_id")["date"].max()        # a batch is "finished" if its last evening is inside the week
key["batch_ends_in_week"] = key["batch_id"].map(last_day).isin(dates)
key.to_csv(out / "answer_key_hidden.csv", index=False)
print(f"answer_key_hidden.csv      {len(key)} batches, {int(key['batch_ends_in_week'].sum())} end inside the week")
print(f"Day 0: {len(sat)} batches already on the shelf + {len(new)} newly delivered")
