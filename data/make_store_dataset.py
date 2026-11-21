# ============================================================
# FRESHSAVE - MAKE THE SINGLE-STORE TRAINING DATASET (SIMULATED)
# ------------------------------------------------------------
# WHAT THIS IS: a SIMULATED dataset for ONE supermarket, made to look like real fresh-food data.
#               It is NOT real store data. Every rule used to make it is written below.
# ONE ROW =     one batch of food on one evening of its shelf life (a "daily snapshot").
#               The question each row answers: will this batch end with 20% or more wasted?
# RUN:          python data/make_store_dataset.py
# MAKES:        data/store/store_train_70.csv (70% of rows, the earlier batches)
#               data/store/store_test_30.csv  (30% of rows, the later batches)
# ============================================================


# ============================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

# WHAT WE DID IN STEP 1:
# - Loaded numpy (random numbers) and pandas (tables).



# ============================================================
# STEP 2: SETTINGS (CHANGE A VALUE HERE, NOT IN THE CODE BELOW)
# ------------------------------------------------------------
# - TOTAL_ROWS      : how many rows the dataset has
# - TRAIN_SHARE     : share of rows used for training. The split is by TIME: the earlier batches train,
#                     the later batches test. A batch is never split between the two files.
# - WASTE_THRESHOLD : a batch is "at risk" if this % (or more) of it is wasted
# - SEED            : same number = same dataset every time
# ============================================================

FOLDER = Path(__file__).resolve().parent / "store"
TOTAL_ROWS = 25555
TRAIN_SHARE = 0.70
WASTE_THRESHOLD = 20.0
SEED = 42
ORDER_BIAS = 1.22                # managers order about 22% more than the normal demand over the shelf life.
                                 # CALIBRATION: chosen so that about 42% of batches end up at risk, the share seen in the Kaggle data.
                                 # It was NOT chosen to make any model score higher.

FOLDER.mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)

# WHAT WE DID IN STEP 2:
# - Fixed the size, the split and the seed.



# ============================================================
# STEP 3: THE STORE'S PRODUCT RANGE
# ------------------------------------------------------------
# Same 8 fresh categories as the Kaggle data. 6 products per category = 48 products.
# For each category (typical retail values):
#   life   : shelf life in days when the batch arrives (min, max)
#   demand : normal units sold per day for ONE product (min, max)
#   cost   : cost price per unit (min, max)
#   cycle  : days between deliveries of the same product (min, max)
# ============================================================

CATEGORIES = {
    "Bakery":       dict(life=(3, 5),   demand=(30, 90),  cost=(1.0, 3.0),  cycle=(1, 2)),
    "Beverages":    dict(life=(8, 14),  demand=(30, 100), cost=(0.8, 2.5),  cycle=(4, 5)),
    "Dairy":        dict(life=(6, 12),  demand=(25, 80),  cost=(1.0, 4.0),  cycle=(3, 4)),
    "Deli":         dict(life=(4, 7),   demand=(15, 45),  cost=(3.0, 8.0),  cycle=(2, 3)),
    "Meat":         dict(life=(3, 6),   demand=(15, 40),  cost=(4.0, 10.0), cycle=(2, 3)),
    "Produce":      dict(life=(4, 8),   demand=(25, 80),  cost=(0.8, 3.0),  cycle=(2, 3)),
    "Ready_to_Eat": dict(life=(2, 4),   demand=(15, 50),  cost=(2.0, 5.0),  cycle=(1, 2)),
    "Seafood":      dict(life=(2, 4),   demand=(8, 25),   cost=(6.0, 14.0), cycle=(2, 2)),
}
PRODUCTS_PER_CATEGORY = 6

products = []
for name, c in CATEGORIES.items():
    for _ in range(PRODUCTS_PER_CATEGORY):
        products.append(dict(category=name, life=c["life"], cycle=c["cycle"],
                             demand=int(rng.integers(c["demand"][0], c["demand"][1] + 1)),
                             cost=round(float(rng.uniform(*c["cost"])), 2)))

# WHAT WE DID IN STEP 3:
# - Made 48 products, each with its own normal daily demand and cost price.



# ============================================================
# STEP 4: THE DELIVERY CALENDAR
# ------------------------------------------------------------
# Each product is delivered again and again, every "cycle" days, over about 4 years.
# Batches are put in date order, because the train/test split follows time.
# ============================================================

DAYS = 1500
deliveries = []
for index, p in enumerate(products):
    day = int(rng.integers(0, p["cycle"][1] + 1))
    while day < DAYS:
        deliveries.append((day, index))
        day += int(rng.integers(p["cycle"][0], p["cycle"][1] + 1))
deliveries.sort()

# WHAT WE DID IN STEP 4:
# - Listed every delivery (day number, product) in date order.



# ============================================================
# STEP 5: SIMULATE EACH BATCH DAY BY DAY
# ------------------------------------------------------------
# What happens to one batch (written the way a store works):
#  1. SHELF LIFE: drawn from the category's range (the supplier's date varies).
#  2. HANDLING SCORE (1 to 10): how well the batch was stored and moved. Mean about 7.
#  3. PROMOTION: 20% of batches are promoted, which lifts demand by 20%.
#  4. ORDER SIZE: the manager orders about "normal demand x shelf life x ORDER_BIAS", with ordering error
#     (sometimes too much, sometimes too little). This is the main source of waste.
#  5. DEMAND LEVEL OF THIS BATCH: unknown on delivery day. Some batches simply sell faster or
#     slower than usual (weather, local events). It shows up only through the daily sales.
#  6. EACH DAY: units sold = normal demand x batch demand level x weekday effect x freshness effect,
#     plus random noise, and never more than what is left.
#     Weekday effect: weekends sell more. Freshness effect: sales slow as expiry nears.
#  7. SPOILAGE: poor handling spoils a small share of the stock every day.
#  8. END OF LIFE: whatever is left on the last day is WASTE.
# One row is saved for every evening that still has days left (the expiry evening is not a row).
# ============================================================

WEEKDAY_EFFECT = [0.95, 0.90, 0.90, 0.95, 1.05, 1.30, 1.25]
rows, batch_waste = [], {}
batch_number = 0
for delivery_day, index in deliveries:
    p = products[index]
    life = int(rng.integers(p["life"][0], p["life"][1] + 1))
    handling = int(np.clip(round(rng.normal(7, 1.8)), 1, 10))
    promoted = int(rng.random() < 0.20)
    order_ratio = float(np.clip(rng.normal(ORDER_BIAS, 0.35), 0.4, 1.8))
    received = max(10, int(round(p["demand"] * life * order_ratio)))
    demand_level = float(rng.lognormal(0, 0.30)) * (1.2 if promoted else 1.0)
    spoil_rate = (10 - handling) / 10 * 0.015

    batch_number += 1
    batch_id = f"B{batch_number:05d}"
    stock = received
    snapshots = []
    for d in range(1, life + 1):
        mean_sales = (p["demand"] * demand_level * WEEKDAY_EFFECT[(delivery_day + d) % 7]
                      * (1.05 - 0.25 * (d - 1) / life))
        sold = int(min(stock, max(0, round(rng.normal(mean_sales, 0.2 * mean_sales + 1)))))
        stock -= sold
        stock -= int(rng.binomial(stock, spoil_rate))
        if d < life:
            snapshots.append(dict(batch_id=batch_id, category=p["category"], shelf_life_days=life,
                                  days_left=life - d, units_received=received, units_left=stock,
                                  avg_daily_demand=p["demand"], units_sold_today=sold, cost_price=p["cost"],
                                  handling_score=handling, is_promoted=promoted))
    waste_pct = stock / received * 100
    label = int(waste_pct >= WASTE_THRESHOLD)
    for row in snapshots:
        row["at_risk"] = label
    rows.extend(snapshots)
    batch_waste[batch_id] = waste_pct
    if len(rows) >= TOTAL_ROWS:
        break

data = pd.DataFrame(rows).iloc[:TOTAL_ROWS]       # the last batch may be cut short; it falls in the test part

print("=== SAMPLE ===")
print(data.head(8).to_string(index=False))
print("\nRows:", len(data), "| Batches:", data["batch_id"].nunique(), "| Products:", len(products))

# WHAT WE DID IN STEP 5:
# - Simulated about 5,000 batches and saved one row per evening, then kept exactly 25,555 rows.



# ============================================================
# STEP 6: SPLIT 70% / 30% BY TIME AND SAVE
# ------------------------------------------------------------
# The first batches (earlier dates) go to training, the later ones to testing.
# The cut is made BETWEEN batches, so no batch has rows in both files. If it were cut inside a batch,
# the model would see part of a batch while learning and be tested on the rest, and the score would look too good.
# The cut is at the batch boundary closest to 70%, so the split is very close to 70:30, not exactly.
# ============================================================

rows_per_batch = data.groupby("batch_id", sort=False).size()
cumulative = rows_per_batch.cumsum()
cut_batch = (cumulative - TRAIN_SHARE * len(data)).abs().idxmin()
train_batches = set(rows_per_batch.index[: list(rows_per_batch.index).index(cut_batch) + 1])

train = data[data["batch_id"].isin(train_batches)]
test = data[~data["batch_id"].isin(train_batches)]
assert not set(train["batch_id"]) & set(test["batch_id"])

train.to_csv(FOLDER / "store_train_70.csv", index=False)
test.to_csv(FOLDER / "store_test_30.csv", index=False)

print("\n=== SPLIT ===")
print(f"Train: {len(train)} rows ({len(train)/len(data):.1%}), {train['batch_id'].nunique()} batches, at risk {train['at_risk'].mean():.1%}")
print(f"Test:  {len(test)} rows ({len(test)/len(data):.1%}), {test['batch_id'].nunique()} batches, at risk {test['at_risk'].mean():.1%}")
print("Batch-level at-risk share:", f"{np.mean([w >= WASTE_THRESHOLD for w in batch_waste.values()]):.1%}")
print("Saved in:", FOLDER)

# WHAT WE DID IN STEP 6:
# - Saved the two files. Each has 12 columns: batch_id, 10 inputs and the answer at_risk.
