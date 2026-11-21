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
# STEP 3: THE STORE'S PRODUCT RANGE (CATEGORY -> SUB-CATEGORY)
# ------------------------------------------------------------
# 8 fresh categories as in the Kaggle data, each with real product types (sub-categories) = 49 products.
# For each sub-category (typical retail values, one product type per line):
#   life   : shelf life in days when the batch arrives (min, max)
#   demand : normal units sold per day (min, max). One value is drawn per product and then kept.
#   cost   : cost price per unit (min, max). One value is drawn per product; each batch varies by about 5%.
# CATEGORY SETTINGS:
#   cycle   : days between deliveries of the same product (min, max)
#   order   : ordering habit, added to ORDER_BIAS. Bakery and ready meals are over-ordered "to be safe",
#             seafood and beverages are ordered tightly.
#   spoil   : how fast poor handling spoils the stock (1.0 = average)
# ============================================================

CATEGORY_SETTINGS = {
    "Bakery":       dict(cycle=(1, 2), order=+0.10, spoil=0.8),
    "Beverages":    dict(cycle=(4, 5), order=-0.10, spoil=0.5),
    "Dairy":        dict(cycle=(3, 4), order=-0.05, spoil=0.8),
    "Deli":         dict(cycle=(2, 3), order=0.00,  spoil=1.0),
    "Meat":         dict(cycle=(2, 3), order=-0.05, spoil=1.2),
    "Produce":      dict(cycle=(2, 3), order=+0.05, spoil=1.2),
    "Ready_to_Eat": dict(cycle=(1, 2), order=+0.10, spoil=1.0),
    "Seafood":      dict(cycle=(2, 2), order=-0.15, spoil=1.5),
}

SUB_CATEGORIES = [   # (category, sub-category, life, demand, cost)
    ("Bakery", "White Bread Loaf", (3, 5), (60, 90), (0.9, 1.4)),
    ("Bakery", "Whole Wheat Bread", (3, 5), (40, 65), (1.2, 1.8)),
    ("Bakery", "Croissants", (2, 3), (25, 50), (0.7, 1.1)),
    ("Bakery", "Bagels", (4, 6), (20, 40), (0.6, 0.9)),
    ("Bakery", "Muffins", (3, 5), (20, 45), (0.8, 1.3)),
    ("Bakery", "Celebration Cakes", (3, 5), (8, 20), (4.0, 8.0)),
    ("Beverages", "Fresh Orange Juice", (8, 12), (40, 80), (1.4, 2.2)),
    ("Beverages", "Fresh Apple Juice", (9, 14), (25, 50), (1.3, 2.0)),
    ("Beverages", "Fruit Smoothies", (4, 7), (20, 45), (1.8, 3.0)),
    ("Beverages", "Cold Brew Coffee", (7, 10), (15, 35), (1.5, 2.5)),
    ("Beverages", "Flavoured Milk", (8, 12), (30, 60), (0.9, 1.4)),
    ("Beverages", "Coconut Water", (9, 14), (20, 40), (1.2, 1.9)),
    ("Dairy", "Whole Milk", (7, 10), (70, 110), (0.8, 1.1)),
    ("Dairy", "Skimmed Milk", (7, 10), (40, 70), (0.8, 1.1)),
    ("Dairy", "Plain Yogurt", (10, 14), (40, 80), (0.5, 0.9)),
    ("Dairy", "Greek Yogurt", (10, 14), (30, 55), (0.9, 1.5)),
    ("Dairy", "Cheddar Cheese", (10, 14), (20, 40), (2.5, 4.0)),
    ("Dairy", "Butter", (12, 14), (15, 30), (2.0, 3.0)),
    ("Dairy", "Fresh Cream", (6, 9), (10, 25), (1.2, 2.0)),
    ("Dairy", "Paneer", (6, 9), (15, 35), (2.0, 3.5)),
    ("Deli", "Sliced Ham", (5, 8), (15, 35), (3.5, 5.5)),
    ("Deli", "Roast Chicken", (3, 4), (12, 25), (4.0, 6.0)),
    ("Deli", "Salami", (7, 10), (10, 22), (3.0, 5.0)),
    ("Deli", "Hummus", (5, 8), (15, 30), (1.5, 2.5)),
    ("Deli", "Potato Salad", (4, 6), (12, 28), (1.2, 2.0)),
    ("Deli", "Fresh Pasta", (5, 8), (10, 22), (1.8, 3.0)),
    ("Meat", "Chicken Breast", (3, 5), (30, 60), (3.0, 4.5)),
    ("Meat", "Minced Beef", (3, 4), (20, 40), (3.5, 5.5)),
    ("Meat", "Pork Chops", (3, 5), (12, 25), (3.5, 5.0)),
    ("Meat", "Lamb Cuts", (3, 5), (8, 18), (6.0, 9.0)),
    ("Meat", "Pork Sausages", (5, 7), (18, 35), (2.0, 3.5)),
    ("Produce", "Bananas", (4, 7), (60, 100), (0.4, 0.7)),
    ("Produce", "Tomatoes", (5, 8), (40, 75), (0.9, 1.5)),
    ("Produce", "Baby Spinach", (4, 6), (20, 45), (1.0, 1.8)),
    ("Produce", "Strawberries", (3, 5), (20, 45), (1.8, 3.0)),
    ("Produce", "Apples", (10, 14), (40, 80), (0.7, 1.2)),
    ("Produce", "Lettuce", (4, 7), (25, 50), (0.7, 1.2)),
    ("Produce", "Mushrooms", (4, 6), (15, 30), (1.5, 2.5)),
    ("Ready_to_Eat", "Sandwiches", (2, 3), (30, 60), (1.2, 2.0)),
    ("Ready_to_Eat", "Sushi Packs", (2, 2), (12, 28), (2.5, 4.0)),
    ("Ready_to_Eat", "Salad Bowls", (2, 4), (20, 40), (2.0, 3.2)),
    ("Ready_to_Eat", "Ready Meals", (4, 6), (18, 35), (2.5, 4.0)),
    ("Ready_to_Eat", "Wraps", (2, 4), (20, 40), (1.3, 2.2)),
    ("Ready_to_Eat", "Soup Pots", (4, 6), (10, 25), (1.5, 2.5)),
    ("Seafood", "Salmon Fillet", (2, 4), (10, 22), (7.0, 11.0)),
    ("Seafood", "Prawns", (2, 3), (8, 18), (6.0, 10.0)),
    ("Seafood", "Cod Fillet", (2, 4), (8, 16), (6.0, 9.0)),
    ("Seafood", "Tuna Steak", (2, 3), (6, 14), (8.0, 13.0)),
    ("Seafood", "Whole Mackerel", (2, 3), (6, 14), (3.0, 5.0)),
]

products = []
for category, sub, life, demand, cost in SUB_CATEGORIES:
    c = CATEGORY_SETTINGS[category]
    products.append(dict(category=category, sub=sub, life=life, cycle=c["cycle"], order=c["order"], spoil=c["spoil"],
                         demand=int(rng.integers(demand[0], demand[1] + 1)),
                         cost=round(float(rng.uniform(*cost)), 2)))

# WHAT WE DID IN STEP 3:
# - Listed 49 real product types under the 8 categories. Each has its own shelf life, demand, price and delivery rhythm.



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
#  4. ORDER SIZE: the manager orders about "normal demand x shelf life x (ORDER_BIAS + category habit)", with ordering error
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
    order_ratio = float(np.clip(rng.normal(ORDER_BIAS + p["order"], 0.35), 0.4, 1.8))
    received = max(10, int(round(p["demand"] * life * order_ratio)))
    demand_level = float(rng.lognormal(0, 0.30)) * (1.2 if promoted else 1.0)
    batch_price_factor = float(rng.uniform(0.95, 1.05))       # supplier price varies a little from batch to batch
    spoil_rate = (10 - handling) / 10 * 0.015 * p["spoil"]

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
            snapshots.append(dict(batch_id=batch_id, category=p["category"], sub_category=p["sub"], shelf_life_days=life,
                                  days_left=life - d, units_received=received, units_left=stock,
                                  avg_daily_demand=p["demand"], units_sold_today=sold, cost_price=round(p["cost"] * batch_price_factor, 2),
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
# - Saved the two files. Each has 13 columns: batch_id, category, sub_category, 8 other inputs and the answer at_risk.
