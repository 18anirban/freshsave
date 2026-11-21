# ============================================================
# FRESHSAVE - MAKE THE RETAIL TRAINING DATASET (SIMULATED, 4 STORES, 2 YEARS)
# ------------------------------------------------------------
# WHAT THIS IS: a SIMULATED dataset for 4 supermarkets, made to look like real fresh-food data.
#               It is NOT real store data. Every rule used to make it is written below.
# ONE ROW =     one batch of food on one evening of its shelf life (a "daily snapshot").
#               The question each row answers: will this batch end with 20% or more wasted?
# RUN:          python data/make_store_dataset.py
# MAKES:        data/store/store_data_full.csv  (all rows, in order of batch delivery)
#               data/store/store_train_70.csv   (70% of rows, the earlier batches)
#               data/store/store_test_30.csv    (30% of rows, the later batches)
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
# - DAYS            : how many days of deliveries (730 = two years)
# - TRAIN_SHARE     : share of rows used for training. The split is by TIME: the earlier batches train,
#                     the later batches test. A batch is never split between the two files.
# - WASTE_THRESHOLD : a batch is "at risk" if this % (or more) of it is wasted
# - ORDER_BIAS      : managers order about this many times the normal demand over the shelf life.
#                     CALIBRATION: chosen so that about 42% of batches end up at risk, the share seen in the real
#                     Kaggle data. It was NOT chosen to make any model score higher.
# - SEED            : same number = same dataset every time
# ============================================================

FOLDER = Path(__file__).resolve().parent / "store"
DAYS = 730
START_DATE = "2023-01-02"          # a Monday (the weekday effect below needs day 0 = Monday)
TRAIN_SHARE = 0.70
WASTE_THRESHOLD = 20.0
ORDER_BIAS = 1.12
SEED = 42

FOLDER.mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)

# WHAT WE DID IN STEP 2:
# - Fixed the period, the split and the seed.



# ============================================================
# STEP 3: THE STORE'S PRODUCT RANGE (CATEGORY -> SUB-CATEGORY)
# ------------------------------------------------------------
# 8 fresh categories as in the Kaggle data, each with real product types (sub-categories): 49 types in total.
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

MARKUP = {"Bakery": 2.2, "Beverages": 1.8, "Dairy": 1.45, "Deli": 1.7, "Meat": 1.45, "Produce": 1.9,
          "Ready_to_Eat": 2.0, "Seafood": 1.5}                 # selling price = cost price x markup
SEASON_AMP = {"Bakery": 0.03, "Beverages": 0.25, "Dairy": 0.0, "Deli": 0.04, "Meat": 0.06, "Produce": 0.12,
              "Ready_to_Eat": 0.08, "Seafood": 0.05}           # summer sells more drinks, salads and barbecue food

# WHAT WE DID IN STEP 3:
# - Listed 49 real product types under the 8 categories. Each has its own shelf life, demand range and price range.
# - Added the markup that turns cost price into selling price, and how much each category moves with the seasons.



# ============================================================
# STEP 3B: THE FOUR STORES
# ------------------------------------------------------------
# The stores differ in size (how much they sell), range (how many product types they stock),
# handling quality (storage and delivery), ordering accuracy and price level.
# ============================================================

STORES = [
    dict(id="STORE_01", size=1.40, types=24, handling=7.6, order=0.00,  price=1.03),   # large urban store
    dict(id="STORE_02", size=1.00, types=20, handling=7.0, order=0.00,  price=1.00),   # medium store
    dict(id="STORE_03", size=0.70, types=16, handling=6.4, order=+0.06, price=0.98),   # small town store
    dict(id="STORE_04", size=1.10, types=20, handling=6.8, order=+0.10, price=1.00),   # suburban store, orders loosely
]
CATEGORY_NAMES = list(CATEGORY_SETTINGS)

products = []   # one entry per (store, product type)
for store_index, store in enumerate(STORES):
    chosen = []
    for category in CATEGORY_NAMES:                        # every store stocks at least 2 types of each category
        options = [i for i, row in enumerate(SUB_CATEGORIES) if row[0] == category]
        chosen += list(rng.choice(options, size=2, replace=False))
    rest = [i for i in range(len(SUB_CATEGORIES)) if i not in chosen]
    chosen += list(rng.choice(rest, size=store["types"] - len(chosen), replace=False))
    for i in sorted(chosen):
        category, sub, life, demand, cost = SUB_CATEGORIES[i]
        c = CATEGORY_SETTINGS[category]
        products.append(dict(store=store, category=category, sub=sub, life=life, cycle=c["cycle"],
                             order=c["order"] + store["order"], spoil=c["spoil"],
                             demand=max(3, int(round(rng.integers(demand[0], demand[1] + 1) * store["size"] * rng.uniform(0.85, 1.15)))),
                             cost=round(float(rng.uniform(*cost)) * rng.uniform(0.97, 1.03), 2),
                             markup=MARKUP[category], season=SEASON_AMP[category]))

# WHAT WE DID IN STEP 3B:
# - Gave each store its own range of product types and its own demand and cost for each type.



# ============================================================
# STEP 4: THE DELIVERY CALENDAR
# ------------------------------------------------------------
# Each product in each store is delivered again and again, every "cycle" days, for DAYS days.
# Batches are put in date order, because the train/test split follows time.
# ============================================================

deliveries = []
for index, p in enumerate(products):
    day = int(rng.integers(0, p["cycle"][1] + 1))
    while day < DAYS:
        deliveries.append((day, index))
        day += int(rng.integers(p["cycle"][0], p["cycle"][1] + 1))
deliveries.sort()

# WHAT WE DID IN STEP 4:
# - Listed every delivery (day number, store and product) in date order.



# ============================================================
# STEP 5: SIMULATE EACH BATCH DAY BY DAY
# ------------------------------------------------------------
# What happens to one batch (written the way a store works):
#  1. SHELF LIFE: drawn from the product's range. In 6% of batches the delivery is late, so the batch
#     arrives with 1 or 2 days less than the manager planned for.
#  2. HANDLING SCORE (1 to 10): how well the batch was stored and moved (the store's average +/- noise).
#     In 2% of batches the score was not recorded (a missing value).
#  3. PROMOTION: 20% of batches are promoted: demand +20% and a 15% lower selling price.
#  4. ORDER SIZE: the manager orders about "normal demand x planned shelf life x (ORDER_BIAS + habit)",
#     with ordering error. This is the main source of waste.
#  5. DEMAND LEVEL OF THIS BATCH: unknown on delivery day. Some batches sell faster or slower than usual.
#     In 5% of batches a shock (bad weather, a competitor's sale) cuts demand by 30% to 50%.
#  6. EACH DAY: units sold = normal demand x batch demand level x weekday effect x freshness effect x season,
#     plus random noise, and never more than what is left. Weekends sell more, sales slow near expiry,
#     drinks and salads sell more in summer.
#  7. SPOILAGE: poor handling spoils a small share of the stock every day.
#  8. END OF LIFE: whatever is left on the last day is WASTE (units_wasted_final, waste_cost).
#  9. STOCK COUNT ERRORS: in 4% of evenings the recorded units_left is a little wrong (a miscount).
# One row is saved for every evening that still has days left (the expiry evening is not a row).
# ============================================================

WEEKDAY_EFFECT = [0.95, 0.90, 0.90, 0.95, 1.05, 1.30, 1.25]
rows, batch_waste = [], {}
batch_number = 0
for delivery_day, index in deliveries:
    p = products[index]
    store = p["store"]
    planned_life = int(rng.integers(p["life"][0], p["life"][1] + 1))
    life = max(2, planned_life - int(rng.integers(1, 3))) if rng.random() < 0.06 else planned_life
    handling = int(np.clip(round(rng.normal(store["handling"], 1.8)), 1, 10))
    handling_recorded = np.nan if rng.random() < 0.02 else handling
    promoted = int(rng.random() < 0.20)
    order_ratio = float(np.clip(rng.normal(ORDER_BIAS + p["order"], 0.35), 0.4, 1.8))
    received = max(10, int(round(p["demand"] * planned_life * order_ratio)))
    demand_level = float(rng.lognormal(0, 0.30)) * (1.2 if promoted else 1.0)
    if rng.random() < 0.05:
        demand_level *= float(rng.uniform(0.5, 0.7))
    batch_price_factor = float(rng.uniform(0.95, 1.05))       # supplier price varies a little from batch to batch
    cost_price = round(p["cost"] * batch_price_factor, 2)
    selling_price = round(cost_price * p["markup"] * store["price"] * (0.85 if promoted else 1.0), 2)
    spoil_rate = (10 - handling) / 10 * 0.015 * p["spoil"]

    batch_number += 1
    batch_id = f"B{batch_number:05d}"
    stock = received
    snapshots = []
    for d in range(1, life + 1):
        date = pd.Timestamp(START_DATE) + pd.Timedelta(days=delivery_day + d)
        season = 1 + p["season"] * np.sin(2 * np.pi * (date.dayofyear - 80) / 365)
        mean_sales = (p["demand"] * demand_level * WEEKDAY_EFFECT[(delivery_day + d) % 7]
                      * (1.05 - 0.25 * (d - 1) / life) * season)
        sold = int(min(stock, max(0, round(rng.normal(mean_sales, 0.2 * mean_sales + 1)))))
        stock -= sold
        stock -= int(rng.binomial(stock, spoil_rate))
        if d < life:
            counted = stock
            if rng.random() < 0.04:                           # a stock-count error
                counted = max(0, int(round(stock * (1 + rng.normal(0, 0.10)))))
            snapshots.append(dict(batch_id=batch_id, store_id=store["id"], date=date.strftime("%Y-%m-%d"),
                                  category=p["category"], sub_category=p["sub"], shelf_life_days=life,
                                  days_left=life - d, units_received=received, units_left=counted,
                                  avg_daily_demand=p["demand"], units_sold_today=sold, cost_price=cost_price,
                                  selling_price=selling_price, handling_score=handling_recorded, is_promoted=promoted))
    waste_pct = stock / received * 100
    for row in snapshots:
        row["units_wasted_final"] = stock
        row["waste_cost"] = round(stock * cost_price, 2)
        row["at_risk"] = int(waste_pct >= WASTE_THRESHOLD)
    rows.extend(snapshots)
    batch_waste[batch_id] = waste_pct

data = pd.DataFrame(rows)
data["handling_score"] = data["handling_score"].astype("Int64")

print("=== SAMPLE ===")
print(data.head(6).to_string(index=False))
print("\nRows:", len(data), "| Batches:", data["batch_id"].nunique(), "| Stores:", data["store_id"].nunique(),
      "| Product types:", data["sub_category"].nunique(), "| Dates:", data["date"].min(), "to", data["date"].max())

# WHAT WE DID IN STEP 5:
# - Simulated every batch of every store for two years and saved one row per evening.



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

data.to_csv(FOLDER / "store_data_full.csv", index=False)
train.to_csv(FOLDER / "store_train_70.csv", index=False)
test.to_csv(FOLDER / "store_test_30.csv", index=False)

print("\n=== SPLIT ===")
print(f"Train: {len(train)} rows ({len(train)/len(data):.1%}), {train['batch_id'].nunique()} batches, at risk {train['at_risk'].mean():.1%}")
print(f"Test:  {len(test)} rows ({len(test)/len(data):.1%}), {test['batch_id'].nunique()} batches, at risk {test['at_risk'].mean():.1%}")
print("Batch-level at-risk share:", f"{np.mean([w >= WASTE_THRESHOLD for w in batch_waste.values()]):.1%}")
print("Saved in:", FOLDER)

# WHAT WE DID IN STEP 6:
# - Saved the full file and the two split files. Each has 18 columns.
# - units_wasted_final, waste_cost and at_risk are OUTCOMES of the batch. Never use them as model inputs.
