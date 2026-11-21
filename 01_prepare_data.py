# ============================================================
# FRESHSAVE - FILE 1 OF 5: PREPARE THE DATA
# ------------------------------------------------------------
# RETAIL PROBLEM:
#   A supermarket throws away fresh food (milk, meat, bakery...) that expires before it sells.
#   Question this project answers: WHEN A BATCH ARRIVES, WILL 20% OR MORE OF IT BE WASTED?
#   If yes, the store can order less, promote it early or mark it down in time.
#
# THIS FILE: reads the Kaggle history file, builds the answer column (the label), builds the input
#            columns (X), splits the data by date and saves it for the next files.
# RUN:       python 01_prepare_data.py
# ============================================================


# ============================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ------------------------------------------------------------
# This block loads all tools needed for:
# - Reading and changing the data (pandas, numpy)
# - Finding folders on any computer (pathlib)
# - Saving the settings for the next files (json)
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd

# WHAT WE DID IN STEP 1:
# - Loaded the libraries. Nothing is calculated yet.



# ============================================================
# STEP 2: SETTINGS (CHANGE A VALUE HERE, NOT IN THE CODE BELOW)
# ------------------------------------------------------------
# - RAW_FILE         : the Kaggle file "Managing Perishable Inventory Data"
# - WASTE_THRESHOLD  : a batch is "at risk" if this % (or more) of it is wasted
# - SPLIT_DATE       : batches before this date are used to TEACH the models,
#                      batches on or after it are used to TEST them
# - SKIP_CATEGORIES  : these have a shelf life of months, a different problem
# ============================================================

FOLDER = Path(__file__).resolve().parent
RAW_FILE = FOLDER / "data" / "raw" / "perishable_goods_management.csv"
OUTPUT = FOLDER / "output"
WASTE_THRESHOLD = 20.0
SPLIT_DATE = "2024-01-01"
SKIP_CATEGORIES = ["Pharmaceuticals", "Frozen_Meals"]

OUTPUT.mkdir(exist_ok=True)

# WHAT WE DID IN STEP 2:
# - Put every setting in one place and made sure the output folder exists.



# ============================================================
# STEP 3: LOAD THE KAGGLE DATA
# ------------------------------------------------------------
# Each row is ONE BATCH of one product delivered to one store:
# how much arrived, how much sold, price, storage conditions, and so on.
# ============================================================

if not RAW_FILE.exists():
    raise SystemExit(f"Kaggle file not found: {RAW_FILE}\n"
                     "Put perishable_goods_management.csv in the data/raw folder and run again.")

raw = pd.read_csv(RAW_FILE)

print("=== SAMPLE RAW DATA ===")
print(raw.head())
print("\nRows:", len(raw), "| Columns:", raw.shape[1], "| Stores:", raw["store_id"].nunique())

# WHAT WE DID IN STEP 3:
# - Read the file and printed the first rows and its size.



# ============================================================
# STEP 4: KEEP FRESH FOOD ONLY
# ------------------------------------------------------------
# Medicines and frozen meals last for months, so waste there is a different problem.
# ============================================================

df = raw[~raw["category"].isin(SKIP_CATEGORIES)].copy()

print("\n=== FRESH-FOOD CATEGORIES KEPT ===")
print(df["category"].value_counts())

# WHAT WE DID IN STEP 4:
# - Removed the categories in SKIP_CATEGORIES. What is left is fresh food.



# ============================================================
# STEP 5: CREATE THE ANSWER COLUMN (THE LABEL)
# ------------------------------------------------------------
# This block works out, for every batch, how much was wasted:
#     units wasted = units received - units sold
#     waste %      = units wasted / units received
# Then it turns that into the class the models must learn:
# - 1 = AT RISK (20% or more of the batch was wasted)
# - 0 = SOLD IN TIME
# ============================================================

df["units_received"] = df["initial_quantity"].clip(lower=1)
df["units_sold"] = np.minimum(df["units_sold"], df["units_received"])
df["units_wasted"] = df["units_received"] - df["units_sold"]
df["waste_pct"] = df["units_wasted"] / df["units_received"] * 100
df["waste_cost"] = df["units_wasted"] * df["cost_price"]
df["at_risk"] = (df["waste_pct"] >= WASTE_THRESHOLD).astype(int)

print("\n=== AT RISK DISTRIBUTION (1 = at risk, 0 = sold in time) ===")
print(df["at_risk"].value_counts())
print("Share at risk:", round(df["at_risk"].mean() * 100, 1), "%")

# WHAT WE DID IN STEP 5:
# - Added units_wasted, waste_pct, waste_cost and the label at_risk.
# - These are RESULTS of the batch. They are used to make the label and for the BI dashboard,
#   but they are NEVER given to the models as inputs (that would be cheating).



# ============================================================
# STEP 6: CREATE NEW INPUT COLUMNS FROM WHAT WE KNOW WHEN THE BATCH ARRIVES
# ------------------------------------------------------------
# A manager can see two things on the day of delivery: how much stock arrived and how fast it normally sells.
# Putting them together gives the most useful clue:
#     stock_cover_days = units received / normal sales per day   (how many days the stock would last)
#     cover_gap_days   = stock_cover_days - shelf_life_days      (positive = more stock than can sell in time)
# ============================================================

df["shelf_life_days"] = df["days_remaining_at_purchase"]
df["stock_cover_days"] = (df["units_received"] / df["daily_demand"].clip(lower=1)).clip(upper=60)
df["cover_gap_days"] = df["stock_cover_days"] - df["shelf_life_days"]
df["margin_pct"] = (df["base_price"] - df["cost_price"]) / df["base_price"].clip(lower=0.01) * 100

print("\n=== NEW COLUMNS (first 5 rows) ===")
print(df[["category", "units_received", "daily_demand", "shelf_life_days",
          "stock_cover_days", "cover_gap_days", "at_risk"]].head())

# WHAT WE DID IN STEP 6:
# - Added stock_cover_days, cover_gap_days and margin_pct.
# - cover_gap_days is the key idea: if the stock lasts longer than the shelf life, some of it will be wasted.



# ============================================================
# STEP 7: DEFINE FEATURES (X) AND TARGET (y)
# ------------------------------------------------------------
# X = the columns used for prediction (known when the batch arrives).
# y = at_risk.
# Columns NOT in X, and why:
# - units_sold, units_wasted, waste_pct, waste_cost: known only AFTER the batch is finished (would leak the answer)
# - spoilage_risk, quality_grade, markdown_applied, discount_pct, selling_price, days_until_expiry:
#   scores or decisions that exist after the fact, or hide the answer
# - record_id, product_id, supplier_id, region: only names or numbers for identification
# ============================================================

NUMERIC_FEATURES = ["shelf_life_days", "units_received", "daily_demand", "demand_variability",
                    "stock_cover_days", "cover_gap_days", "base_price", "cost_price", "margin_pct",
                    "spoilage_sensitivity", "storage_temp", "temp_deviation", "temp_abuse_events",
                    "distribution_hours", "handling_score", "packaging_score", "supplier_score", "is_promoted"]
CATEGORY_FEATURES = ["category"]
FEATURES = NUMERIC_FEATURES + CATEGORY_FEATURES

# columns kept only to describe a batch in the BI dashboard (never used as model input)
INFO_COLUMNS = ["record_id", "store_id", "product_name", "transaction_date",
                "units_sold", "units_wasted", "waste_pct", "waste_cost"]

print("\n=== FEATURES (X) ===")
print(FEATURES)
print("\nNumber of features:", len(FEATURES))

# WHAT WE DID IN STEP 7:
# - Listed the inputs for the models. 18 numbers plus the category (text).



# ============================================================
# STEP 8: TRAIN/TEST SPLIT BY DATE
# ------------------------------------------------------------
# - TRAIN = batches before 2024-01-01 (the models learn from the past)
# - TEST  = batches from 2024-01-01 (the models are judged on the "future")
# WHY NOT A RANDOM SPLIT? A random split mixes the same months, stores and products into both sets,
# so the score looks better than it would be in real life. Real life = predict tomorrow from yesterday.
# ============================================================

df["transaction_date"] = pd.to_datetime(df["transaction_date"])
train = df[df["transaction_date"] < SPLIT_DATE]
test = df[df["transaction_date"] >= SPLIT_DATE]

print("\n=== TRAIN/TEST SIZES ===")
print("Training batches (before", SPLIT_DATE + "):", len(train), "| at risk:", round(train["at_risk"].mean() * 100, 1), "%")
print("Testing batches  (from", SPLIT_DATE + "):  ", len(test), "| at risk:", round(test["at_risk"].mean() * 100, 1), "%")

if len(train) == 0 or len(test) == 0:
    raise SystemExit("The split date leaves no data on one side. Change SPLIT_DATE in STEP 2.")

# WHAT WE DID IN STEP 8:
# - Split the batches into a past set (to learn) and a later set (to test).



# ============================================================
# STEP 9: SAVE THE PREPARED DATA FOR THE NEXT FILES
# ------------------------------------------------------------
# Files 2 to 5 read these files, so they do not need to repeat the work above.
# ============================================================

columns = INFO_COLUMNS + FEATURES + ["at_risk"]
train[columns].to_csv(OUTPUT / "train.csv", index=False)
test[columns].to_csv(OUTPUT / "test.csv", index=False)
json.dump({"numeric": NUMERIC_FEATURES, "category": CATEGORY_FEATURES, "target": "at_risk",
           "waste_threshold": WASTE_THRESHOLD, "split_date": SPLIT_DATE},
          open(OUTPUT / "settings.json", "w"), indent=2)

print("\n=== SAVED ===")
print("output/train.csv, output/test.csv, output/settings.json")
print("\nNext: python 02_knn_model.py")

# WHAT WE DID IN STEP 9:
# - Saved train.csv, test.csv and the list of features in the output folder.
