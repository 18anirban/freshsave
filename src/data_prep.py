# ====================================================================
# SHARED DATA PREPARATION  (used by KNN, Decision Tree and Comparison)
# One place for: loading, features, missing values, skewness, profile.
# ====================================================================
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "store"
MODEL_DIR = ROOT / "models"

TARGET = "at_risk"
OUTCOME_COLS = ["units_wasted_final", "waste_cost", "at_risk"]   # never inputs
ID_COLS = ["batch_id", "store_id", "date", "sub_category"]        # never inputs
RAW_NUMERIC = ["shelf_life_days", "days_left", "units_received", "units_left",
               "avg_daily_demand", "units_sold_today", "cost_price",
               "selling_price", "handling_score", "is_promoted"]
DERIVED = ["sold_so_far", "cover_gap", "margin_pct"]
CATEGORIES = ["Bakery", "Beverages", "Dairy", "Deli", "Meat", "Produce",
              "Ready_to_Eat", "Seafood"]

COLUMN_INFO = [
    ("batch_id", "ID", "One delivery of one product. Same on every daily row of that batch."),
    ("store_id", "ID", "Which of the 4 stores."),
    ("date", "ID", "The evening this snapshot was taken."),
    ("category", "Input (one-hot)", "Product family, e.g. Dairy, Meat."),
    ("sub_category", "ID (not used)", "Product type, e.g. Fresh Milk. Too many values; category is enough."),
    ("shelf_life_days", "Input", "Total days the product can be sold."),
    ("days_left", "Input", "Days left before expiry on this evening."),
    ("units_received", "Input", "Units delivered in this batch."),
    ("units_left", "Input", "Units still on the shelf this evening."),
    ("avg_daily_demand", "Input", "Normal units sold per day for this product."),
    ("units_sold_today", "Input", "Units sold on this day."),
    ("cost_price", "Input", "What the store paid per unit."),
    ("selling_price", "Input", "Shelf price per unit."),
    ("handling_score", "Input (2% missing)", "Cold-chain / handling quality score."),
    ("is_promoted", "Input", "1 if on promotion."),
    ("units_wasted_final", "OUTCOME (blocked)", "Units thrown away at end of life - the future, never an input."),
    ("waste_cost", "OUTCOME (blocked)", "Money lost to waste - the future, never an input."),
    ("at_risk", "TARGET", "1 if 20% or more of the batch ends up wasted."),
]


# ---------------- STEP 1: load train / test ----------------
def load_data():
    train = pd.read_csv(DATA_DIR / "store_train_70.csv")
    test = pd.read_csv(DATA_DIR / "store_test_30.csv")
    return train, test


# ---------------- STEP 2: build features ----------------
def build_features(df):
    """Raw columns + 3 derived columns + one-hot category. Outcomes excluded."""
    out = df[RAW_NUMERIC].astype(float).copy()
    out["sold_so_far"] = df["units_received"] - df["units_left"]
    out["cover_gap"] = df["units_left"] / df["avg_daily_demand"] - df["days_left"]
    out["margin_pct"] = (df["selling_price"] - df["cost_price"]) / df["selling_price"]
    for c in CATEGORIES:
        out["cat_" + c] = (df["category"] == c).astype(float)
    return out


FEATURE_NAMES = RAW_NUMERIC + DERIVED + ["cat_" + c for c in CATEGORIES]
NUMERIC_COLS = RAW_NUMERIC + DERIVED


# ---------------- STEP 3: missing values (fit on TRAIN only) ----------------
def fit_imputer(X_train):
    return SimpleImputer(strategy="median").fit(X_train)


def apply_imputer(imputer, X):
    return pd.DataFrame(imputer.transform(X), columns=X.columns, index=X.index)


# ---------------- STEP 4: skewness ----------------
def skewness_table(X_train, cols=None):
    cols = cols or NUMERIC_COLS
    rows = []
    for c in cols:
        s = float(X_train[c].skew())
        nonneg = bool((X_train[c] >= 0).all())
        rows.append({"column": c, "skewness": round(s, 2),
                     "skewed (|skew|>1)": abs(s) > 1,
                     "log1p possible": nonneg,
                     "skew after log1p": round(float(np.log1p(X_train[c]).skew()), 2) if nonneg else None})
    return pd.DataFrame(rows).sort_values("skewness", key=abs, ascending=False).reset_index(drop=True)


def skewed_columns(X_train):
    t = skewness_table(X_train)
    ok = t["skewed (|skew|>1)"] & t["log1p possible"]
    ok &= t["skew after log1p"].abs() < t["skewness"].abs()   # only if it really helps
    ok &= t["column"] != "is_promoted"                        # 0/1 column: leave alone
    return t[ok]["column"].tolist()


# ---------------- STEP 5: data profile for the app ----------------
def profile_data(train, test):
    full = pd.concat([train, test], ignore_index=True)
    b_tr, b_te = train.drop_duplicates("batch_id"), test.drop_duplicates("batch_id")
    return {
        "rows_total": len(full), "rows_train": len(train), "rows_test": len(test),
        "batches_total": full["batch_id"].nunique(),
        "batches_train": b_tr["batch_id"].nunique(), "batches_test": b_te["batch_id"].nunique(),
        "stores": full["store_id"].nunique(), "categories": full["category"].nunique(),
        "sub_categories": full["sub_category"].nunique(),
        "train_dates": (train["date"].min(), train["date"].max()),
        "test_dates": (test["date"].min(), test["date"].max()),
        "risk_rate_train": float(train[TARGET].mean()), "risk_rate_test": float(test[TARGET].mean()),
        "risk_rate_batch_train": float(b_tr[TARGET].mean()), "risk_rate_batch_test": float(b_te[TARGET].mean()),
        "batch_overlap": len(set(train["batch_id"]) & set(test["batch_id"])),
        "missing": {c: int(full[c].isna().sum()) for c in full.columns if full[c].isna().any()},
        "waste_cost_total": float(b_tr["waste_cost"].sum() + b_te["waste_cost"].sum()),
        "by_category": full.groupby("category")[TARGET].mean().round(3).to_dict(),
        "by_store": full.groupby("store_id")[TARGET].mean().round(3).to_dict(),
    }
