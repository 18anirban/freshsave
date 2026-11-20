"""
features.py - data preparation + feature engineering (shared by training, daily files and the app)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

CATEGORIES = ["Bakery", "Beverages", "Dairy", "Deli", "Meat", "Produce", "Ready_to_Eat", "Seafood"]

# Columns only known AFTER a batch is finished -> NEVER used as model inputs (would be cheating).
OUTCOME_COLS = ["units_sold", "units_wasted", "waste_pct", "revenue", "profit", "waste_cost",
                "was_spoiled", "profit_margin_pct"]
EXCLUDED = {   # shown in Tab 4 "Preparation"
    **{c: "Outcome - only known after the batch is finished (would leak the answer)" for c in OUTCOME_COLS},
    "spoilage_risk": "A risk score already inside the dataset (the model would copy it)",
    "quality_grade": "Assigned after the fact",
    "markdown_applied": "A decision, not an input", "discount_pct": "A decision, not an input",
    "selling_price": "Depends on the discount decision",
    "days_until_expiry": "Replaced by our simulated daily days_left",
    "shelf_life_days": "Replaced by the compressed demo shelf life",
    "days_remaining_at_purchase": "Only used to build the demo shelf life",
    "record_id / product_id / supplier_id / region": "Identifiers, not predictive",
}
# Static facts about a batch that are known on arrival (copied into every daily file)
STATIC_COLS = ["daily_demand", "demand_variability", "base_price", "cost_price", "spoilage_sensitivity",
               "storage_temp", "temp_deviation", "temp_abuse_events", "distribution_hours",
               "handling_score", "packaging_score", "supplier_score", "is_promoted"]
# Right-skewed columns -> log1p is applied for BOTH models
LOG_COLS = ["daily_demand", "qty_received", "qty_left", "units_sold_total", "observed_speed",
            "base_price", "cost_price"]


def load_raw(path=None) -> pd.DataFrame:
    return pd.read_csv(path or C.RAW_CSV)


def prepare_batches(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per batch: fresh categories only, with the label and the compressed shelf life."""
    df = raw[~raw["category"].isin(C.DROP_CATEGORIES)].copy()
    q = df["initial_quantity"].clip(lower=1).astype(int)
    s = np.minimum(df["units_sold"].astype(int), q)
    b = pd.DataFrame({
        "batch_id": "B" + df["record_id"].astype(str).str.zfill(6),
        "store_id": df["store_id"], "category": df["category"], "product_name": df["product_name"],
        "transaction_date": pd.to_datetime(df["transaction_date"]),
        "qty_received": q, "units_sold_final": s,
    })
    for c in STATIC_COLS:
        b[c] = df[c].values
    b["life"] = np.clip(df["days_remaining_at_purchase"].astype(int), C.LIFE_MIN, C.LIFE_MAX).values
    b["final_waste_pct"] = (q - s) / q * 100
    b["at_risk"] = (b["final_waste_pct"] >= C.WASTE_AT_RISK_PCT).astype(int)   # THE LABEL
    b["final_waste_cost"] = (q - s) * b["cost_price"]
    return b.reset_index(drop=True)


def add_engineered(s: pd.DataFrame) -> pd.DataFrame:
    """Evening features. Needs: selling_days_elapsed, days_left, qty_received, qty_left, units_sold_total."""
    s = s.copy()
    k = s["selling_days_elapsed"].clip(lower=1)
    s["observed_speed"] = s["units_sold_total"] / k                                   # units/day so far
    s["cover_expected"] = (s["qty_left"] / s["daily_demand"].clip(lower=1)).clip(upper=50)
    s["cover_observed"] = (s["qty_left"] / s["observed_speed"].clip(lower=0.5)).clip(upper=50)
    s["gap_expected"] = (s["cover_expected"] - s["days_left"]).clip(-20, 50)
    s["gap_observed"] = (s["cover_observed"] - s["days_left"]).clip(-20, 50)          # days needed minus days left
    s["sell_through"] = s["units_sold_total"] / s["qty_received"].clip(lower=1)
    required = s["qty_left"] / s["days_left"].clip(lower=1)                           # units/day needed to clear it
    s["speed_ratio"] = (s["observed_speed"] / required.clip(lower=0.1)).clip(upper=20)
    return s


def build_X(snap: pd.DataFrame):
    """Return (X, feature_names): log1p on skewed columns + one-hot category."""
    s = add_engineered(snap)
    X = pd.DataFrame(index=s.index)
    X["selling_days_elapsed"] = s["selling_days_elapsed"]
    X["days_left"] = s["days_left"]
    for c in LOG_COLS:
        X["log_" + c] = np.log1p(s[c].clip(lower=0))
    for c in ["cover_expected", "cover_observed", "gap_expected", "gap_observed", "sell_through", "speed_ratio",
              "demand_variability", "spoilage_sensitivity", "storage_temp", "temp_deviation",
              "temp_abuse_events", "distribution_hours", "handling_score", "packaging_score",
              "supplier_score", "is_promoted"]:
        X[c] = s[c]
    for cat in CATEGORIES:
        X["cat_" + cat] = (s["category"] == cat).astype(int)
    X = X.astype(float).fillna(0.0)
    return X, list(X.columns)


def skew_table(snap: pd.DataFrame) -> pd.DataFrame:
    """Skewness / kurtosis before and after log1p (shown in Tab 4)."""
    s = add_engineered(snap)
    rows = []
    for c in LOG_COLS:
        v = s[c].clip(lower=0).to_numpy(float); lv = np.log1p(v)
        rows.append({"column": c, "skew_before": skew(v), "skew_after_log1p": skew(lv),
                     "kurtosis_before": kurtosis(v), "kurtosis_after_log1p": kurtosis(lv)})
    return pd.DataFrame(rows)
