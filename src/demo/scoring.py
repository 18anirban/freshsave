# ====================================================================
# DEMO - SCORING
# File check -> prepare data -> model predicts -> risk level, reason,
# suggested action. No Streamlit in this file, so it is easy to test.
# ====================================================================
import joblib
import numpy as np
import pandas as pd

from src.data_prep import (MODEL_DIR, RAW_NUMERIC, CATEGORIES, OUTCOME_COLS,
                           build_features, apply_imputer)
from src.model_deployment_train import knn_model

ID_REQUIRED = ["batch_id", "store_id", "date", "category", "sub_category"]
REQUIRED = ID_REQUIRED + RAW_NUMERIC
HIGH, MEDIUM = 0.70, 0.40                      # risk bands (chance of waste)
MODEL_FILES = {"Decision Tree": "decision_tree.joblib", "KNN": "knn.joblib"}
DISCOUNTS = [10, 15, 20, 25, 30, 40, 50]
ACTIONS = (["Not decided", "No action"] + [f"Discount {d}%" for d in DISCOUNTS] +
           ["Move stock", "Reduce next order", "Other"])
# colour of the dropdown box after a selection (white until the manager decides)
ACTION_COLOURS = {"Not decided": "#ffffff", "No action": "#d1fae5", "Move stock": "#dbeafe",
                  "Reduce next order": "#ede9fe", "Other": "#e5e7eb", "Discount": "#fde68a"}


def action_family(action):
    return "Discount" if str(action).startswith("Discount") else str(action)


def action_colour(action):
    return ACTION_COLOURS.get(action_family(action), "#ffffff")


# ============================================================
# STEP 1: Load the trained model
# ============================================================
def models_ready():
    return all((MODEL_DIR / f).exists() for f in MODEL_FILES.values())


def load_model(name):
    return joblib.load(MODEL_DIR / MODEL_FILES[name])


# ============================================================
# STEP 2: Check the uploaded file
# ============================================================
# errors = stop (fix and upload again). warnings = fine, but good to know.
def check_file(df, prev_date=None, expected_store=None):
    errors, warnings = [], []
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        return ["Missing columns: " + ", ".join(miss)], warnings
    if df.empty:
        return ["The file has no rows."], warnings
    if [c for c in OUTCOME_COLS if c in df.columns]:
        warnings.append("Outcome columns found (" + ", ".join(c for c in OUTCOME_COLS if c in df.columns) + "). They are ignored.")
    if df["batch_id"].duplicated().any():
        errors.append(f"{int(df['batch_id'].duplicated().sum())} duplicate batch_id rows. One row per batch per day.")
    if df["date"].nunique() != 1:
        errors.append("The file must have one date only.")
    elif prev_date is not None and str(df["date"].iloc[0]) <= str(prev_date):
        errors.append(f"Date {df['date'].iloc[0]} is not after the previous day ({prev_date}).")
    if df["store_id"].nunique() != 1:
        errors.append("The file must be for one store only.")
    elif expected_store is not None and df["store_id"].iloc[0] != expected_store:
        errors.append(f"Store {df['store_id'].iloc[0]} does not match {expected_store} used on Day 0.")
    bad_cat = sorted(set(df["category"].dropna()) - set(CATEGORIES))
    if bad_cat:
        errors.append("Unknown category: " + ", ".join(bad_cat))
    for c in RAW_NUMERIC:
        num = pd.to_numeric(df[c], errors="coerce")
        n_text = int((num.isna() & df[c].notna()).sum())
        n_blank = int(df[c].isna().sum())
        if n_text:
            errors.append(f"{c}: {n_text} values are not numbers.")
        elif n_blank and c == "handling_score":
            warnings.append(f"handling_score: {n_blank} blank values. They are filled with the training median.")
        elif n_blank:
            errors.append(f"{c}: {n_blank} blank values.")
        if (num < 0).any():
            errors.append(f"{c}: negative values found.")
    if "is_promoted" in df and not set(pd.to_numeric(df["is_promoted"], errors="coerce").dropna()) <= {0, 1}:
        errors.append("is_promoted must be 0 or 1.")
    n_over = int((pd.to_numeric(df["units_left"], errors="coerce") > pd.to_numeric(df["units_received"], errors="coerce")).sum())
    if n_over:
        warnings.append(f"{n_over} rows have more units left than received (stock-count error?).")
    return errors, warnings


# ============================================================
# STEP 3: Model predicts the chance of waste for every batch
# ============================================================
def predict(df, name, bundle):
    d = df.copy()
    for c in RAW_NUMERIC:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if name == "KNN":
        X = knn_model.transform(d, bundle["prep"])
    else:
        X = apply_imputer(bundle["prep"]["imputer"], build_features(d))
    return bundle["model"].predict_proba(X)[:, 1]


# ============================================================
# STEP 4: Risk level, reason, suggested action
# ============================================================
def risk_level(p):
    return "High" if p >= HIGH else "Medium" if p >= MEDIUM else "Low"


def reason(row):
    days_of_stock = row["units_left"] / row["avg_daily_demand"] if row["avg_daily_demand"] else float("inf")
    base = f"{int(row['units_left'])} units left = {days_of_stock:.1f} days of normal sales, {int(row['days_left'])} day(s) of life left"
    if row["level"] == "High":
        return "Too much stock for the time left: " + base
    if row["level"] == "Medium":
        return "Borderline: " + base
    return "Should sell in time: " + base


def suggested_action(level):
    return {"High": "Discount or move stock today", "Medium": "Watch closely, check tomorrow", "Low": "No action"}[level]


def score(df, name, bundle):
    """Uploaded file -> same rows plus chance, level, reason, suggested action. Highest risk first."""
    d = df.drop(columns=[c for c in OUTCOME_COLS if c in df.columns]).copy()
    for c in RAW_NUMERIC:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["chance"] = predict(d, name, bundle)
    d["level"] = d["chance"].map(risk_level)
    d["reason"] = d.apply(reason, axis=1)
    d["suggested"] = d["level"].map(suggested_action)
    d["action"] = "Not decided"
    return d.sort_values("chance", ascending=False).reset_index(drop=True)
