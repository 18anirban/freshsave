"""
scoring.py - validate an uploaded daily file, clean it, and score it with the ACTIVE model
==========================================================================================
Used by the app every time a file is uploaded (Tab 1) or the model is switched (Tab 4).
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src import features as F
from src.models import MODEL_FILES          # {"Decision Tree": "decision_tree.joblib", "KNN": "knn.joblib"}

REQUIRED = ["date", "day_no", "batch_id", "store_id", "category", "product_name", "delivered_date", "expiry_date",
            "selling_days_elapsed", "days_left", "qty_received", "qty_left", "units_sold_today", "units_sold_total",
            "base_price", "current_price", "cost_price"] + [c for c in F.STATIC_COLS if c not in ("base_price", "cost_price")]
NUMERIC = [c for c in REQUIRED if c not in ("date", "batch_id", "store_id", "category", "product_name",
                                              "delivered_date", "expiry_date")]
_cache = {}


def validate_and_clean(df: pd.DataFrame):
    """Return (clean_df, errors, warnings). ANY error blocks the upload; warnings do not."""
    errors, warnings = [], []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return None, [f"Missing columns: {', '.join(missing)}"], warnings
    d = df[REQUIRED].copy()
    for c in ["batch_id", "store_id", "category", "product_name"]:
        d[c] = d[c].astype(str).str.strip()
    for c in NUMERIC:                                           # text -> numbers; bad values become empty
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ["date", "delivered_date", "expiry_date"]:
        d[c] = pd.to_datetime(d[c], errors="coerce")
    bad = d[NUMERIC + ["date"]].isna().any(axis=1)
    if bad.any():
        warnings.append(f"{int(bad.sum())} row(s) had empty or non-numeric values and were removed.")
        d = d[~bad]
    if d.empty:
        return None, ["The file has no usable rows."], warnings
    if (d["store_id"] != C.STORE_ID).any():
        errors.append(f"This app is for {C.STORE_ID} only; the file has other store IDs: "
                      f"{sorted(d.loc[d['store_id'] != C.STORE_ID, 'store_id'].unique())[:3]}")
    if d["day_no"].nunique() != 1:
        errors.append(f"One file must hold exactly one day; found day numbers {sorted(d['day_no'].unique().astype(int))[:5]}")
    dup = d.duplicated("batch_id")
    if dup.any():
        warnings.append(f"{int(dup.sum())} duplicate batch row(s) removed (kept the first).")
        d = d[~dup]
    if (d[["qty_left", "qty_received", "units_sold_total", "days_left"]] < 0).any().any():
        errors.append("Negative quantities or days found - please check the file.")
    if (d["qty_left"] > d["qty_received"]).any():
        errors.append("Some batches have more stock left than was received.")
    unknown = sorted(set(d["category"]) - set(F.CATEGORIES))
    if unknown:
        warnings.append(f"Unknown categories {unknown}: scored as 'none of the known categories'.")
    return d.reset_index(drop=True), errors, warnings


def active_model_name() -> str:
    p = C.MODELS_DIR / "active_model.json"
    name = json.load(open(p))["active"] if p.exists() else "Decision Tree"
    return name if name in MODEL_FILES else "Decision Tree"     # an old file may still name a model that no longer exists


def set_active_model(name: str):
    p = C.MODELS_DIR / "active_model.json"
    info = json.load(open(p))
    info["active"] = name
    json.dump(info, open(p, "w"))


def load_model(name: str):
    if name not in _cache:
        _cache[name] = joblib.load(C.MODELS_DIR / MODEL_FILES[name])
    return _cache[name]


def reason_text(row) -> str:
    """Plain-English reason, built from the 'gap' feature (days needed to sell vs days left)."""
    if row["days_left"] <= 0:
        return "Expires tonight - remove from the shelf."
    speed = row["units_sold_total"] / max(row["selling_days_elapsed"], 1)
    if speed <= 0:
        return f"Nothing sold yet and only {int(row['days_left'])} day(s) left."
    need = row["qty_left"] / max(speed, 0.5)
    return f"At the current pace it needs {need:.1f} days to sell out, only {int(row['days_left'])} left."


def score(df: pd.DataFrame, model_name: str = None) -> pd.DataFrame:
    """Add 'risk' (0-1) and 'reason'. Batches expiring tonight are not scored (nothing left to predict)."""
    model_name = model_name or active_model_name()
    s = df.copy()
    s["risk"] = np.nan
    live = s["days_left"] > 0
    if live.any():
        X, names = F.build_X(s[live])
        expected = json.load(open(C.MODELS_DIR / "feature_names.json"))
        s.loc[live, "risk"] = load_model(model_name).predict_proba(X[expected])[:, 1]
    s["reason"] = s.apply(reason_text, axis=1)
    s["model_used"] = model_name
    return s
