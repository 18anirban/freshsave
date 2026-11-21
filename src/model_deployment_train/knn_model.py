# ====================================================================
# MODEL 1 : K-NEAREST NEIGHBOURS (KNN)
# Idea: to judge a new batch, find the k most similar past batch-days
# and let them vote. Similar past cases wasted -> this one is at risk.
# ====================================================================
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from src.data_prep import (NUMERIC_COLS, FEATURE_NAMES, build_features, fit_imputer,
                           apply_imputer, skewed_columns)

K_VALUES = [5, 11, 21, 31, 51]
WEIGHTS = ["uniform", "distance"]


# ============================================================
# STEP 1: Prepare the data the way KNN needs it
# ============================================================
# KNN measures DISTANCE, so (a) very skewed columns are squeezed with
# log1p and (b) every column is scaled to 0-1 so no column dominates.
def fit_preprocessing(train_df):
    X = build_features(train_df)
    imputer = fit_imputer(X)                         # fill missing with train median
    X = apply_imputer(imputer, X)
    log_cols = skewed_columns(X)                     # chosen from TRAIN only
    X[log_cols] = np.log1p(X[log_cols])
    scaler = MinMaxScaler().fit(X)                   # 0..1 scale
    return {"imputer": imputer, "log_cols": log_cols, "scaler": scaler}


def transform(df, prep):
    X = apply_imputer(prep["imputer"], build_features(df))
    X[prep["log_cols"]] = np.log1p(X[prep["log_cols"]].clip(lower=0))
    return pd.DataFrame(prep["scaler"].transform(X), columns=FEATURE_NAMES, index=X.index)
# WHAT WE DID IN STEP 1: filled blanks, log-squeezed skewed columns, scaled all to 0-1.


# ============================================================
# STEP 2: Choose k and weighting (cross-validation on TRAIN only)
# ============================================================
# GroupKFold keeps all days of one batch together, so the model is never
# tested on a batch it has partly seen. The test set is NOT touched here.
def tune(X_train, y_train, groups, sample=15000, seed=42):
    idx = np.random.RandomState(seed).choice(len(X_train), min(sample, len(X_train)), replace=False)
    Xs, ys, gs = X_train.iloc[idx], y_train.iloc[idx], groups.iloc[idx]
    rows = []
    for w in WEIGHTS:
        for k in K_VALUES:
            sc = cross_val_score(KNeighborsClassifier(n_neighbors=k, weights=w),
                                 Xs, ys, groups=gs, cv=GroupKFold(5), scoring="f1")
            rows.append({"k": k, "weights": w, "cv_f1": sc.mean(), "cv_std": sc.std()})
    grid = pd.DataFrame(rows)
    best = grid.sort_values("cv_f1", ascending=False).iloc[0]
    return grid, int(best["k"]), best["weights"]
# WHAT WE DID IN STEP 2: tried 10 settings, kept the best by cross-validated F1.


# ============================================================
# STEP 3: Train the final KNN on all training rows
# ============================================================
def train(X_train, y_train, k, weights):
    t0 = time.time()
    model = KNeighborsClassifier(n_neighbors=k, weights=weights).fit(X_train, y_train)
    return model, time.time() - t0
# WHAT WE DID IN STEP 3: KNN "training" = storing the training rows.


# ============================================================
# STEP 4: Predict the chance of waste
# ============================================================
def predict_proba(model, X):
    t0 = time.time()
    p = model.predict_proba(X)[:, 1]
    return p, time.time() - t0


# ============================================================
# STEP 5: Explain one prediction - show the neighbours that voted
# ============================================================
def explain_neighbors(model, X_row, train_df, n=5):
    dist, idx = model.kneighbors(X_row, n_neighbors=n)
    nb = train_df.iloc[idx[0]][["category", "days_left", "units_left", "avg_daily_demand", "at_risk"]].copy()
    nb.insert(0, "distance", np.round(dist[0], 3))
    return nb.reset_index(drop=True)


def run(train_df, test_df):
    """Whole KNN pipeline -> dict of results (used by model_comparison)."""
    prep = fit_preprocessing(train_df)
    Xtr, Xte = transform(train_df, prep), transform(test_df, prep)
    grid, k, w = tune(Xtr, train_df["at_risk"], train_df["batch_id"])
    model, t_train = train(Xtr, train_df["at_risk"], k, w)
    p_test, t_pred = predict_proba(model, Xte)
    p_train, _ = predict_proba(model, Xtr)
    return {"name": "KNN", "prep": prep, "model": model, "grid": grid, "best_k": k, "best_weights": w,
            "prob_test": p_test, "prob_train": p_train, "train_seconds": t_train,
            "predict_seconds": t_pred, "X_test": Xte, "X_train": Xtr}
