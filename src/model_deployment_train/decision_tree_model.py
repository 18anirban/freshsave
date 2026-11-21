# ====================================================================
# MODEL 2 : DECISION TREE
# Idea: a flowchart of yes/no questions learned from past data, e.g.
# "is stock cover bigger than days left?" -> at risk.
# ====================================================================
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier, export_text

from src.data_prep import FEATURE_NAMES, build_features, fit_imputer, apply_imputer

DEPTHS = [4, 6, 8, 10, 12, 14]
MIN_LEAVES = [10, 30, 60]


# ============================================================
# STEP 1: Prepare the data for the tree
# ============================================================
# A tree asks "is X bigger than a number?" so scaling / log is NOT needed.
# We keep real units so the rules stay readable. Only blanks are filled.
def fit_preprocessing(train_df):
    imputer = fit_imputer(build_features(train_df))
    return {"imputer": imputer}


def transform(df, prep):
    return apply_imputer(prep["imputer"], build_features(df))
# WHAT WE DID IN STEP 1: filled blanks only; no scaling, no log.


# ============================================================
# STEP 2: Choose depth and leaf size (cross-validation on TRAIN only)
# ============================================================
# Too deep = memorises the past (overfitting). We record train F1 and
# validation F1 side by side so the overfitting is visible.
def tune(X_train, y_train, groups):
    rows = []
    for d in DEPTHS:
        for leaf in MIN_LEAVES:
            r = cross_validate(DecisionTreeClassifier(max_depth=d, min_samples_leaf=leaf, random_state=42),
                               X_train, y_train, groups=groups, cv=GroupKFold(5),
                               scoring="f1", return_train_score=True)
            rows.append({"max_depth": d, "min_samples_leaf": leaf,
                         "cv_f1": r["test_score"].mean(), "train_f1": r["train_score"].mean()})
    grid = pd.DataFrame(rows)
    best = grid.sort_values("cv_f1", ascending=False).iloc[0]
    return grid, int(best["max_depth"]), int(best["min_samples_leaf"])
# WHAT WE DID IN STEP 2: tried 18 settings, kept the best by cross-validated F1.


# ============================================================
# STEP 3: Train the final tree
# ============================================================
def train(X_train, y_train, depth, leaf):
    t0 = time.time()
    model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=42).fit(X_train, y_train)
    return model, time.time() - t0


# ============================================================
# STEP 4: Predict the chance of waste
# ============================================================
def predict_proba(model, X):
    t0 = time.time()
    p = model.predict_proba(X)[:, 1]
    return p, time.time() - t0


# ============================================================
# STEP 5: Explain - rules, importance, path of one prediction
# ============================================================
def rules_text(model, depth=3):
    return export_text(model, feature_names=FEATURE_NAMES, max_depth=depth)


def importance_table(model):
    t = pd.DataFrame({"feature": FEATURE_NAMES, "importance": model.feature_importances_})
    return t.sort_values("importance", ascending=False).reset_index(drop=True)


def explain_path(model, X_row):
    """Questions the tree asked for one row, in plain words."""
    tree = model.tree_
    steps = []
    for node in model.decision_path(X_row).indices:
        if tree.children_left[node] == tree.children_right[node]:
            continue
        f, thr = FEATURE_NAMES[tree.feature[node]], tree.threshold[node]
        val = float(X_row.iloc[0][f])
        steps.append(f"{f} = {val:.2f}  ->  {'<=' if val <= thr else '>'} {thr:.2f}")
    return steps


def run(train_df, test_df):
    prep = fit_preprocessing(train_df)
    Xtr, Xte = transform(train_df, prep), transform(test_df, prep)
    grid, depth, leaf = tune(Xtr, train_df["at_risk"], train_df["batch_id"])
    model, t_train = train(Xtr, train_df["at_risk"], depth, leaf)
    p_test, t_pred = predict_proba(model, Xte)
    p_train, _ = predict_proba(model, Xtr)
    return {"name": "Decision Tree", "prep": prep, "model": model, "grid": grid,
            "best_depth": depth, "best_leaf": leaf, "prob_test": p_test, "prob_train": p_train,
            "train_seconds": t_train, "predict_seconds": t_pred, "X_test": Xte, "X_train": Xtr,
            "rules": rules_text(model), "importance": importance_table(model)}
