"""
knn_model.py - THE KNN MODEL (K-NEAREST NEIGHBOURS)
====================================================
IDEA IN ONE LINE (for sir):
    "To judge a batch tonight, find the most similar batch-evenings from the past and let them
     vote: of the 11 nearest cases, how many ended up as waste?"

KNN does not learn rules. It REMEMBERS the training cases. For a new batch it measures the
distance to every remembered case and looks at the closest ones. That is why every number is
first scaled to 0-1: otherwise a big column (units left, in hundreds) would drown out a small
one (days left, 1 to 10).

Every model file in src/models/ offers the same small set of names, so train.py, scoring.py and
the app can treat all models alike:
    NAME, FILE, PROB_COLUMN, IMPORTANCE_COLUMN, PARAM_GRID,
    base_estimator(), build(best_params), training_rows(), importance()
"""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import config as C

# ============================================================
# STEP 1: NAMES THE REST OF THE PROJECT USES FOR THIS MODEL
# ------------------------------------------------------------
# NAME             : what the user sees on screen
# FILE             : the file train.py saves in models/
# PROB_COLUMN      : column in test_predictions.csv.gz holding this model's risk
# IMPORTANCE_COLUMN: column in feature_importance.csv for this model
# ============================================================
NAME = "KNN"
FILE = "knn.joblib"
PROB_COLUMN = "p_knn"
IMPORTANCE_COLUMN = "knn_permutation"
IMPORTANCE_LABEL = "Drop in F1 when shuffled"    # axis text of the importance chart

# ------------------------------------------------------------
# WHAT WE DID IN STEP 1:
# - Gave the model one name, one file name and two column names, so no other file
#   has to hard-code "KNN".
# ------------------------------------------------------------


# ============================================================
# STEP 2: THE SETTINGS TRAIN.PY TRIES (GRID SEARCH)
# ------------------------------------------------------------
# n_neighbors : how many similar past cases vote (small = jumpy, large = smoother)
# weights     : "uniform"  -> every neighbour has one equal vote
#               "distance" -> closer neighbours count more
# The "knn__" prefix means "this setting belongs to the step named knn in the pipeline".
# ============================================================
PARAM_GRID = {"knn__n_neighbors": [5, 11, 21], "knn__weights": ["uniform", "distance"]}


def base_estimator():
    """Scaler + KNN glued together, so new rows are always scaled exactly like the training rows."""
    return Pipeline([("scale", MinMaxScaler()),
                     ("knn", KNeighborsClassifier(n_jobs=-1))])


def build(best_params):
    """The final KNN, built with the best settings found."""
    knn_params = {k.split("__", 1)[1]: v for k, v in best_params.items()}
    return Pipeline([("scale", MinMaxScaler()),
                     ("knn", KNeighborsClassifier(n_jobs=-1, **knn_params))])

# ------------------------------------------------------------
# WHAT WE DID IN STEP 2:
# - Used a Pipeline: step 1 scales every number to 0-1 (Min-Max scaling), step 2 is KNN.
# - Listed 3 x 2 = 6 setting combinations for the tuning step.
# - The scaler learns its minimum and maximum from the training rows only.
# ------------------------------------------------------------


# ============================================================
# STEP 3: WHICH ROWS TO TRAIN ON
# ------------------------------------------------------------
# KNN keeps every training row in memory and compares each new batch with all of them.
# With hundreds of thousands of rows that becomes slow, so we keep a random sample of
# at most KNN_MAX_TRAIN_ROWS (set in config.py).
# ============================================================
def training_rows(X, y, rng):
    if len(X) <= C.KNN_MAX_TRAIN_ROWS:
        return X, y
    keep = np.sort(rng.choice(len(X), size=C.KNN_MAX_TRAIN_ROWS, replace=False))
    return X.iloc[keep], y[keep]

# ------------------------------------------------------------
# WHAT WE DID IN STEP 3:
# - Kept all rows if there are few; otherwise a random sample of 60,000.
# - The sample is random (not the first rows), so every month and every category is represented.
# ------------------------------------------------------------


# ============================================================
# STEP 4: WHICH FACTS MATTERED MOST (PERMUTATION IMPORTANCE)
# ------------------------------------------------------------
# KNN has no built-in importance. So we shuffle ONE column at a time on test rows and see how
# much the F1 score drops. A big drop means the model depended on that column.
# ============================================================
def importance(model, names, X_sample, y_sample):
    result = permutation_importance(model, X_sample, y_sample, n_repeats=2,
                                    random_state=C.SEED, scoring="f1")
    return pd.Series(result.importances_mean, index=names)

# ------------------------------------------------------------
# WHAT WE DID IN STEP 4:
# - Measured the F1 drop when each column is shuffled (3,000 test rows, 2 repeats).
# ------------------------------------------------------------


# ============================================================
# STEP 5: SHOW THE NEIGHBOURS BEHIND ONE PREDICTION (FOR EXPLAINING)
# ------------------------------------------------------------
# For one batch this returns the closest remembered cases, with their distance, their real
# outcome and a few readable facts. The majority outcome in this table IS the prediction.
# ============================================================
def explain_neighbours(model, X_row, names):
    scaler, knn = model.named_steps["scale"], model.named_steps["knn"]
    distance, position = knn.kneighbors(scaler.transform(X_row))
    remembered = pd.DataFrame(scaler.inverse_transform(knn._fit_X[position[0]]), columns=list(names))
    outcome = knn.classes_[knn._y[position[0]]]

    table = pd.DataFrame({
        "Evening since delivery": remembered["selling_days_elapsed"].round(0).astype(int),
        "Days left": remembered["days_left"].round(0).astype(int),
        "Share already sold": (remembered["sell_through"] * 100).round(0).astype(int).astype(str) + "%",
        "Days needed to sell out minus days left": remembered["gap_observed"].round(1),
        "Distance": distance[0].round(3),
        "What happened": ["Wasted 20%+" if o == 1 else "Sold in time" for o in outcome],
    })
    return table

# ------------------------------------------------------------
# WHAT WE DID IN STEP 5:
# - Scaled the batch the same way as training, asked KNN for its closest cases, and turned
#   them back into real numbers (un-scaled) so they are readable.
# - The app shows this table under "How the model works" in the Model comparison tab.
# ------------------------------------------------------------
