"""
decision_tree_model.py - THE DECISION TREE MODEL
=================================================
IDEA IN ONE LINE (for sir):
    "The tree learns a list of yes/no questions, for example 'is days_left 2 or less?' then
     'has less than half of the batch been sold?', and follows them from the top to a final
     answer: at risk of waste, or not."

Unlike KNN, the tree learns RULES that can be printed and read. It does not measure distances,
so it needs no scaling: the rules use the real numbers.

Every model file in src/models/ offers the same small set of names, so train.py, scoring.py and
the app can treat all models alike:
    NAME, FILE, PROB_COLUMN, IMPORTANCE_COLUMN, PARAM_GRID,
    base_estimator(), build(best_params), training_rows(), importance()
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

import config as C

# ============================================================
# STEP 1: NAMES THE REST OF THE PROJECT USES FOR THIS MODEL
# ------------------------------------------------------------
# NAME             : what the user sees on screen
# FILE             : the file train.py saves in models/
# PROB_COLUMN      : column in test_predictions.csv.gz holding this model's risk
# IMPORTANCE_COLUMN: column in feature_importance.csv for this model
# ============================================================
NAME = "Decision Tree"
FILE = "decision_tree.joblib"
PROB_COLUMN = "p_dt"
IMPORTANCE_COLUMN = "decision_tree"
IMPORTANCE_LABEL = "Importance"                   # axis text of the importance chart

# ------------------------------------------------------------
# WHAT WE DID IN STEP 1:
# - Gave the model one name, one file name and two column names, so no other file
#   has to hard-code "Decision Tree".
# ------------------------------------------------------------


# ============================================================
# STEP 2: THE SETTINGS TRAIN.PY TRIES (GRID SEARCH)
# ------------------------------------------------------------
# max_depth        : how many questions deep the tree may go (deeper = more detail, risk of memorising)
# min_samples_leaf : every final answer must rest on at least this many cases (higher = steadier)
# train.py tries every combination and keeps the one with the best F1 score.
# ============================================================
PARAM_GRID = {"max_depth": [6, 10, 14], "min_samples_leaf": [20, 100]}


def base_estimator():
    """An empty tree, used while searching for the best settings."""
    return DecisionTreeClassifier(random_state=C.SEED)


def build(best_params):
    """The final tree, built with the best settings found."""
    return DecisionTreeClassifier(random_state=C.SEED, **best_params)

# ------------------------------------------------------------
# WHAT WE DID IN STEP 2:
# - Listed 3 x 2 = 6 setting combinations for the tuning step.
# - random_state = SEED (42) so the same data always gives the same tree.
# - A tree does not need scaled inputs, so it is a plain classifier (no pipeline).
# ------------------------------------------------------------


# ============================================================
# STEP 3: WHICH ROWS TO TRAIN ON
# ------------------------------------------------------------
# The tree is fast, so it learns from EVERY evening row of 2023.
# ============================================================
def training_rows(X, y, rng):
    return X, y

# ------------------------------------------------------------
# WHAT WE DID IN STEP 3:
# - Returned all rows unchanged (KNN, in contrast, keeps only a sample).
# ------------------------------------------------------------


# ============================================================
# STEP 4: WHICH FACTS MATTERED MOST
# ------------------------------------------------------------
# The tree counts how much each question reduced the confusion between "at risk" and
# "not at risk". Bigger number = that fact decided more cases.
# ============================================================
def importance(model, names, X_sample, y_sample):
    return pd.Series(model.feature_importances_, index=names)

# ------------------------------------------------------------
# WHAT WE DID IN STEP 4:
# - Read the importance straight from the trained tree (no extra calculation needed).
# ------------------------------------------------------------


# ============================================================
# STEP 5: PRINT THE TOP OF THE TREE AS PLAIN-TEXT RULES (FOR EXPLAINING)
# ------------------------------------------------------------
# The full tree is very large, so we print only the first 3 levels. Read from the top:
# if the condition is true go to the first branch, otherwise to the second.
# ============================================================
def rules_text(model, names, depth=3):
    return export_text(model, feature_names=list(names), max_depth=depth)

# ------------------------------------------------------------
# WHAT WE DID IN STEP 5:
# - Turned the first levels of the trained tree into readable if/else text.
#   The Model comparison tab shows this when you open "How the model works".
# ------------------------------------------------------------
