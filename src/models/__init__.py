"""
src/models - ONE FILE PER MODEL
================================
    decision_tree_model.py   the Decision Tree (learns readable yes/no rules)
    knn_model.py             KNN (compares a batch with the most similar past cases)

REGISTRY is the single list of models the project knows. To add a third model, write another
file with the same names as these two and add it here: train.py, scoring.py and the app pick
it up without further changes.
"""
from src.models import decision_tree_model, knn_model

# ============================================================
# STEP 1: THE LIST OF MODELS
# ------------------------------------------------------------
# name on screen -> the file that contains the model's code
# MODEL_FILES    -> name on screen -> the file saved in models/ by train.py
# ============================================================
REGISTRY = {m.NAME: m for m in (decision_tree_model, knn_model)}
MODEL_FILES = {name: m.FILE for name, m in REGISTRY.items()}

# ------------------------------------------------------------
# WHAT WE DID IN STEP 1:
# - Collected the two model files in one dictionary. The rest of the project asks this
#   dictionary instead of writing "Decision Tree" or "KNN" in many places.
# ------------------------------------------------------------
