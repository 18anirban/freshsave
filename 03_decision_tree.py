# ============================================================
# FRESHSAVE - FILE 3 OF 5: DECISION TREE MODEL
# ------------------------------------------------------------
# QUESTION: when a batch arrives, will 20% or more of it be wasted?
# DECISION TREE IDEA: learn a list of yes/no questions, for example
#           "is the stock cover longer than the shelf life?  is the handling score low?"
#           and follow the answers down to "at risk" or "sold in time". The questions can be printed and read.
# RUN:      python 03_decision_tree.py      (after 01_prepare_data.py)
# ============================================================


# ============================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ------------------------------------------------------------
# This block loads all tools needed for:
# - Data handling (pandas, numpy)
# - Data preprocessing (filling gaps, encoding text)
# - Decision Tree model building
# - Model evaluation
# ============================================================

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)

# WHAT WE DID IN STEP 1:
# - Loaded the libraries.



# ============================================================
# STEP 2: LOAD THE PREPARED DATA (made by 01_prepare_data.py)
# ------------------------------------------------------------
# - train.csv : batches before the split date. The tree learns from these.
# - test.csv  : later batches. The tree is judged on these, they are never used for learning.
# ============================================================

FOLDER = Path(__file__).resolve().parent
OUTPUT = FOLDER / "output"
DEPTHS_TO_TRY = [3, 4, 5, 6, 8, 10, 14]
MIN_BATCHES_PER_LEAF = 20

if not (OUTPUT / "train.csv").exists():
    raise SystemExit("output/train.csv not found. Run:  python 01_prepare_data.py")

train = pd.read_csv(OUTPUT / "train.csv")
test = pd.read_csv(OUTPUT / "test.csv")
settings = json.load(open(OUTPUT / "settings.json"))

print("=== DATA LOADED ===")
print("Training batches:", len(train), "| Testing batches:", len(test))

# WHAT WE DID IN STEP 2:
# - Read the train and test files and the list of features.
# - MIN_BATCHES_PER_LEAF = 20 stops the tree from memorising single batches.



# ============================================================
# STEP 3: DEFINE FEATURES (X) AND TARGET (y)
# ------------------------------------------------------------
# X contains all input features used for prediction (known when the batch arrives).
# y contains the answer: 1 = at risk (20%+ wasted), 0 = sold in time.
# ============================================================

num_cols = settings["numeric"]
cat_cols = settings["category"]

X_train, y_train = train[num_cols + cat_cols], train["at_risk"]
X_test, y_test = test[num_cols + cat_cols], test["at_risk"]

print("\n=== FEATURES (X) SAMPLE ===")
print(X_train.head())

print("\n=== TARGET (y) SAMPLE ===")
print(y_train.head())

# WHAT WE DID IN STEP 3:
# - Split each file into X (inputs) and y (the answer).



# ============================================================
# STEP 4: SEPARATE NUMERICAL AND CATEGORICAL FEATURES
# ------------------------------------------------------------
# Numbers (shelf life, quantity, price...) and text (category) need different preparation.
# The two lists come from file 1.
# ============================================================

print("\n=== NUMERICAL COLUMNS ===")
print(num_cols)

print("\n=== CATEGORICAL COLUMNS ===")
print(cat_cols)

# WHAT WE DID IN STEP 4:
# - Confirmed which columns are numbers and which is text.



# ============================================================
# STEP 5: IMPUTE MISSING NUMERICAL VALUES
# ------------------------------------------------------------
# This block fills missing numbers with the median of the training data.
# The median is learned from TRAIN only, then reused for TEST.
# ============================================================

imputer = SimpleImputer(strategy="median")
X_train_num = imputer.fit_transform(X_train[num_cols])
X_test_num = imputer.transform(X_test[num_cols])

# WHAT WE DID IN STEP 5:
# - Filled any gaps in the number columns.



# ============================================================
# STEP 6: ENCODE CATEGORICAL FEATURES INTO NUMERIC FORM
# ------------------------------------------------------------
# This block converts the category text (Dairy, Meat...) into 0/1 columns using One-Hot Encoding.
# ============================================================

encoder = OneHotEncoder(handle_unknown="ignore")
X_train_cat = encoder.fit_transform(X_train[cat_cols])
X_test_cat = encoder.transform(X_test[cat_cols])

# WHAT WE DID IN STEP 6:
# - Turned "Dairy" into a column cat_Dairy = 1 or 0, and the same for each category.



# ============================================================
# STEP 7: COMBINE NUMERICAL AND CATEGORICAL FEATURES
# ------------------------------------------------------------
# This block merges the number columns and the 0/1 category columns into the final tables.
# ============================================================

feature_names = list(num_cols) + list(encoder.get_feature_names_out(cat_cols))
X_train_ready = np.hstack([X_train_num, X_train_cat.toarray()])
X_test_ready = np.hstack([X_test_num, X_test_cat.toarray()])

print("\n=== FINAL TRAINING FEATURE SHAPE ===")
print(X_train_ready.shape)

print("=== FINAL TEST FEATURE SHAPE ===")
print(X_test_ready.shape)

# WHAT WE DID IN STEP 7:
# - Built one table of numbers for training and one for testing (same columns in both).



# ============================================================
# STEP 8: SCALE FEATURES - NOT NEEDED FOR A DECISION TREE
# ------------------------------------------------------------
# KNN needed scaling because it measures distance. A tree only asks "is this number above or below a cut-off?",
# so the size of a column does not matter. We skip scaling, and that is a good thing:
# the rules the tree prints stay in real units (days, units), not in 0-1 numbers.
# ============================================================

X_train_scaled, X_test_scaled = X_train_ready, X_test_ready      # same names as file 2, but unchanged numbers

# WHAT WE DID IN STEP 8:
# - Skipped scaling on purpose.



# ============================================================
# STEP 9: CHOOSE THE BEST TREE DEPTH
# ------------------------------------------------------------
# Depth = how many questions in a row the tree may ask. Too shallow = too simple, too deep = memorises the past.
# We try several depths with 5-fold cross-validation on the TRAINING data only (the test data stays untouched).
# The score used is F1.
# ============================================================

results = []
for depth in DEPTHS_TO_TRY:
    tree_try = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=MIN_BATCHES_PER_LEAF, random_state=42)
    score = cross_val_score(tree_try, X_train_scaled, y_train, cv=5, scoring="f1").mean()
    results.append({"max_depth": depth, "cv_f1": round(score, 4)})

depth_table = pd.DataFrame(results)
best_depth = int(depth_table.sort_values("cv_f1", ascending=False).iloc[0]["max_depth"])

print("\n=== F1 SCORE FOR EACH DEPTH (cross-validation on training data) ===")
print(depth_table.to_string(index=False))
print("Best depth:", best_depth)

# WHAT WE DID IN STEP 9:
# - Tried several depths and kept the best one.



# ============================================================
# STEP 10: TRAIN THE DECISION TREE
# ------------------------------------------------------------
# The tree looks for the yes/no question that best separates "wasted" from "sold in time",
# splits the batches, and repeats inside each group until it reaches the best depth.
# ============================================================

tree = DecisionTreeClassifier(max_depth=best_depth, min_samples_leaf=MIN_BATCHES_PER_LEAF, random_state=42)
tree.fit(X_train_scaled, y_train)

# WHAT WE DID IN STEP 10:
# - Built the tree with the best depth.



# ============================================================
# STEP 11: TEST THE MODEL ON THE LATER (2024) BATCHES
# ------------------------------------------------------------
# - y_pred : the class the model predicts (1 = at risk)
# - y_prob : the risk from 0 to 1 = share of wasted batches in the leaf this batch lands in
# ============================================================

y_pred = tree.predict(X_test_scaled)
y_prob = tree.predict_proba(X_test_scaled)[:, 1]

print("\n=== DECISION TREE TEST RESULTS ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# WHAT WE DID IN STEP 11:
# - Predicted every test batch and printed accuracy, the confusion matrix and the report.



# ============================================================
# STEP 12: EXPLAIN TEST RESULTS IN PLAIN LANGUAGE
# ------------------------------------------------------------
# This block prints what the numbers mean for the store, using this model's own results.
# ============================================================

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"""
TEST RESULT EXPLANATION
-----------------------
Accuracy:
    {accuracy_score(y_test, y_pred):.1%} of the {len(y_test)} test batches were predicted correctly.

Confusion Matrix:
    TN = {tn} batches correctly predicted to sell in time
    FP = {fp} batches predicted at risk but they sold in time (a false alarm: we may mark down too early)
    FN = {fn} batches predicted safe but they were wasted (a MISS: this is the costly mistake)
    TP = {tp} batches correctly predicted at risk

Precision:
    Of all batches flagged at risk, {precision:.1%} really were at risk.

Recall:
    Of all batches that really were at risk, the tree caught {recall:.1%}.

F1-score:
    {f1_score(y_test, y_pred):.3f} - one number that balances precision and recall.
""")

# WHAT WE DID IN STEP 12:
# - Explained every number in store language. A MISS costs waste, a FALSE ALARM only costs a small discount.



# ============================================================
# STEP 13: SAVE THE RESULTS FOR THE COMPARISON AND THE BI DASHBOARD
# ------------------------------------------------------------
# File 4 compares models and file 5 builds the Power BI files. Both read these two files.
# ============================================================

pd.DataFrame({"record_id": test["record_id"], "y_true": y_test, "risk": y_prob, "predicted": y_pred}) \
    .to_csv(OUTPUT / "tree_predictions.csv", index=False)

json.dump({"model": "Decision Tree", "best_depth": best_depth, "train_rows": int(len(train)), "test_rows": int(len(test)),
           "accuracy": accuracy_score(y_test, y_pred), "precision": precision, "recall": recall,
           "f1": f1_score(y_test, y_pred), "roc_auc": roc_auc_score(y_test, y_prob),
           "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
          open(OUTPUT / "tree_results.json", "w"), indent=2)

print("Saved: output/tree_predictions.csv and output/tree_results.json")

# WHAT WE DID IN STEP 13:
# - Saved each test batch's risk and the summary numbers.

# the printed rules and the importance of each feature (used in file 5 and for sir's questions)
feature_names_list = feature_names
open(OUTPUT / "tree_rules.txt", "w").write(export_text(tree, feature_names=feature_names_list, max_depth=4))
pd.DataFrame({"feature": feature_names_list, "importance": tree.feature_importances_}) \
    .sort_values("importance", ascending=False).to_csv(OUTPUT / "tree_feature_importance.csv", index=False)



# ============================================================
# STEP 14: RANDOM SINGLE-BATCH TEST
# ------------------------------------------------------------
# This block tests the model on a randomly selected batch from the test set. Every run picks a different one.
# It prints:
# - The batch's profile
# - The model's prediction and the real outcome
# - The exact yes/no questions the tree asked for this batch
# ============================================================

example_index = random.randint(0, len(X_test_scaled) - 1)
example_features = X_test_scaled[example_index].reshape(1, -1)

prediction = tree.predict(example_features)[0]
risk = tree.predict_proba(example_features)[0, 1]
actual = y_test.iloc[example_index]

print("\n=== RANDOM SINGLE BATCH TEST ===")
print("Selected Test Index:", example_index)
print("Predicted:", "AT RISK" if prediction == 1 else "Sold in time", f"(risk {risk:.0%})")
print("Actual:   ", "AT RISK (wasted 20%+)" if actual == 1 else "Sold in time")

print("\n=== BATCH PROFILE ===")
print(test.iloc[example_index][["category", "product_name", "shelf_life_days", "units_received", "daily_demand",
                                "stock_cover_days", "cover_gap_days", "spoilage_sensitivity", "handling_score"]])

print("\n=== THE QUESTIONS THE TREE ASKED FOR THIS BATCH ===")
nodes = tree.decision_path(example_features).indices
for node in nodes:
    if tree.tree_.children_left[node] == tree.tree_.children_right[node]:      # a leaf: the answer
        share = tree.tree_.value[node][0]
        share = share[1] / share.sum()
        print(f"  ANSWER: {share:.0%} of the {tree.tree_.n_node_samples[node]} similar past batches were wasted")
    else:
        name = feature_names[tree.tree_.feature[node]]
        cut = tree.tree_.threshold[node]
        value = example_features[0, tree.tree_.feature[node]]
        print(f"  Is {name} <= {cut:.2f}?  (this batch: {value:.2f})  ->", "YES" if value <= cut else "NO")

# WHAT WE DID IN STEP 14:
# - Picked one test batch, showed the prediction against the truth, and printed the path through the tree.
