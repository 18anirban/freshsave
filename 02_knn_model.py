# ============================================================
# FRESHSAVE - FILE 2 OF 5: KNN MODEL
# ------------------------------------------------------------
# QUESTION: when a batch arrives, will 20% or more of it be wasted?
# KNN IDEA: find the K most similar past batches and let them vote.
#           If most of them ended up wasted, this batch is probably at risk too.
# RUN:      python 02_knn_model.py      (after 01_prepare_data.py)
# ============================================================


# ============================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ------------------------------------------------------------
# This block loads all tools needed for:
# - Data handling (pandas, numpy)
# - Data preprocessing (filling gaps, encoding text, scaling)
# - KNN model building
# - Model evaluation
# ============================================================

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)

# WHAT WE DID IN STEP 1:
# - Loaded the libraries.



# ============================================================
# STEP 2: LOAD THE PREPARED DATA (made by 01_prepare_data.py)
# ------------------------------------------------------------
# - train.csv : batches before the split date. KNN learns from these.
# - test.csv  : later batches. KNN is judged on these, they are never used for learning.
# MAX_TRAIN_ROWS: KNN compares every new batch with every stored batch, so a very large
#                 training set makes it slow. We keep a random sample if there are more rows than this.
# ============================================================

FOLDER = Path(__file__).resolve().parent
OUTPUT = FOLDER / "output"
MAX_TRAIN_ROWS = 60000
K_VALUES_TO_TRY = [3, 5, 7, 11, 15, 21]

if not (OUTPUT / "train.csv").exists():
    raise SystemExit("output/train.csv not found. Run:  python 01_prepare_data.py")

train = pd.read_csv(OUTPUT / "train.csv")
test = pd.read_csv(OUTPUT / "test.csv")
settings = json.load(open(OUTPUT / "settings.json"))

if len(train) > MAX_TRAIN_ROWS:
    train = train.sample(MAX_TRAIN_ROWS, random_state=42).reset_index(drop=True)

print("=== DATA LOADED ===")
print("Training batches:", len(train), "| Testing batches:", len(test))

# WHAT WE DID IN STEP 2:
# - Read the train and test files and the list of features.



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
# STEP 8: SCALE FEATURES TO 0-1 RANGE
# ------------------------------------------------------------
# KNN measures DISTANCE between batches. A column like "units received" (hundreds) would drown out
# a column like "is promoted" (0 or 1). Min-Max scaling puts every column between 0 and 1,
# so each column counts fairly. (The Decision Tree does not need this step.)
# ============================================================

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_ready)
X_test_scaled = scaler.transform(X_test_ready)

print("\n=== EXAMPLE SCALED TRAINING ROW ===")
print(np.round(X_train_scaled[0], 2))

# WHAT WE DID IN STEP 8:
# - Scaled every column to 0-1 (the scale is learned from TRAIN only).



# ============================================================
# STEP 9: CHOOSE THE BEST K (NUMBER OF NEIGHBOURS)
# ------------------------------------------------------------
# K = how many similar batches vote. Too small = jumpy, too large = blurry.
# We try several values and test each with 5-fold cross-validation on the TRAINING data only
# (the test data stays untouched). The score used is F1.
# ============================================================

search_rows = min(len(X_train_scaled), 8000)               # a sample keeps this step quick
idx = np.random.RandomState(42).choice(len(X_train_scaled), search_rows, replace=False)

results = []
for k in K_VALUES_TO_TRY:
    score = cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train_scaled[idx], y_train.iloc[idx],
                            cv=5, scoring="f1").mean()
    results.append({"k": k, "cv_f1": round(score, 4)})

k_table = pd.DataFrame(results)
best_k = int(k_table.sort_values("cv_f1", ascending=False).iloc[0]["k"])

print("\n=== F1 SCORE FOR EACH K (cross-validation on training data) ===")
print(k_table.to_string(index=False))
print("Best K:", best_k)

# WHAT WE DID IN STEP 9:
# - Tried K = 3, 5, 7, 11, 15, 21 and kept the best one. The heart-attack example used a fixed K = 7.



# ============================================================
# STEP 10: TRAIN THE KNN CLASSIFIER
# ------------------------------------------------------------
# KNN does not "learn rules". Training only STORES the scaled training batches.
# The work happens when it predicts: it finds the K closest stored batches and takes the majority vote.
# ============================================================

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train)

# WHAT WE DID IN STEP 10:
# - Stored the training batches with the best K.



# ============================================================
# STEP 11: TEST THE MODEL ON THE LATER (2024) BATCHES
# ------------------------------------------------------------
# - y_pred : the class the model predicts (1 = at risk)
# - y_prob : the risk from 0 to 1 = share of the K neighbours that were wasted
# ============================================================

y_pred = knn.predict(X_test_scaled)
y_prob = knn.predict_proba(X_test_scaled)[:, 1]

print("\n=== KNN TEST RESULTS ===")
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
    Of all batches that really were at risk, KNN caught {recall:.1%}.

F1-score:
    {f1_score(y_test, y_pred):.3f} - one number that balances precision and recall.
""")

# WHAT WE DID IN STEP 12:
# - Explained every number in store language. A MISS costs waste, a FALSE ALARM only costs a small discount.



# ============================================================
# STEP 13: SAVE THE RESULTS FOR THE COMPARISON AND THE BI DASHBOARD
# ------------------------------------------------------------
# File 4 compares models and file 5 builds the dashboard tables. Both read these two files.
# ============================================================

pd.DataFrame({"record_id": test["record_id"], "y_true": y_test, "risk": y_prob, "predicted": y_pred}) \
    .to_csv(OUTPUT / "knn_predictions.csv", index=False)

json.dump({"model": "KNN", "best_k": best_k, "train_rows": int(len(train)), "test_rows": int(len(test)),
           "accuracy": accuracy_score(y_test, y_pred), "precision": precision, "recall": recall,
           "f1": f1_score(y_test, y_pred), "roc_auc": roc_auc_score(y_test, y_prob),
           "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
          open(OUTPUT / "knn_results.json", "w"), indent=2)

print("Saved: output/knn_predictions.csv and output/knn_results.json")

# WHAT WE DID IN STEP 13:
# - Saved each test batch's risk and the summary numbers.



# ============================================================
# STEP 14: RANDOM SINGLE-BATCH TEST
# ------------------------------------------------------------
# This block tests the model on a randomly selected batch from the test set. Every run picks a different one.
# It prints:
# - The batch's profile
# - The model's prediction and the real outcome
# - The K most similar past batches that voted, and how each of them ended
# ============================================================

example_index = random.randint(0, len(X_test_scaled) - 1)
example_features = X_test_scaled[example_index].reshape(1, -1)

prediction = knn.predict(example_features)[0]
risk = knn.predict_proba(example_features)[0, 1]
actual = y_test.iloc[example_index]

print("\n=== RANDOM SINGLE BATCH TEST ===")
print("Selected Test Index:", example_index)
print("Predicted:", "AT RISK" if prediction == 1 else "Sold in time", f"(risk {risk:.0%})")
print("Actual:   ", "AT RISK (wasted 20%+)" if actual == 1 else "Sold in time")

print("\n=== BATCH PROFILE ===")
print(test.iloc[example_index][["category", "product_name", "shelf_life_days", "units_received", "daily_demand",
                                "stock_cover_days", "cover_gap_days", "spoilage_sensitivity", "handling_score"]])

distances, neighbour_index = knn.kneighbors(example_features)
neighbours = train.iloc[neighbour_index[0]][["category", "shelf_life_days", "units_received", "cover_gap_days", "at_risk"]].copy()
neighbours["distance"] = np.round(distances[0], 3)
neighbours["what_happened"] = np.where(neighbours["at_risk"] == 1, "Wasted 20%+", "Sold in time")

print(f"\n=== THE {best_k} MOST SIMILAR PAST BATCHES (THEY VOTED) ===")
print(neighbours.drop(columns="at_risk").to_string(index=False))

# WHAT WE DID IN STEP 14:
# - Picked one test batch, showed the prediction against the truth, and showed the neighbours behind the vote.
