# ============================================================
# FRESHSAVE - FILE 4 OF 5: COMPARE THE MODELS
# ------------------------------------------------------------
# QUESTION: which model finds the at-risk batches better, KNN or the Decision Tree?
# THIS FILE: puts both models (and one simple rule) side by side on the SAME test batches,
#            prints a table, draws charts and says which model we recommend.
# RUN:       python 04_compare_models.py      (after files 02 and 03)
# ============================================================


# ============================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ------------------------------------------------------------
# This block loads all tools needed for:
# - Reading the saved results (json, pandas)
# - Drawing charts (matplotlib)
# - A few metrics for the simple rule (scikit-learn)
# ============================================================

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                 # draw into picture files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve)

# WHAT WE DID IN STEP 1:
# - Loaded the libraries.



# ============================================================
# STEP 2: LOAD THE RESULTS OF BOTH MODELS
# ------------------------------------------------------------
# Files 2 and 3 saved, for every test batch, the risk each model gave. We read them here.
# ============================================================

FOLDER = Path(__file__).resolve().parent
OUTPUT = FOLDER / "output"

for needed in ["knn_results.json", "tree_results.json"]:
    if not (OUTPUT / needed).exists():
        raise SystemExit(f"output/{needed} not found. Run files 02 and 03 first.")

test = pd.read_csv(OUTPUT / "test.csv")
knn_results = json.load(open(OUTPUT / "knn_results.json"))
tree_results = json.load(open(OUTPUT / "tree_results.json"))
knn_pred = pd.read_csv(OUTPUT / "knn_predictions.csv")
tree_pred = pd.read_csv(OUTPUT / "tree_predictions.csv")

# both models must have been tested on exactly the same batches
assert (knn_pred["record_id"].values == tree_pred["record_id"].values).all(), "The two models were tested on different batches."

print("=== RESULTS LOADED ===")
print("Test batches:", len(test), "| KNN K =", knn_results["best_k"], "| Tree depth =", tree_results["best_depth"])

# WHAT WE DID IN STEP 2:
# - Read the saved results and checked both models used the same test batches.



# ============================================================
# STEP 3: A SIMPLE RULE AS A YARDSTICK
# ------------------------------------------------------------
# Before we say a model is good, we ask: how well would a plain rule do?
#   RULE: "if the stock would last longer than the shelf life (cover_gap_days > 0), call it at risk"
# If a model cannot beat this, it adds nothing.
# ============================================================

y_true = test["at_risk"].to_numpy()
rule_pred = (test["cover_gap_days"] > 0).astype(int).to_numpy()
tn, fp, fn, tp = confusion_matrix(y_true, rule_pred).ravel()
rule_results = {"model": "Simple rule", "accuracy": accuracy_score(y_true, rule_pred),
                "precision": precision_score(y_true, rule_pred), "recall": recall_score(y_true, rule_pred),
                "f1": f1_score(y_true, rule_pred), "roc_auc": np.nan,
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

# WHAT WE DID IN STEP 3:
# - Scored the one-line rule on the same test batches. It has no risk number, so no ROC-AUC.



# ============================================================
# STEP 4: THE COMPARISON TABLE
# ------------------------------------------------------------
# - Accuracy : share of all batches predicted correctly
# - Precision: of the batches flagged at risk, how many really were
# - Recall   : of the batches that really were at risk, how many we caught
# - F1       : one number balancing precision and recall
# - ROC-AUC  : how well the risk number ranks risky above safe batches (0.5 = guessing, 1.0 = perfect)
# ============================================================

columns = ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "tp", "fp", "tn", "fn"]
table = pd.DataFrame([{**knn_results, "model": "KNN"}, {**tree_results, "model": "Decision Tree"}, rule_results])[columns]
table.to_csv(OUTPUT / "model_comparison.csv", index=False)

print("\n=== MODEL COMPARISON (same", len(test), "test batches) ===")
print(table.round(3).to_string(index=False))

# WHAT WE DID IN STEP 4:
# - Built one table with every number side by side and saved it (the BI dashboard reads it too).



# ============================================================
# STEP 5: WHAT THE MISTAKES COST
# ------------------------------------------------------------
# A MISS (an at-risk batch we call safe) means the waste happens. A FALSE ALARM costs only an early discount.
# Here we add up the real waste cost (units wasted x cost price) of the at-risk batches each model caught or missed.
# ============================================================

at_risk = test["at_risk"] == 1
total_waste = test.loc[at_risk, "waste_cost"].sum()
money = []
for name, predicted in [("KNN", knn_pred["predicted"]), ("Decision Tree", tree_pred["predicted"]),
                        ("Simple rule", pd.Series(rule_pred))]:
    caught = test.loc[at_risk & (predicted.values == 1), "waste_cost"].sum()
    money.append({"model": name, "waste_cost_in_flagged_batches": round(caught),
                  "waste_cost_missed": round(total_waste - caught), "share_caught": round(caught / total_waste, 3)})
money = pd.DataFrame(money)
money.to_csv(OUTPUT / "waste_by_model.csv", index=False)

print("\n=== WASTE COST OF THE AT-RISK BATCHES THAT EACH MODEL CAUGHT OR MISSED ===")
print("Total waste cost in at-risk batches:", round(total_waste))
print(money.to_string(index=False))

# WHAT WE DID IN STEP 5:
# - Turned recall into money: how much of the real waste cost each model could have warned us about.



# ============================================================
# STEP 6: DRAW THE CHARTS (saved as pictures in output/)
# ------------------------------------------------------------
# 1) bars of the main metrics, 2) ROC curves, 3) confusion matrices
# ============================================================

fig, ax = plt.subplots(figsize=(8, 4.5))
metrics = ["accuracy", "precision", "recall", "f1"]
x = np.arange(len(metrics))
for i, row in enumerate(table.itertuples()):
    ax.bar(x + i * 0.27, [getattr(row, m) for m in metrics], 0.27, label=row.model)
ax.set_xticks(x + 0.27)
ax.set_xticklabels([m.capitalize() for m in metrics])
ax.set_ylim(0, 1.15); ax.set_ylabel("Score"); ax.set_title("Model comparison on the test batches"); ax.legend(loc="upper center", ncol=3)
fig.tight_layout(); fig.savefig(OUTPUT / "chart_metrics.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.5, 5))
for name, pred in [("KNN", knn_pred), ("Decision Tree", tree_pred)]:
    fpr, tpr, _ = roc_curve(pred["y_true"], pred["risk"])
    ax.plot(fpr, tpr, label=name)
ax.plot([0, 1], [0, 1], "--", color="grey", label="Guessing")
ax.set_xlabel("False alarm rate"); ax.set_ylabel("Share of at-risk batches caught"); ax.set_title("ROC curve"); ax.legend()
fig.tight_layout(); fig.savefig(OUTPUT / "chart_roc.png", dpi=150); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
for ax, name, pred in [(axes[0], "KNN", knn_pred), (axes[1], "Decision Tree", tree_pred)]:
    cm = confusion_matrix(pred["y_true"], pred["predicted"])
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", fontsize=12)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Safe", "At risk"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["Safe", "At risk"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(name)
fig.tight_layout(); fig.savefig(OUTPUT / "chart_confusion.png", dpi=150); plt.close(fig)

print("\nCharts saved: output/chart_metrics.png, chart_roc.png, chart_confusion.png")

# WHAT WE DID IN STEP 6:
# - Saved three pictures you can paste into slides.



# ============================================================
# STEP 7: THE RECOMMENDATION
# ------------------------------------------------------------
# Rule: the best model has the highest average of recall and F1. Recall counts because a missed batch becomes waste.
# ============================================================

models_only = table[table["model"].isin(["KNN", "Decision Tree"])].set_index("model")
score = (models_only["recall"] + models_only["f1"]) / 2
winner = score.idxmax()
other = [m for m in models_only.index if m != winner][0]

print(f"""
RECOMMENDATION
--------------
Use the {winner}.
  Recall: {models_only.loc[winner, 'recall']:.1%} against {models_only.loc[other, 'recall']:.1%} for the {other}.
  F1:     {models_only.loc[winner, 'f1']:.3f} against {models_only.loc[other, 'f1']:.3f}.
  The simple rule scored F1 {rule_results['f1']:.3f}: {'both models beat it' if models_only['f1'].min() > rule_results['f1'] else 'NOT beaten by: ' + ', '.join(models_only.index[models_only['f1'] <= rule_results['f1']]) + ' - say this honestly'}.
Reminder: the data is synthetic (Kaggle), so this proves the method, not savings in a real store.
""")

# WHAT WE DID IN STEP 7:
# - Picked the winner by recall and F1 and printed the reason in words, including the check against the simple rule.
