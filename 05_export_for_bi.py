# ============================================================
# FRESHSAVE - FILE 5 OF 5: EXPORT THE FILES FOR POWER BI
# ------------------------------------------------------------
# THIS FILE: turns the results of files 1 to 4 into four clean tables (CSV) for the Power BI dashboard.
# RUN:       python 05_export_for_bi.py      (after files 02, 03 and 04)
# THEN:      open Power BI and follow powerbi/POWERBI_GUIDE.md
# ============================================================


# ============================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ------------------------------------------------------------
# This block loads all tools needed for reading and writing tables (pandas) and finding folders (pathlib).
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

# WHAT WE DID IN STEP 1:
# - Loaded the libraries.



# ============================================================
# STEP 2: LOAD EVERYTHING THE EARLIER FILES SAVED
# ------------------------------------------------------------
# - test.csv           : every test batch with its outcome (file 1)
# - knn_predictions    : KNN risk per batch (file 2)
# - tree_predictions   : Decision Tree risk per batch (file 3)
# - model_comparison   : the comparison table (file 4)
# ============================================================

FOLDER = Path(__file__).resolve().parent
OUTPUT = FOLDER / "output"
BI = FOLDER / "powerbi"
BI.mkdir(exist_ok=True)

for needed in ["test.csv", "knn_predictions.csv", "tree_predictions.csv", "model_comparison.csv"]:
    if not (OUTPUT / needed).exists():
        raise SystemExit(f"output/{needed} not found. Run files 01 to 04 first.")

test = pd.read_csv(OUTPUT / "test.csv", parse_dates=["transaction_date"])
knn = pd.read_csv(OUTPUT / "knn_predictions.csv")
tree = pd.read_csv(OUTPUT / "tree_predictions.csv")

# WHAT WE DID IN STEP 2:
# - Read the saved files.



# ============================================================
# STEP 3: TABLE 1 - ONE ROW PER BATCH (batches_scored.csv)
# ------------------------------------------------------------
# Each test batch with what really happened and what each model said.
# "result" says if the model was right: TP (caught), TN (correctly safe), FP (false alarm), FN (missed).
# ============================================================

def result(actual, predicted):
    return np.select([(actual == 1) & (predicted == 1), (actual == 0) & (predicted == 0), (actual == 0) & (predicted == 1)],
                     ["TP caught", "TN correctly safe", "FP false alarm"], default="FN missed")

batches = test.copy()
batches["month"] = batches["transaction_date"].dt.strftime("%Y-%m")
batches["knn_risk"] = knn["risk"].round(3).values
batches["knn_predicted"] = knn["predicted"].values
batches["tree_risk"] = tree["risk"].round(3).values
batches["tree_predicted"] = tree["predicted"].values
batches["knn_result"] = result(batches["at_risk"].values, batches["knn_predicted"].values)
batches["tree_result"] = result(batches["at_risk"].values, batches["tree_predicted"].values)
batches["waste_pct"] = batches["waste_pct"].round(1)
batches["waste_cost"] = batches["waste_cost"].round(2)
batches.to_csv(BI / "batches_scored.csv", index=False)

print("batches_scored.csv   :", len(batches), "rows")

# WHAT WE DID IN STEP 3:
# - Saved the batch table, with both models' risk and a right/wrong tag for each model.



# ============================================================
# STEP 4: TABLE 2 - ONE ROW PER CATEGORY (category_summary.csv)
# ------------------------------------------------------------
# For each category: how many batches, how many were really at risk, the waste cost,
# and how much of that waste each model would have warned us about.
# ============================================================

def summarise(g):
    risky = g["at_risk"] == 1
    return pd.Series({
        "batches": len(g), "at_risk_batches": int(risky.sum()),
        "at_risk_share": round(risky.mean(), 3), "waste_cost": round(g["waste_cost"].sum()),
        "waste_cost_caught_by_knn": round(g.loc[risky & (g["knn_predicted"] == 1), "waste_cost"].sum()),
        "waste_cost_caught_by_tree": round(g.loc[risky & (g["tree_predicted"] == 1), "waste_cost"].sum()),
    })

by_category = batches.groupby("category").apply(summarise, include_groups=False).reset_index()
by_category = by_category.astype({c: int for c in by_category.columns if c not in ("category", "at_risk_share")})
by_category.to_csv(BI / "category_summary.csv", index=False)
print("category_summary.csv :", len(by_category), "rows")

# WHAT WE DID IN STEP 4:
# - Added up the batches and the waste cost per category.



# ============================================================
# STEP 5: TABLE 3 - ONE ROW PER PRODUCT (product_summary.csv)
# ------------------------------------------------------------
# The same view per product, so the dashboard can list the products that waste the most.
# ============================================================

by_product = batches.groupby(["category", "product_name"]).apply(summarise, include_groups=False).reset_index()
by_product = by_product.astype({c: int for c in by_product.columns if c not in ("category", "product_name", "at_risk_share")})
by_product = by_product.sort_values("waste_cost", ascending=False)
by_product.to_csv(BI / "product_summary.csv", index=False)
print("product_summary.csv  :", len(by_product), "rows")

# WHAT WE DID IN STEP 5:
# - Saved the product table, biggest waste cost first.



# ============================================================
# STEP 6: TABLE 4 AND 5 - MODEL COMPARISON AND WHAT THE TREE LOOKS AT
# ------------------------------------------------------------
# - model_comparison.csv : the table from file 4
# - feature_importance.csv: which inputs the Decision Tree relied on most (0 to 1, adds up to 1)
# ============================================================

pd.read_csv(OUTPUT / "model_comparison.csv").to_csv(BI / "model_comparison.csv", index=False)
pd.read_csv(OUTPUT / "waste_by_model.csv").to_csv(BI / "waste_by_model.csv", index=False)
pd.read_csv(OUTPUT / "tree_feature_importance.csv").round(4).to_csv(BI / "feature_importance.csv", index=False)

print("model_comparison.csv, waste_by_model.csv, feature_importance.csv copied")
print("\nAll files are in the powerbi folder. Next: open Power BI and follow powerbi/POWERBI_GUIDE.md")

# WHAT WE DID IN STEP 6:
# - Copied the comparison and importance tables so Power BI reads everything from one folder.
