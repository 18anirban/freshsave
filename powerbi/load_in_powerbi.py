# ============================================================
# FRESHSAVE - LOAD ALL TABLES INTO POWER BI WITH ONE SCRIPT
# ------------------------------------------------------------
# HOW TO USE (once):
#   1. Power BI Desktop > File > Options and settings > Options > Python scripting
#      > set the Python home directory to your project's .venv (or your Python). It needs pandas.
#   2. Home > Get data > More... > Python script > Connect
#   3. Paste this whole file, change PROJECT_FOLDER below, click OK
#   4. In the Navigator tick all six tables and click Load
# AFTER THAT: run  python run_all.py  in VS Code, then click Home > Refresh in Power BI.
# ============================================================


# ============================================================
# STEP 1: IMPORT REQUIRED LIBRARIES AND SET THE PROJECT FOLDER
# ------------------------------------------------------------
# CHANGE THE NEXT LINE to the folder where your freshsave project is.
# ============================================================

import pandas as pd

PROJECT_FOLDER = r"C:\Users\18ani\Documents\freshsave"

# WHAT WE DID IN STEP 1:
# - Said where the project is. Keep the r before the quote marks and use the full path.



# ============================================================
# STEP 2: READ EACH CSV INTO A TABLE
# ------------------------------------------------------------
# Power BI turns every table (DataFrame) in this script into a table in the report.
# The names on the left become the table names, the same names as in the guide.
# ============================================================

batches_scored = pd.read_csv(PROJECT_FOLDER + r"\powerbi\batches_scored.csv", parse_dates=["transaction_date"])
category_summary = pd.read_csv(PROJECT_FOLDER + r"\powerbi\category_summary.csv")
product_summary = pd.read_csv(PROJECT_FOLDER + r"\powerbi\product_summary.csv")
model_comparison = pd.read_csv(PROJECT_FOLDER + r"\powerbi\model_comparison.csv")
waste_by_model = pd.read_csv(PROJECT_FOLDER + r"\powerbi\waste_by_model.csv")
feature_importance = pd.read_csv(PROJECT_FOLDER + r"\powerbi\feature_importance.csv")

# WHAT WE DID IN STEP 2:
# - Loaded the six tables. Do not add other tables to this script, or they will show up in Power BI too.
