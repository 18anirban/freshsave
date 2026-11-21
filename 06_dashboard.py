# ============================================================
# FRESHSAVE - FILE 6: THE DASHBOARD (a web page on your own computer)
# ------------------------------------------------------------
# THIS FILE: shows the results of files 1 to 5 as a dashboard in the browser.
#            Nothing is sent anywhere: it runs on your computer at http://localhost:8501
# RUN:       streamlit run 06_dashboard.py      (after python run_all.py)
# ============================================================


# ============================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ------------------------------------------------------------
# - streamlit : turns this Python file into a web page
# - pandas    : reads and groups the tables
# ============================================================

from pathlib import Path

import pandas as pd
import streamlit as st

# WHAT WE DID IN STEP 1:
# - Loaded the web-page tool and pandas.



# ============================================================
# STEP 2: PAGE SETUP AND LOAD THE TABLES
# ------------------------------------------------------------
# File 5 wrote the tables into dashboard_data. If they are missing, we say what to run.
# ============================================================

st.set_page_config(page_title="FreshSave dashboard", layout="wide")

FOLDER = Path(__file__).resolve().parent
DATA = FOLDER / "dashboard_data"
OUTPUT = FOLDER / "output"

if not (DATA / "batches_scored.csv").exists():
    st.title("FreshSave dashboard")
    st.error("The dashboard tables do not exist yet. In the terminal run:  python run_all.py")
    st.stop()


@st.cache_data
def load(name):
    return pd.read_csv(DATA / name)


batches = load("batches_scored.csv")
comparison = load("model_comparison.csv")
importance = load("feature_importance.csv")

st.title("FreshSave: which fresh-food batches will be wasted?")
st.caption("Kaggle data is synthetic. The results show the method works, not real savings. Test period: 2024.")

# WHAT WE DID IN STEP 2:
# - Set up the page and read three tables. The other tables are worked out from batches_scored.



# ============================================================
# STEP 3: FILTERS ON THE LEFT
# ------------------------------------------------------------
# The numbers in the sections below follow these filters. (The model comparison table
# is always for ALL test batches, because that is how the models were tested.)
# ============================================================

with st.sidebar:
    st.header("Filters")
    categories = sorted(batches["category"].unique())
    chosen_categories = st.multiselect("Category", categories, default=categories)
    stores = sorted(batches["store_id"].unique())
    chosen_stores = st.multiselect("Store", stores, default=stores)

view = batches[batches["category"].isin(chosen_categories) & batches["store_id"].isin(chosen_stores)]
if view.empty:
    st.warning("No batches match these filters. Choose at least one category and one store.")
    st.stop()

# WHAT WE DID IN STEP 3:
# - Kept only the batches that match the chosen categories and stores. "view" is used below.



# ============================================================
# STEP 4: HEADLINE NUMBERS
# ------------------------------------------------------------
# - Waste cost  : units wasted x cost price, added up
# - At risk     : batches where 20%+ was wasted
# - Recall      : of the really at-risk batches, the share the model flagged
# ============================================================

def recall(frame, result_column):
    risky = frame[frame["at_risk"] == 1]
    return (risky[result_column] == "TP caught").mean() if len(risky) else float("nan")


a, b, c, d, e = st.columns(5)
a.metric("Total waste cost", f"${view['waste_cost'].sum():,.0f}")
b.metric("Batches", f"{len(view):,}")
c.metric("At risk (20%+ wasted)", f"{view['at_risk'].mean():.1%}")
d.metric("KNN caught", f"{recall(view, 'knn_result'):.1%}")
e.metric("Decision Tree caught", f"{recall(view, 'tree_result'):.1%}")

# WHAT WE DID IN STEP 4:
# - Showed five numbers. The last two are recall: how many of the truly at-risk batches each model warned us about.



# ============================================================
# STEP 5: WHERE THE WASTE IS
# ------------------------------------------------------------
# Waste cost by category and by month.
# ============================================================

st.header("Where the waste is")
left, right = st.columns(2)
with left:
    st.subheader("Waste cost by category")
    st.bar_chart(view.groupby("category")["waste_cost"].sum().sort_values(ascending=False))
with right:
    st.subheader("Waste cost by month")
    st.line_chart(view.groupby("month")["waste_cost"].sum())

# WHAT WE DID IN STEP 5:
# - Drew two charts from the filtered batches.



# ============================================================
# STEP 6: KNN OR DECISION TREE?
# ------------------------------------------------------------
# - The table and chart show every score side by side (all test batches).
# - "Waste cost caught" turns recall into money for the batches you filtered.
# - A Simple rule is included as a yardstick: stock lasts longer than shelf life = at risk.
# ============================================================

st.header("KNN or Decision Tree?")
scores = comparison.set_index("model")[["accuracy", "precision", "recall", "f1", "roc_auc"]]
left, right = st.columns(2)
with left:
    st.subheader("Scores on all test batches")
    st.dataframe(scores.round(3))
with right:
    st.subheader("Accuracy, precision, recall, F1")
    st.bar_chart(scores[["accuracy", "precision", "recall", "f1"]])

risky = view[view["at_risk"] == 1]
caught = pd.DataFrame({
    "model": ["KNN", "Decision Tree"],
    "Waste cost caught (flagged in time)": [risky.loc[risky["knn_result"] == "TP caught", "waste_cost"].sum(),
                                            risky.loc[risky["tree_result"] == "TP caught", "waste_cost"].sum()],
    "Waste cost missed": [risky.loc[risky["knn_result"] == "FN missed", "waste_cost"].sum(),
                          risky.loc[risky["tree_result"] == "FN missed", "waste_cost"].sum()],
}).set_index("model")
st.subheader("Waste cost in at-risk batches: caught or missed (filtered batches)")
st.bar_chart(caught)

models_only = scores.loc[[m for m in scores.index if m in ("KNN", "Decision Tree")]]
winner = ((models_only["recall"] + models_only["f1"]) / 2).idxmax()
st.success(f"Recommendation: **{winner}** (best average of recall and F1). A missed at-risk batch becomes waste, so recall counts most.")
if "Simple rule" in scores.index:
    st.info(f"Yardstick: the simple rule scores F1 {scores.loc['Simple rule', 'f1']:.3f}. "
            "A model is only useful if it clearly beats this.")

with st.expander("ROC curves and confusion matrices"):
    for picture in ["chart_roc.png", "chart_confusion.png", "chart_metrics.png"]:
        if (OUTPUT / picture).exists():
            st.image(str(OUTPUT / picture))

# WHAT WE DID IN STEP 6:
# - Compared the models in numbers, a chart and money, and printed the recommendation and the yardstick.



# ============================================================
# STEP 7: PRODUCTS TO WATCH
# ------------------------------------------------------------
# The products with the most waste cost, with how often their batches were at risk.
# ============================================================

st.header("Products to watch")
products = view.groupby(["category", "product_name"]).agg(
    batches=("record_id", "count"), at_risk_share=("at_risk", "mean"), waste_cost=("waste_cost", "sum")
).reset_index().sort_values("waste_cost", ascending=False).head(15)
products["at_risk_share"] = (products["at_risk_share"] * 100).round(1)
products["waste_cost"] = products["waste_cost"].round(0)
st.dataframe(products.rename(columns={"at_risk_share": "at risk %"}), hide_index=True)

# WHAT WE DID IN STEP 7:
# - Listed the 15 products that waste the most money.



# ============================================================
# STEP 8: WHAT THE DECISION TREE LOOKS AT
# ------------------------------------------------------------
# Importance = how much each input helped the tree separate wasted from sold batches (adds up to 1).
# ============================================================

st.header("What the Decision Tree looks at")
left, right = st.columns(2)
with left:
    st.bar_chart(importance.sort_values("importance", ascending=False).head(10).set_index("feature"))
with right:
    rules = OUTPUT / "tree_rules.txt"
    if rules.exists():
        with st.expander("The tree's questions, in words"):
            st.code(rules.read_text())

# WHAT WE DID IN STEP 8:
# - Showed the 10 most important inputs and the printed rules of the tree.



# ============================================================
# STEP 9: LOOK AT EVERY BATCH
# ------------------------------------------------------------
# Pick which batches to see. "missed" are the ones a model called safe but that were wasted.
# ============================================================

st.header("Every batch")
show = st.selectbox("Show", ["All batches", "At risk batches", "Missed by the Decision Tree", "Missed by KNN",
                             "False alarms by the Decision Tree"])
if show == "At risk batches":
    rows = view[view["at_risk"] == 1]
elif show == "Missed by the Decision Tree":
    rows = view[view["tree_result"] == "FN missed"]
elif show == "Missed by KNN":
    rows = view[view["knn_result"] == "FN missed"]
elif show == "False alarms by the Decision Tree":
    rows = view[view["tree_result"] == "FP false alarm"]
else:
    rows = view

columns = ["transaction_date", "store_id", "category", "product_name", "shelf_life_days", "units_received",
           "units_wasted", "waste_pct", "waste_cost", "knn_risk", "tree_risk", "knn_result", "tree_result"]
st.caption(f"{len(rows):,} batches. Click a column title to sort.")
st.dataframe(rows.sort_values("waste_cost", ascending=False)[columns].head(1000), hide_index=True)

# WHAT WE DID IN STEP 9:
# - Showed the batch table, biggest waste first (the first 1000 rows), with each model's risk and whether it was right.
