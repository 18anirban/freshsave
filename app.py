"""
FreshSave - app.py   (run with:  streamlit run app.py)
=======================================================
A proof-of-concept for ONE store. This file is only the "traffic controller": it builds the sidebar, loads the
data once, draws the step bar and hands each tab to its own file in ui/.

   Step bar (top)             Setup -> Upload day files -> Model & recommendations -> Manager dashboard
   Tab 1  Setup               ui/tab_setup.py       upload the master file + choose KNN or Decision Tree
   Tab 2  Evening upload      ui/tab_upload.py      upload Day 1 ... Day 14
   Tab 3  Daily decisions     ui/tab_decisions.py   what to discount / move / discard, and decide
   Tab 4  Manager dashboard   ui/tab_dashboard.py   waste, predictions, 15-day order plan (opens after Day 14)
   Tab 5  Model comparison    ui/tab_models.py      how the two models were tested and how they work

The real work is in src/:  scoring.py (validate + predict)  rules.py (risk -> action)  store.py (memory + Power BI)
                           master.py (master file)  order_plan.py (next 15 days)  models/ (KNN, Decision Tree)
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from src import master as M, scoring, store
from ui import helpers, step_bar, tab_dashboard, tab_decisions, tab_models, tab_setup, tab_upload

st.set_page_config(page_title="FreshSave", layout="wide")


# ============================================================
# STEP 1: STOP IF THE MODELS ARE NOT TRAINED YET
# ------------------------------------------------------------
# train.py creates the model files. The app never reads the Kaggle file itself.
# ============================================================
if not helpers.models_ready():
    st.title("FreshSave")
    st.error("The models have not been trained yet. Open a terminal in this folder and run:  python src/train.py")
    st.stop()

# ------------------------------------------------------------
# WHAT WE DID IN STEP 1:
# - Checked that every model file exists (Decision Tree and KNN) before drawing anything else.
# ------------------------------------------------------------


# ============================================================
# STEP 2: SIDEBAR - THE SETTINGS THE MANAGER (OR YOU, IN THE DEMO) CAN CHANGE
# ------------------------------------------------------------
# Rules: when to act and how strong the markdown is.   Assumption: how shoppers react to a markdown.
# ============================================================
if "flash" not in st.session_state:
    st.session_state["flash"] = []

with st.sidebar:
    st.header("FreshSave")
    st.caption(f"Store: **{C.STORE_ID}**  |  Proof of concept")
    active = scoring.active_model_name()
    st.markdown(f"**Active model:** {active}")
    st.caption("Change it in Tab 1 (Setup).")
    st.subheader("Rules")
    low = st.slider("Act when risk is above (%)", 10, 90, int(C.RULES["low"] * 100), 5) / 100
    high = st.slider("Strong action above (%)", 30, 95, int(C.RULES["high"] * 100), 5) / 100
    md_mid = st.slider("Medium markdown (%)", 5, 50, C.RULES["md_mid"], 5)
    md_high = st.slider("Strong markdown (%)", 10, 70, C.RULES["md_high"], 5)
    rules_cfg = {"low": low, "high": max(high, low), "md_mid": md_mid, "md_high": md_high}
    st.subheader("Assumption")
    uplift = st.slider("Sales response to a markdown", 0.25, 4.0, float(C.UPLIFT_PER_MARKDOWN), 0.25,
                       help="Sales rise by this number x the markdown %. 0.75 means 20% off gives +15% sales. "
                            "This is an ASSUMPTION, not measured. It only affects the 'estimated savings'.")
    st.caption(f"20% off gives about +{uplift*20:.0f}% sales.")

# ------------------------------------------------------------
# WHAT WE DID IN STEP 2:
# - Kept the sidebar exactly as before (rules and assumption sliders).
# ------------------------------------------------------------


# ============================================================
# STEP 3: LOAD EVERYTHING ONCE AND SHARE IT WITH THE TABS
# ------------------------------------------------------------
# store.refresh rebuilds every table from the saved uploads and decisions (and refreshes the Power BI CSVs).
# The tabs receive one bundle called "ctx", so they do not need many loose variables.
# ============================================================
master = M.load_master()
data = store.refresh(active, rules_cfg, uplift)
ctx = SimpleNamespace(active=active, rules_cfg=rules_cfg, uplift=uplift, data=data, scores=data["scores"],
                      have_data=not data["scores"].empty, days=store.uploaded_days(),
                      master=master, has_master=M.master_exists())

# ------------------------------------------------------------
# WHAT WE DID IN STEP 3:
# - Collected the model choice, settings, tables, uploaded days and the master file in one object.
# ------------------------------------------------------------


# ============================================================
# STEP 4: TITLE, MESSAGES AND THE STEP BAR
# ------------------------------------------------------------
# Messages ("flash") are saved by the buttons in the tabs and shown here after the page re-runs.
# ============================================================
st.title("FreshSave: decide today what to discount, move or discard")
st.caption("Each evening: upload the closing stock, see what is at risk, decide, and track the waste.")
step_bar.render_step_bar(ctx.has_master, len(ctx.days))
for kind, msg in st.session_state["flash"]:
    getattr(st, kind)(msg)
st.session_state["flash"] = []
if ctx.has_master and len(ctx.days) >= C.N_DAYS:
    st.success(f"All {C.N_DAYS} days are uploaded. The **Manager dashboard** tab is now open.")

# ------------------------------------------------------------
# WHAT WE DID IN STEP 4:
# - Drew the four-step bar, then any message from the last click.
# - When Day 14 is in, a green banner tells the user the manager dashboard is open.
# ------------------------------------------------------------


# ============================================================
# STEP 5: THE FIVE TABS
# ------------------------------------------------------------
# Each tab is one function in its own file under ui/.
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["1. Setup", "2. Evening upload", "3. Daily decisions",
                                        "4. Manager dashboard", "5. Model comparison"])
with tab1:
    tab_setup.render(ctx)
with tab2:
    tab_upload.render(ctx)
with tab3:
    tab_decisions.render(ctx)
with tab4:
    tab_dashboard.render(ctx)
with tab5:
    tab_models.render(ctx)

# ------------------------------------------------------------
# WHAT WE DID IN STEP 5:
# - Replaced 400 lines of tab code with five short calls. The tab code moved, unchanged in logic,
#   to ui/tab_*.py.
# ------------------------------------------------------------
