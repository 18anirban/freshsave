"""
tab_setup.py - TAB 1: SETUP  (master file + model choice)
==========================================================
Two things happen here, in this order:
  1. Upload the MASTER FILE: the fortnight's batch list (delivery date, expiry date, quantity, prices).
  2. Choose the MODEL that will score the batches: KNN or Decision Tree.
Only after the master file is saved do the other tabs open for use.
"""
import pandas as pd
import streamlit as st

import config as C
from src import master as M, scoring
from ui.helpers import read_json, show_df, money

MODELS = list(scoring.MODEL_FILES)
MODEL_HELP = {
    "Decision Tree": "Learns a list of yes/no questions (for example: is 2 or fewer days left? has less than half "
                     "been sold?) and follows them to an answer. Easy to read and explain: the rules can be printed.",
    "KNN": "Compares tonight's batch with the most similar past batch-evenings and lets them vote. "
           "No rules to read, but easy to explain with examples: you can show the neighbours that voted.",
}


# ============================================================
# STEP 1: UPLOAD AND SAVE THE MASTER FILE
# ------------------------------------------------------------
# Two ways in: upload your own file, or load the sample master file that make_daily_files.py created.
# The file is checked (columns, dates, numbers, duplicates) before it is saved.
# ============================================================
def _save_master(df, filename):
    clean, errors, warnings = M.validate_and_clean_master(df)
    if errors:
        return False, f"{filename}: " + " ".join(errors)
    M.save_master(clean)
    return True, f"{filename}: master file saved ({len(clean)} batches)." + (f" Notes: {' '.join(warnings)}" if warnings else "")


def _master_block(ctx):
    st.subheader("Step 1. Upload the master file")
    st.write("The master file lists every batch of the fortnight: batch, product, category, delivery date, "
             "expiry date (this gives the batch life), quantity received and prices.")
    file = st.file_uploader("Master file (CSV)", type="csv", key="master_upload")
    c1, c2, c3 = st.columns(3)

    if c1.button("Check and save the master file", disabled=file is None):
        try:
            ok, msg = _save_master(pd.read_csv(file), file.name)
        except Exception as ex:                                    # unreadable file
            ok, msg = False, f"{file.name}: could not be read ({ex})"
        st.session_state["flash"].append(("success" if ok else "error", msg))
        st.rerun()

    if c2.button("Load the sample master file from the folder", disabled=not C.MASTER_SAMPLE.exists(),
                 help="DailyFiles-Retail/Master_Fortnight_Batches.csv (made by src/make_daily_files.py)"):
        ok, msg = _save_master(pd.read_csv(C.MASTER_SAMPLE), C.MASTER_SAMPLE.name)
        st.session_state["flash"].append(("success" if ok else "error", msg))
        st.rerun()

    if c3.button("Remove the master file", disabled=not ctx.has_master):
        M.remove_master()
        st.session_state["flash"].append(("success", "The master file was removed. Uploaded days are kept."))
        st.rerun()

# ------------------------------------------------------------
# WHAT WE DID IN STEP 1:
# - Added a file box and three buttons: save the uploaded file, load the sample file, remove the saved file.
# - The file is validated by src/master.py before it is saved in state/master.csv.
# ------------------------------------------------------------


# ============================================================
# STEP 2: SHOW WHAT WAS LOADED
# ------------------------------------------------------------
# Headline numbers, the deliveries in the fortnight, and a preview of the first rows.
# ============================================================
def _master_summary(ctx):
    if not ctx.has_master:
        st.info("No master file yet. Upload it above (or load the sample file) to continue.")
        return
    s = M.summary(ctx.master)
    a, b, c, d, e = st.columns(5)
    a.metric("Batches", f"{s['batches']:,}")
    b.metric("Products", s["products"])
    c.metric("Units received", f"{s['units']:,.0f}")
    d.metric("Stock value (at cost)", money(s["stock_value"]))
    e.metric("Average batch life", f"{s['avg_life']:.1f} days")
    dl = s["deliveries"].assign(delivered_date=lambda x: x["delivered_date"].dt.strftime("%a %d %b"))
    with st.expander("Deliveries in the fortnight and a preview of the file"):
        show_df(dl.rename(columns={"delivered_date": "Delivery date", "batches": "Batches"}))
        show_df(ctx.master.head(15))

# ------------------------------------------------------------
# WHAT WE DID IN STEP 2:
# - Showed batches, products, units, stock value and average batch life.
# - Listed the delivery dates (the Sunday deliveries and Wednesday top-ups) and the first 15 rows.
# ------------------------------------------------------------


# ============================================================
# STEP 3: CHOOSE THE MODEL
# ------------------------------------------------------------
# The choice is saved in models/active_model.json. Every score, recommendation and the dashboard
# use it. Switching re-scores all uploaded days at once.
# ============================================================
def _switch():
    scoring.set_active_model(st.session_state["model_pick"])


def _model_block(ctx):
    st.subheader("Step 2. Choose the model")
    info = read_json("active_model.json")
    current = info["active"] if info["active"] in MODELS else MODELS[0]
    st.selectbox("Model that scores the batches", MODELS, index=MODELS.index(current), key="model_pick", on_change=_switch)
    st.info(MODEL_HELP[ctx.active])
    st.success(f"Default winner on the 2024 test data: **{info['winner']}**. {info['why']}")
    st.caption("Open the Model comparison tab for the full test results of both models.")

# ------------------------------------------------------------
# WHAT WE DID IN STEP 3:
# - Moved the model choice here from the Model comparison tab, with a plain explanation of each model.
# ------------------------------------------------------------


# ============================================================
# STEP 4: THE WHOLE SETUP TAB
# ============================================================
def render(ctx):
    _master_block(ctx)
    _master_summary(ctx)
    st.divider()
    _model_block(ctx)
    st.divider()
    if ctx.has_master:
        st.success("Setup is complete. Go to the **Evening upload** tab and upload Day 1.")
    else:
        st.warning("Save the master file to unlock the next tabs.")

# ------------------------------------------------------------
# WHAT WE DID IN STEP 4:
# - Put the master file, its summary and the model choice on one page, and told the user the next move.
# ------------------------------------------------------------
