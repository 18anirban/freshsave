"""
tab_setup.py - TAB 1: SETUP  (master file + model choice)
==========================================================
Three boxes, in this order (each one is a separate box on the screen):
  1. TRAIN: upload the master Kaggle data, click Train. Both models learn from it, live, inside the app.
  2. MODEL: choose which trained model scores the batches: KNN or Decision Tree.
  3. MASTER FILE: upload the fortnight's batch list (delivery date, expiry date, quantity, prices).
Only after the master file is saved do the other tabs open for use.
"""
import json
import time

import pandas as pd
import streamlit as st

import config as C
from src import features as F, master as M, scoring
from src import train as T
from ui.helpers import read_json, show_df, money

MODELS = list(scoring.MODEL_FILES)
MODEL_HELP = {
    "Decision Tree": "Learns a list of yes/no questions (for example: is 2 or fewer days left? has less than half "
                     "been sold?) and follows them to an answer. Easy to read and explain: the rules can be printed.",
    "KNN": "Compares tonight's batch with the most similar past batch-evenings and lets them vote. "
           "No rules to read, but easy to explain with examples: you can show the neighbours that voted.",
}


# ============================================================
# STEP 0: BOX 1 - UPLOAD THE MASTER KAGGLE DATA AND TRAIN THE MODELS (LIVE)
# ------------------------------------------------------------
# This is where the models are BUILT. The uploaded file is the history the models learn from.
# It runs the same code as "python src/train.py" and shows the progress on the screen.
# ============================================================
def _run_training(raw, source):
    lines = []
    with st.status(f"Training both models on {source} ...", expanded=True) as status:
        def progress(msg):
            lines.append(msg)
            st.text(msg)
        try:
            T.main(raw=raw, progress=progress)
        except Exception as ex:
            status.update(label="Training stopped", state="error")
            st.session_state["flash"].append(("error", f"Training failed: {ex}"))
            return
        scoring.clear_model_cache()
        status.update(label="Training finished", state="complete")
    st.session_state["flash"].append(("success", "Both models are trained and saved. Choose the model in Box 2."))
    st.rerun()


def _train_block():
    with st.container(border=True):
        st.subheader("Box 1. Upload the master Kaggle data and train the models")
        st.write("The Kaggle history file (every store, every batch, how much sold and how much was wasted) is what "
                 "the models **learn from**. Upload it, click Train, and both models are built and tested here, live.")
        file = st.file_uploader("Master Kaggle data (CSV, perishable_goods_management.csv)", type="csv", key="kaggle_upload")
        c1, c2 = st.columns(2)
        if c1.button("Train both models on the uploaded file", type="primary", disabled=file is None):
            try:
                raw = pd.read_csv(file)
            except Exception as ex:
                st.session_state["flash"].append(("error", f"{file.name}: could not be read ({ex})"))
                st.rerun()
            problems = F.validate_raw(raw)
            if problems:
                st.session_state["flash"].append(("error", f"{file.name}: " + " ".join(problems)))
                st.rerun()
            _run_training(raw, file.name)
        if c2.button("Train on the file already in data/raw", disabled=not C.RAW_CSV.exists(),
                     help="data/raw/perishable_goods_management.csv"):
            _run_training(None, C.RAW_CSV.name)
        p = C.MODELS_DIR / "trained_on.json"
        if p.exists():
            info = json.load(open(p))
            st.success(f"Models are trained: {info['rows']:,} rows from {info['stores']} stores, "
                       f"on {info['trained_at']} (took {info['seconds']} s). Upload a different file to train again.")
        elif (C.MODELS_DIR / "active_model.json").exists():
            st.success("Models are trained (by python src/train.py). Upload a file here to train them again.")
        else:
            st.warning("No trained models yet. Upload the Kaggle file and click Train. It takes a few minutes.")

# ------------------------------------------------------------
# WHAT WE DID IN STEP 0:
# - Added a separate box with a file uploader and a Train button.
# - The uploaded file is checked (right columns, enough rows, dates before and after the split) and then
#   src/train.py trains and tests both models, writing each step to the screen.
# - After training the app forgets the old models in memory so the new ones are used.
# ------------------------------------------------------------


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
    st.subheader("Box 3. Upload the fortnight's batch list (master file)")
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
    st.subheader("Box 2. Choose the model")
    info = read_json("active_model.json")
    current = info["active"] if info["active"] in MODELS else MODELS[0]
    st.selectbox("Model that scores the batches", MODELS, index=MODELS.index(current), key="model_pick", on_change=_switch)
    st.info(MODEL_HELP[ctx.active])
    st.success(f"Default winner on the 2024 test data: **{info['winner']}**. {info['why']}")
    st.caption("Open the Model comparison tab for the full test results of both models.")
    with st.expander("Where is the model used? What does it do?"):
        st.markdown(
            "- **Built** in Box 1: it learns from the Kaggle history which batches ended up wasted.\n"
            "- **Saved** as a file in the `models/` folder of this project on this computer.\n"
            "- **Used** every time a day file is uploaded (Tab 2): it reads each batch's days left, share sold and selling speed, "
            "and gives a **risk** (chance that 20%+ of the batch is wasted). Tab 3 turns that risk into an action.\n"
            "- **Not a server:** nothing is deployed online. The app loads the file from disk and runs it on this machine.")
# ------------------------------------------------------------
# WHAT WE DID IN STEP 3:
# - Moved the model choice here from the Model comparison tab, with a plain explanation of each model.
# ------------------------------------------------------------


# ============================================================
# STEP 4: THE WHOLE SETUP TAB
# ============================================================
def render_untrained():
    """Shown instead of the whole app until the models exist."""
    _train_block()


def render(ctx):
    _train_block()
    with st.container(border=True):
        _model_block(ctx)
    with st.container(border=True):
        _master_block(ctx)
        _master_summary(ctx)
    if ctx.has_master:
        st.success("Setup is complete. Go to the **Evening upload** tab and upload Day 1.")
    else:
        st.warning("Save the master file (Box 3) to unlock the next tabs.")

# ------------------------------------------------------------
# WHAT WE DID IN STEP 4:
# - Three separate boxes: train, choose the model, master file. Then told the user the next move.
# - render_untrained() shows only Box 1, so the app can start before any model exists.
# ------------------------------------------------------------
