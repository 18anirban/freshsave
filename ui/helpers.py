"""
helpers.py - small things every tab needs
==========================================
Kept in one place so the tab files stay short and readable.
"""
import json
from pathlib import Path

import streamlit as st

import config as C
from src import scoring, store
from src import master as M
from src.models import MODEL_FILES


# ============================================================
# STEP 1: SHOW A TABLE OR A CHART
# ------------------------------------------------------------
# Streamlit changed the name of the "full width" option between versions. These two functions
# try the new name first and fall back to the old one, so the app runs on old and new versions.
# ============================================================
def show_fig(fig):
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def show_df(df, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def money(x):
    return f"${x:,.0f}"

# ------------------------------------------------------------
# WHAT WE DID IN STEP 1:
# - Wrote one way to show tables and charts at full width, and one way to format money.
# ------------------------------------------------------------


# ============================================================
# STEP 2: FILES THE APP READS
# ------------------------------------------------------------
# models_ready : True only when train.py has produced every file the app needs
# read_json    : read one of the small files train.py saved in models/
# daily_file   : find DayNN_*.csv in the project folder (or the Windows Downloads folder)
# ============================================================
def models_ready():
    need = ["active_model.json", *MODEL_FILES.values(), "feature_names.json", "data_profile.json",
            "metrics_summary.csv", "curves.json", "test_predictions.csv.gz"]
    return all((C.MODELS_DIR / n).exists() for n in need)


def read_json(name):
    return json.load(open(C.MODELS_DIR / name))


def daily_file(day):
    for folder in (C.DAILY_DIR, C.WINDOWS_DAILY_DIR):
        try:
            hits = sorted(Path(folder).glob(f"Day{day:02d}_*.csv"))
        except OSError:
            hits = []
        if hits:
            return hits[0]
    return None

# ------------------------------------------------------------
# WHAT WE DID IN STEP 2:
# - models_ready now looks for the model files of every model in the registry (KNN and Decision Tree).
# ------------------------------------------------------------


# ============================================================
# STEP 3: INGEST ONE DAILY FILE
# ------------------------------------------------------------
# validate -> clean -> compare with the master file -> save -> log.
# Returns (ok, message) for the green / red message on the screen.
# ============================================================
def ingest(df, filename, master=None):
    clean, errors, warnings = scoring.validate_and_clean(df)
    if errors:
        return False, f"{filename}: " + " ".join(errors)
    warnings = warnings + M.check_day_against_master(clean, master)
    day = store.save_upload(clean)
    store.log_upload(day, filename, len(clean), warnings)
    return True, f"{filename}: Day {day} saved ({len(clean)} batches)." + (f" Notes: {' '.join(warnings)}" if warnings else "")

# ------------------------------------------------------------
# WHAT WE DID IN STEP 3:
# - Same as before, plus one new check: the day's batches are compared with the master file
#   (missing batches, different quantity or expiry date). These are notes, not errors.
# ------------------------------------------------------------
