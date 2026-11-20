"""
tab_upload.py - TAB 2: EVENING UPLOAD
======================================
Upload tonight's closing-stock file: Day 1, then Day 2, ... up to Day 14 (one at a time or many at once).
Each file is checked and cleaned, and is compared with the master file. After a day is uploaded, open the
Daily decisions tab to see the recommendations for the next day.
"""
import pandas as pd
import streamlit as st

import config as C
from src import store
from ui.helpers import daily_file, ingest, show_df


# ============================================================
# STEP 1: THE UPLOAD BOX AND THE THREE WAYS TO LOAD DAYS
# ------------------------------------------------------------
#   1. Upload one or more files yourself
#   2. "Load next day from folder": takes the next missing DayNN file from DailyFiles-Retail
#   3. Demo shortcut: loads every missing day and approves the suggestions (tagged 'auto-demo')
# ============================================================
def _upload_block(ctx):
    st.subheader("Upload tonight's closing stock")
    st.write("In a real store this file arrives automatically from the inventory system each evening. "
             "Here you upload it yourself (Day 1, then Day 2, ... up to Day 14). "
             "After each upload, open the **Daily decisions** tab to see what to do next.")
    files = st.file_uploader("Choose one or more daily files (CSV)", type="csv", accept_multiple_files=True)
    c1, c2, c3 = st.columns(3)
    if c1.button("Check, clean and score the file(s)", disabled=not files):
        for f in files:
            try:
                ok, msg = ingest(pd.read_csv(f), f.name, ctx.master)
            except Exception as ex:                                   # unreadable file
                ok, msg = False, f"{f.name}: could not be read ({ex})"
            st.session_state["flash"].append(("success" if ok else "error", msg))
        st.rerun()

    days = ctx.days
    nxt = next((d for d in range(1, C.N_DAYS + 1) if d not in days), None)
    if c2.button(f"Load next day from folder (Day {nxt})" if nxt else "All 14 days loaded", disabled=nxt is None):
        p = daily_file(nxt)
        if p is None:
            st.session_state["flash"].append(("error", f"Day {nxt} file not found. Run: python src/make_daily_files.py"))
        else:
            ok, msg = ingest(pd.read_csv(p), p.name, ctx.master)
            st.session_state["flash"].append(("success" if ok else "error", msg))
        st.rerun()
    with c3:
        st.caption("Demo shortcut")
        if st.button("Load ALL remaining days + auto-approve", disabled=nxt is None,
                     help="Loads every missing day and approves the suggestions on each day's list. "
                          "These decisions are tagged 'auto-demo' - not a real manager."):
            for d in [d for d in range(1, C.N_DAYS + 1) if d not in days]:
                p = daily_file(d)
                if p is not None:
                    ingest(pd.read_csv(p), p.name, ctx.master)
            fresh = store.load_scores(ctx.active, ctx.rules_cfg)
            for d in store.uploaded_days():
                if store.load_decisions().query("day_no == @d").empty:
                    store.save_decisions(store.auto_approve(fresh[fresh.day_no == d], 15, ctx.rules_cfg))
            st.session_state["flash"].append(("success", "All days loaded. Suggestions were auto-approved (tagged 'auto-demo')."))
            st.rerun()

# ------------------------------------------------------------
# WHAT WE DID IN STEP 1:
# - Kept the three ways to load days exactly as before; every upload now also checks the day against the master file.
# ------------------------------------------------------------


# ============================================================
# STEP 2: WHICH DAYS ARE IN, AND THE UPLOAD HISTORY
# ============================================================
def _status_block(ctx):
    days, scores, have_data = ctx.days, ctx.scores, ctx.have_data
    st.progress(min(len(days) / C.N_DAYS, 1.0), text=f"{len(days)} of {C.N_DAYS} days uploaded")
    st.markdown("**Days uploaded**")
    rows = []
    for d in range(1, C.N_DAYS + 1):
        n = int(len(scores[scores.day_no == d])) if have_data else 0
        rows.append({"Day": d, "Date": (pd.Timestamp(C.DEMO_START) + pd.Timedelta(days=d - 1)).strftime("%a %d %b"),
                     "Status": "Uploaded" if d in days else "Waiting", "Batches on shelf": n or "",
                     "Delivery": "Sunday delivery" if d in C.DELIVERY_SUNDAYS else
                                 "Wednesday top-up" if d in C.DELIVERY_WEDNESDAYS else ""})
    show_df(pd.DataFrame(rows))
    st.markdown("**What the checks look for:** all columns present, one day per file, this store only, no duplicate "
                "batches, no negative quantities, no stock left above stock received, and agreement with the master file.")
    log = store.read_upload_log()
    if not log.empty:
        st.markdown("**Upload history and data-quality notes**")
        show_df(log.sort_values("uploaded_at", ascending=False))

# ------------------------------------------------------------
# WHAT WE DID IN STEP 2:
# - Added a progress bar (x of 14 days) above the existing day-by-day status table.
# ------------------------------------------------------------


# ============================================================
# STEP 3: START AGAIN
# ------------------------------------------------------------
# Clears the uploaded days, the decisions and the Power BI files. The master file and the
# model choice are kept (remove the master file on the Setup tab if you want a different one).
# ============================================================
def _reset_block():
    with st.expander("Start again (clears uploads, decisions and Power BI files)"):
        if st.checkbox("Yes, I want to clear everything") and st.button("Reset"):
            store.reset_all()
            st.session_state["flash"].append(("success", "Everything was cleared."))
            st.rerun()

# ------------------------------------------------------------
# WHAT WE DID IN STEP 3:
# - Kept the reset exactly as it was.
# ------------------------------------------------------------


# ============================================================
# STEP 4: THE WHOLE UPLOAD TAB
# ------------------------------------------------------------
# Locked until the master file is saved on the Setup tab.
# ============================================================
def render(ctx):
    if not ctx.has_master:
        st.info("Start on the **Setup** tab: save the master file first. Then come back here to upload Day 1.")
        return
    _upload_block(ctx)
    _status_block(ctx)
    _reset_block()

# ------------------------------------------------------------
# WHAT WE DID IN STEP 4:
# - This tab now waits for the master file, as requested.
# ------------------------------------------------------------
