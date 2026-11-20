"""
step_bar.py - THE HORIZONTAL STEP STRIP AT THE TOP
===================================================
    Step 1  Setup                      master file + model choice
    Step 2  Upload day files           Day 1 ... Day 14
    Step 3  Model and recommendations  what to do for the next day
    Step 4  Manager dashboard          waste + order for the next 15 days
Green = done, blue = where you are now, grey = still to come.
"""
import streamlit as st

import config as C


# ============================================================
# STEP 1: DECIDE WHICH STEP IS DONE, CURRENT OR WAITING
# ------------------------------------------------------------
#   no master file            -> Step 1 is current
#   master saved, no day yet  -> Step 2 is current
#   1 to 13 days uploaded     -> Step 3 is current (each new day gives new recommendations)
#   all 14 days uploaded      -> Step 4 is current
# ============================================================
def step_states(has_master: bool, n_days: int):
    all_in = n_days >= C.N_DAYS
    done = [has_master, n_days >= 1, all_in, False]
    current = 1 if not has_master else 2 if n_days == 0 else 3 if not all_in else 4
    notes = ["Master file + model",
             f"{n_days} of {C.N_DAYS} days uploaded",
             "Plan for the next day" if n_days else "Starts after Day 1",
             "Waste + 15-day order" if all_in else f"Opens after all {C.N_DAYS} days"]
    return done, current, notes

# ------------------------------------------------------------
# WHAT WE DID IN STEP 1:
# - Worked out from two facts (master saved? how many days?) which step is current.
# ------------------------------------------------------------


# ============================================================
# STEP 2: DRAW THE FOUR BOXES
# ============================================================
TITLES = ["Setup", "Upload day files", "Model & recommendations", "Manager dashboard"]


def render_step_bar(has_master: bool, n_days: int):
    done, current, notes = step_states(has_master, n_days)
    boxes = []
    for i, (title, note) in enumerate(zip(TITLES, notes), start=1):
        if done[i - 1] and i != current:
            bg, fg, mark = "#1b8a5a", "#ffffff", "&#10003;"
        elif i == current:
            bg, fg, mark = "#2563eb", "#ffffff", str(i)
        else:
            bg, fg, mark = "rgba(128,128,128,0.18)", "inherit", str(i)
        boxes.append(f"<div style='flex:1;min-width:150px;padding:10px 12px;border-radius:10px;background:{bg};color:{fg};'>"
                     f"<div style='font-size:12px;opacity:0.85;'>STEP {mark}</div>"
                     f"<div style='font-size:15px;font-weight:600;'>{title}</div>"
                     f"<div style='font-size:12px;opacity:0.85;'>{note}</div></div>")
    st.markdown("<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px;'>" + "".join(boxes) + "</div>",
                unsafe_allow_html=True)

# ------------------------------------------------------------
# WHAT WE DID IN STEP 2:
# - Drew four boxes in one row: green with a tick when done, blue for the current step, grey for the rest.
# - Only colours and text; the logic is in step_states().
# ------------------------------------------------------------
