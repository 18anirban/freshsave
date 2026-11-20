"""
tab_decisions.py - TAB 3: DAILY DECISIONS  (for the store manager)
===================================================================
For any uploaded day the model has scored every batch. This tab shows the batches that need attention, the
suggested action, and lets the manager Approve, Change or Reject each one. The system suggests, the manager decides.
"""
import pandas as pd
import streamlit as st

import config as C
from src import rules, store
from ui.helpers import money, show_df

LABELS = {"Approve": "Approve", "Change": "Change", "Reject": "Reject"}


# ============================================================
# STEP 1: THE FIVE HEADLINE NUMBERS OF THE DAY
# ------------------------------------------------------------
# batches on shelf, how many are at risk, money at stake (chance of waste x units left x cost),
# what must be discarded tonight, and the expected waste in units.
# ============================================================
def _summary_cards(g, live, flagged, disc):
    a, b, c, d_, e = st.columns(5)
    a.metric("Batches on shelf", len(g))
    b.metric("At risk", len(flagged))
    c.metric("Money at stake", money(flagged.money_at_stake.sum()))
    d_.metric("Discard tonight", len(disc), money(disc.discard_cost.sum()) + " lost", delta_color="off")
    e.metric("Expected waste (units)", f"{int((live.risk * live.qty_left).sum()):,}")
    st.caption("Money at stake = chance of waste x units left x cost price. The model suggests; **you decide**.")

# ------------------------------------------------------------
# WHAT WE DID IN STEP 1:
# - Showed five numbers for the chosen day, calculated from the model's risk for each batch.
# ------------------------------------------------------------


# ============================================================
# STEP 2: THE ACTION LIST WITH APPROVE / CHANGE / REJECT
# ------------------------------------------------------------
# The top batches by money at stake, with the reason in plain words and the suggested price.
# The manager edits the Decision column (and the markdown % when choosing Change), then saves.
# Saved decisions are written to state/decisions.csv and to the Power BI files.
# ============================================================
def _action_list(ctx, day, g, n_show):
    r = ctx.rules_cfg
    st.subheader("Today's action list")
    acts = rules.action_list(g, n_show, r)
    if acts.empty:
        st.success("Nothing is above the risk threshold today.")
        return
    saved = ctx.data["decisions"]
    saved = saved[saved.day_no == day].set_index("batch_id") if not saved.empty else pd.DataFrame()
    tbl = pd.DataFrame({
        "Batch": acts.batch_id, "Product": acts.product_name, "Category": acts.category,
        "Left": acts.qty_left, "Days left": acts.days_left, "Risk %": (acts.risk * 100).round(0),
        "Why": acts.reason, "Suggested": acts.action, "New price": acts.new_price,
        "Money at stake": acts.money_at_stake,
        "Decision": [saved.loc[b, "decision"] if len(saved) and b in saved.index else "Approve" for b in acts.batch_id],
        "Your markdown %": [saved.loc[b, "final_markdown"] if len(saved) and b in saved.index else m
                            for b, m in zip(acts.batch_id, acts.suggested_markdown)]})
    edited = st.data_editor(
        tbl, hide_index=True, key=f"ed_{day}_{n_show}_{ctx.active}_{r['low']}_{r['high']}_{r['md_mid']}_{r['md_high']}",
        disabled=[c for c in tbl.columns if c not in ("Decision", "Your markdown %")],
        column_config={
            "Decision": st.column_config.SelectboxColumn("Decision", options=list(LABELS), required=True),
            "Your markdown %": st.column_config.NumberColumn("Your markdown %", min_value=0, max_value=90, step=5,
                                                              help="Used only when Decision = Change"),
            "Risk %": st.column_config.ProgressColumn("Risk %", min_value=0, max_value=100, format="%d"),
            "Money at stake": st.column_config.NumberColumn(format="$%.0f"),
            "New price": st.column_config.NumberColumn(format="$%.2f"),
            "Why": st.column_config.TextColumn(width="large")})
    st.caption("Approve = follow the suggestion | Change = use your own markdown | Reject = leave the price as it is.")
    if st.button("Save my decisions for this day", type="primary"):
        final = [s if d == "Approve" else (0 if d == "Reject" else m)
                 for d, s, m in zip(edited["Decision"], acts.suggested_markdown, edited["Your markdown %"])]
        repo = [bool(r_) if d == "Approve" else (False if d == "Reject" else bool(r_))
                for d, r_ in zip(edited["Decision"], acts.reposition)]
        store.save_decisions(store.make_decision_rows(acts, edited["Decision"], final, repo))
        st.session_state["flash"].append(("success", f"Saved {len(acts)} decisions for Day {day}."))
        st.rerun()
    n_repo = int(acts.reposition.sum())
    if n_repo:
        st.info(f"{n_repo} batch(es) are suggested for the front display / eye level / a bundle.")

# ------------------------------------------------------------
# WHAT WE DID IN STEP 2:
# - Kept the editable table and the Save button exactly as before.
# - The only change is that the settings now come from ctx (the sidebar values), not from loose variables.
# ------------------------------------------------------------


# ============================================================
# STEP 3: BATCHES TO DISCARD TONIGHT
# ============================================================
def _discard_block(disc):
    with st.expander(f"Discard tonight ({len(disc)} batches, {money(disc.discard_cost.sum())} of stock)"):
        show_df(disc[["batch_id", "product_name", "category", "qty_left", "cost_price", "discard_cost"]]
                .sort_values("discard_cost", ascending=False)
                .rename(columns={"qty_left": "Units to discard", "discard_cost": "Cost lost"}))

# ------------------------------------------------------------
# WHAT WE DID IN STEP 3:
# - Listed the batches that expire tonight and what they cost.
# ------------------------------------------------------------


# ============================================================
# STEP 4: SCORECARD - WHAT HAPPENED UP TO THIS DAY
# ------------------------------------------------------------
# Left: batches that finished tonight (were they flagged on arrival? were they really wasted?).
# Right: what the manager decided yesterday and the estimated waste avoided.
# ============================================================
def _scorecard(ctx, day):
    data, r = ctx.data, ctx.rules_cfg
    if day < 2 or data["outcomes"].empty:
        return
    st.subheader(f"Scorecard: what happened up to Day {day}")
    o = data["outcomes"]
    closed_now = o[o.closed_day == day]
    dec_prev = data["decisions"][data["decisions"].day_no == day - 1] if not data["decisions"].empty else pd.DataFrame()
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(f"**Batches that finished tonight ({len(closed_now)})**")
        if len(closed_now):
            f_ = closed_now.risk_first_evening >= r["low"]
            t_ = closed_now.actually_at_risk == 1
            st.write(f"Flagged on arrival and really wasted: **{int((f_ & t_).sum())}**  |  "
                     f"Missed: **{int((~f_ & t_).sum())}**  |  False alarms: **{int((f_ & ~t_).sum())}**")
            st.write(f"Waste cost tonight: **{money(closed_now.waste_cost.sum())}**")
        else:
            st.write("No batch finished tonight.")
    with s2:
        st.markdown("**Yesterday's decisions**")
        if len(dec_prev):
            st.write(f"Approved **{int((dec_prev.decision == 'Approve').sum())}**, changed "
                     f"**{int((dec_prev.decision == 'Change').sum())}**, rejected **{int((dec_prev.decision == 'Reject').sum())}**.")
            ef = data["effects"]
            ef = ef[ef.day_no == day - 1] if not ef.empty else ef
            if len(ef):
                st.write(f"Estimated waste cost avoided so far: **{money(ef.waste_cost_avoided.sum())}** "
                         f"(assumption-based; batches still on the shelf are not counted yet).")
        else:
            st.write("No decisions were saved for yesterday.")

# ------------------------------------------------------------
# WHAT WE DID IN STEP 4:
# - Compared the first-evening risk of finished batches with what really happened (hits, misses, false alarms).
# ------------------------------------------------------------


# ============================================================
# STEP 5: THE WHOLE DECISIONS TAB
# ------------------------------------------------------------
# Pick a day (default: the latest uploaded) and how many batches to show, then the four blocks above.
# ============================================================
def render(ctx):
    if not ctx.has_master:
        st.info("Start on the **Setup** tab: save the master file first.")
        return
    if not ctx.have_data:
        st.info("Upload a day in the Evening upload tab to see the manager's list.")
        return

    days, scores, r = ctx.days, ctx.scores, ctx.rules_cfg
    top_l, top_r = st.columns([1, 1])
    day = top_l.selectbox("Day", days, index=len(days) - 1,
                          format_func=lambda d: f"Day {d} - {(pd.Timestamp(C.DEMO_START) + pd.Timedelta(days=d-1)).strftime('%A %d %b')}")
    n_show = top_r.slider("How many batches to show", 10, 15, 15)
    g = scores[scores.day_no == day]
    live = g[g.days_left > 0]
    flagged = live[live.risk >= r["low"]]
    disc = g[g.days_left == 0]

    _summary_cards(g, live, flagged, disc)
    _action_list(ctx, day, g, n_show)
    _discard_block(disc)
    _scorecard(ctx, day)

# ------------------------------------------------------------
# WHAT WE DID IN STEP 5:
# - Same screen as before, now one function per block. It waits for the master file and for at least one day.
# ------------------------------------------------------------
