"""
tab_dashboard.py - TAB 4: MANAGER DASHBOARD  (opens when all 14 days are uploaded)
===================================================================================
Answers the manager's two questions after the fortnight:
  1. How much did we waste, and did the predictions come true?
  2. How much should we order for the next 15 days?
Waste is real (from the uploaded files). Anything marked "estimated" uses the assumed sales response to a
markdown (see the sidebar). Nothing here is a measured saving.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

import config as C
from src import order_plan
from ui.helpers import money, show_df, show_fig


# ============================================================
# STEP 1: THE WAIT SCREEN (LOCK UNTIL ALL 14 DAYS ARE IN)
# ------------------------------------------------------------
# The dashboard is meant for the end of the fortnight. Before that we show how many days are missing.
# A tick-box lets you preview it earlier (useful for testing and for the demo).
# ============================================================
def _gate(ctx):
    n = len(ctx.days)
    if n >= C.N_DAYS:
        return True
    st.info(f"The manager dashboard opens when all {C.N_DAYS} days are uploaded. "
            f"So far: **{n} of {C.N_DAYS}**. Keep uploading in the Evening upload tab.")
    st.progress(n / C.N_DAYS, text=f"{n} of {C.N_DAYS} days uploaded")
    return st.checkbox(f"Show a preview with the {n} day(s) uploaded so far", key="dash_preview")

# ------------------------------------------------------------
# WHAT WE DID IN STEP 1:
# - Returned True when 14 of 14 days are in; otherwise showed a progress bar and a preview tick-box.
# ------------------------------------------------------------


# ============================================================
# STEP 2: HEADLINE NUMBERS AND THE WASTE SECTION
# ------------------------------------------------------------
# Real numbers: batches finished, actual waste cost, waste % of stock, decisions saved.
# Estimated numbers: waste avoided, revenue recovered, discount given away, net cash effect.
# ============================================================
def _kpis(ctx, o, dec):
    k = st.columns(5)
    k[0].metric("Days uploaded", len(ctx.days))
    k[1].metric("Batches finished", len(o))
    k[2].metric("Actual waste cost", money(o.waste_cost.sum()) if len(o) else "$0")
    k[3].metric("Waste % of stock", f"{o.units_wasted.sum() / max(o.qty_received.sum(), 1):.1%}" if len(o) else "-")
    k[4].metric("Decisions saved", len(dec))


def _waste_section(o, ef):
    avoided = ef.waste_cost_avoided.sum() if len(ef) else 0
    given = ef.margin_given_away.sum() if len(ef) else 0
    rec = ef.revenue_recovered.sum() if len(ef) else 0
    st.subheader("Waste: actual vs estimated with your decisions")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Waste cost avoided (estimated)", money(avoided))
    e2.metric("Revenue recovered (estimated)", money(rec))
    e3.metric("Discount given away (estimated)", money(given))
    e4.metric("Net cash effect (estimated)", money(rec - given))
    if len(ef) and rec - given < 0:
        st.info("With the current assumption, markdowns **reduce waste but cost more margin than they recover**. "
                "Markdowns pay for themselves only when shoppers respond strongly to a discount. "
                "Move the 'Sales response' slider in the sidebar to see how this changes "
                "(it needs about 3 here: 20% off giving roughly +60% sales).")
    byday = o.groupby("closed_day").waste_cost.sum().rename("Actual (baseline)").to_frame()
    if len(ef):
        byday["Estimated with decisions"] = byday["Actual (baseline)"] - ef.groupby("closed_day").waste_cost_avoided.sum().reindex(byday.index).fillna(0)
    else:
        byday["Estimated with decisions"] = byday["Actual (baseline)"]
    byday = byday.reset_index().melt("closed_day", var_name="Version", value_name="Waste cost")
    show_fig(px.bar(byday, x="closed_day", y="Waste cost", color="Version", barmode="group",
                    labels={"closed_day": "Day the batch finished"}, title="Waste cost by day"))
    cat = o.groupby("category").waste_cost.sum().rename("Actual (baseline)").to_frame()
    cat["Estimated with decisions"] = cat["Actual (baseline)"] - (ef.groupby("category").waste_cost_avoided.sum().reindex(cat.index).fillna(0) if len(ef) else 0)
    cat = cat.reset_index().melt("category", var_name="Version", value_name="Waste cost")
    show_fig(px.bar(cat, x="category", y="Waste cost", color="Version", barmode="group", title="Waste cost by category"))

# ------------------------------------------------------------
# WHAT WE DID IN STEP 2:
# - Kept the headline numbers and both waste charts exactly as before.
# ------------------------------------------------------------


# ============================================================
# STEP 3: ORDER PLAN FOR THE NEXT 15 DAYS  (NEW)
# ------------------------------------------------------------
# For every product: how much to order, in how many deliveries, and what it costs.
#   order = 15 days of expected sales + 5% buffer - stock on the shelf that will still sell in time
#   products the model keeps flagging (or that really wasted 20%+) arrive in smaller, more frequent deliveries
# The whole calculation is in src/order_plan.py; this block only displays it.
# ============================================================
def _order_plan_section(ctx):
    plan = ctx.data["plan"]
    st.subheader(f"Recommended order for the next {C.PLAN_DAYS} days")
    t = order_plan.plan_totals(plan)
    a, b, c, d = st.columns(4)
    a.metric("Total order cost", money(t["cost"]))
    b.metric("Units to order", f"{t['units']:,}")
    c.metric("Deliveries", t["deliveries"])
    d.metric("High-risk products", f"{t['high_risk']} of {t['products']}")
    table = order_plan.display_plan(plan)
    show_df(table)
    st.download_button("Download the order plan (CSV)", table.to_csv(index=False), "order_plan_next_15_days.csv")
    with st.expander("How this is calculated"):
        st.markdown(f"""
- **Sold per day** = units sold in the uploaded days / number of days uploaded, per product.
- **Order for {C.PLAN_DAYS} days** = sold per day x {C.PLAN_DAYS} x {1 + C.PLAN_SAFETY_BUFFER:.2f} (a {C.PLAN_SAFETY_BUFFER:.0%} safety buffer), minus the stock on the shelf that will still sell before it expires.
- **High-risk product** = the model ({ctx.active}) flagged at least {C.PLAN_HIGH_RISK_SHARE:.0%} of its shelf-evenings, or it really wasted {C.PLAN_HIGH_WASTE_PCT:.0f}% or more of what it received.
- **Largest safe delivery** = sold per day x batch life x {C.PLAN_FILL_LOW_RISK:.0%} ({C.PLAN_FILL_HIGH_RISK:.0%} for high-risk products). The order is split into as many deliveries as needed to stay under it.
- Batch life comes from the master file (expiry date - delivery date + 1).
- Assumption: sales in the next {C.PLAN_DAYS} days continue at the speed of the last fortnight. Seasonal changes are not modelled.
""")

# ------------------------------------------------------------
# WHAT WE DID IN STEP 3:
# - Showed total cost, units, deliveries and high-risk products, then the product-by-product plan with a CSV download.
# - Explained the formula in plain words, so the manager (and sir) can check the numbers.
# ------------------------------------------------------------


# ============================================================
# STEP 4: CAN WE TRUST THE PREDICTIONS? (PREDICTED VS ACTUAL)
# ------------------------------------------------------------
# Each finished batch is compared with the risk it showed on its FIRST evening (the honest test)
# and on its last evening before expiry (easier, because most of the answer is already visible).
# ============================================================
def _prediction_section(ctx, o):
    r = ctx.rules_cfg
    st.subheader("Can we trust the predictions? (predicted vs actual)")
    st.caption("Each finished batch is compared with the risk it showed on its FIRST evening (the honest test) "
               "and on its last evening before expiry (easier, because most of the answer is already visible).")
    m1, m2 = st.columns(2)
    for col, key, title in [(m1, "risk_first_evening", "First evening"), (m2, "risk_last_evening", "Last evening")]:
        x = o.dropna(subset=[key])
        pred = x[key] >= r["low"]
        act = x.actually_at_risk == 1
        with col:
            st.markdown(f"**{title}** ({len(x)} batches)")
            if len(x):
                st.write(f"Accuracy **{(pred == act).mean():.0%}** | caught **{int((pred & act).sum())}** of "
                         f"**{int(act.sum())}** really at-risk batches | false alarms **{int((pred & ~act).sum())}**")
                x = x.assign(bin=pd.cut(x[key], [0, .2, .4, .6, .8, 1.0], include_lowest=True).astype(str))
                cb = x.groupby("bin").agg(predicted=(key, "mean"), actual=("actually_at_risk", "mean"), n=(key, "size")).reset_index()
                show_fig(px.bar(cb.melt("bin", ["predicted", "actual"]), x="bin", y="value", color="variable", barmode="group",
                                labels={"bin": "Risk band", "value": "Share at risk"}, title=title))

# ------------------------------------------------------------
# WHAT WE DID IN STEP 4:
# - Kept the predicted-vs-actual comparison as before.
# ------------------------------------------------------------


# ============================================================
# STEP 5: MANAGER DECISIONS AND ACTIONS PER DAY
# ------------------------------------------------------------
# Override rate (how often the manager changed or rejected the suggestion), who was right on rejections,
# and a chart of actions per day.
# ============================================================
def _decisions_section(ctx, o, dec):
    st.subheader("Manager decisions: who was right?")
    if dec.empty:
        st.write("No decisions saved yet.")
    else:
        auto = int((dec.decided_by == "auto-demo").sum())
        manual = len(dec) - auto
        d1, d2, d3 = st.columns(3)
        over = dec[dec.decision.isin(["Change", "Reject"])]
        d1.metric("Override rate", f"{len(over) / len(dec):.0%}", f"{len(over)} of {len(dec)} decisions", delta_color="off")
        d2.metric("Manual decisions", manual)
        d3.metric("Auto-approved (demo shortcut)", auto)
        rj = dec[dec.decision == "Reject"].merge(o[["batch_id", "actually_at_risk"]], on="batch_id")
        if len(rj):
            st.write(f"Of **{len(rj)}** rejected suggestions, the batch really ended up at risk **{int(rj.actually_at_risk.sum())}** "
                     f"times (the model was right), and not at risk **{int((rj.actually_at_risk == 0).sum())}** times (the manager was right).")
        else:
            st.write("No suggestion has been rejected and finished yet.")

    sm = ctx.data["summary"]
    st.subheader("Actions taken per day")
    show_fig(px.bar(sm.melt("day_no", ["at_risk_flagged", "reposition_flags", "discard_tonight"]), x="day_no", y="value",
                    color="variable", barmode="group", labels={"day_no": "Day", "value": "Batches", "variable": ""}))
    st.download_button("Download daily summary (CSV)", sm.to_csv(index=False), "daily_summary.csv")

# ------------------------------------------------------------
# WHAT WE DID IN STEP 5:
# - Kept the override rate, the "who was right" sentence and the actions-per-day chart.
# ------------------------------------------------------------


# ============================================================
# STEP 6: THE WHOLE DASHBOARD TAB
# ------------------------------------------------------------
# Order on screen: warning, headline numbers, waste, ORDER PLAN, prediction check, decisions.
# The order plan sits right after the waste, because that is the manager's next decision.
# ============================================================
def render(ctx):
    if not ctx.has_master:
        st.info("Start on the **Setup** tab: save the master file first.")
        return
    if not ctx.have_data:
        st.info("Upload a day in the Evening upload tab. This page fills in as days arrive.")
        return
    if not _gate(ctx):
        return

    o, dec, ef = ctx.data["outcomes"], ctx.data["decisions"], ctx.data["effects"]
    st.warning("**Read this first.** Waste, flags and prediction accuracy come from the uploaded files. "
               "The daily sales path inside each batch is SIMULATED, and every figure marked *estimated* uses the "
               "assumed sales response to a markdown. Nothing here is a measured saving.")
    _kpis(ctx, o, dec)
    if o.empty:
        st.info("Batches finish when their days left reach 0. Upload more days to see waste results.")
    else:
        _waste_section(o, ef)
    _order_plan_section(ctx)
    if not o.empty:
        _prediction_section(ctx, o)
        _decisions_section(ctx, o, dec)
    st.caption("These tables are also saved for Power BI in the powerbi_data folder after every change.")

# ------------------------------------------------------------
# WHAT WE DID IN STEP 6:
# - Waited for the master file and for all 14 days (or the preview tick-box).
# - Put the 15-day order plan right after the waste section.
# ------------------------------------------------------------
