"""
order_plan.py - HOW MUCH TO ORDER FOR THE NEXT 15 DAYS  (Manager dashboard)
============================================================================
The manager's question after two weeks: "How much of each product should I order for the next
15 days, so that we neither run out nor waste?"

The model's job here: it flags products whose batches keep ending up at risk. Those products are
delivered in SMALLER, MORE FREQUENT batches, so less stock is left on the shelf when it expires.
Everything is per PRODUCT (not per batch), because next fortnight's batches do not exist yet.
"""
import math

import numpy as np
import pandas as pd

import config as C
from src import master as M


# ============================================================
# STEP 1: COLLECT THE FACTS PER PRODUCT
# ------------------------------------------------------------
#   avg_daily_sales : units sold per day, over the uploaded days
#   batch_life_days : how long a batch lasts (from the master file; if there is none, from the daily files)
#   cost_price      : average cost of one unit
#   wasted units / waste %: from batches that really finished (outcomes)
#   flagged_share   : share of the product's shelf-evenings that the model flagged as at risk
#   stock_now       : units still on the shelf after the last uploaded day (batches with days left)
# ============================================================
def product_facts(scores, outcomes, master, rules_cfg):
    n_days = max(int(scores["day_no"].nunique()), 1)
    facts = scores.groupby("product_name").agg(category=("category", "first"),
                                                units_sold=("units_sold_today", "sum")).reset_index()
    facts["avg_daily_sales"] = facts["units_sold"] / n_days

    source = M.with_batch_life(master) if master is not None and not master.empty else \
        M.with_batch_life(scores.drop_duplicates("batch_id")[M.MASTER_COLS])
    per_product = source.groupby("product_name").agg(batch_life_days=("batch_life_days", "median"),
                                                     cost_price=("cost_price", "mean")).reset_index()
    facts = facts.merge(per_product, on="product_name", how="left")
    facts["batch_life_days"] = facts["batch_life_days"].fillna(1).clip(lower=1)
    facts["cost_price"] = facts["cost_price"].fillna(0.0)

    if outcomes is not None and len(outcomes):
        w = outcomes.groupby("product_name").agg(wasted_units=("units_wasted", "sum"),
                                                 received=("qty_received", "sum")).reset_index()
        w["waste_pct"] = w["wasted_units"] / w["received"].clip(lower=1) * 100
        facts = facts.merge(w[["product_name", "wasted_units", "waste_pct"]], on="product_name", how="left")
    else:
        facts["wasted_units"], facts["waste_pct"] = 0.0, 0.0
    facts[["wasted_units", "waste_pct"]] = facts[["wasted_units", "waste_pct"]].fillna(0.0)

    live = scores[scores["days_left"] > 0]
    share = (live["risk"] >= rules_cfg["low"]).groupby(live["product_name"]).mean().rename("flagged_share")
    facts = facts.merge(share, on="product_name", how="left")
    facts["flagged_share"] = facts["flagged_share"].fillna(0.0)

    last = scores[(scores["day_no"] == scores["day_no"].max()) & (scores["days_left"] > 0)].copy()
    last["w"] = last["qty_left"] * last["days_left"]
    now = last.groupby("product_name").agg(stock_now=("qty_left", "sum"), w=("w", "sum")).reset_index()
    now["days_left_avg"] = now["w"] / now["stock_now"].clip(lower=1)
    facts = facts.merge(now[["product_name", "stock_now", "days_left_avg"]], on="product_name", how="left")
    facts[["stock_now", "days_left_avg"]] = facts[["stock_now", "days_left_avg"]].fillna(0.0)
    return facts

# ------------------------------------------------------------
# WHAT WE DID IN STEP 1:
# - Built one row per product with its sales speed, batch life, cost, real waste so far,
#   how often the model flagged it, and what is still on the shelf.
# ------------------------------------------------------------


# ============================================================
# STEP 2: HOW MUCH TO ORDER
# ------------------------------------------------------------
#   demand       = average daily sales x 15 days x 1.05  (5% safety buffer)
#   carry-over   = stock on the shelf that will still sell before it expires
#                = smaller of (stock now) and (daily sales x its days left)
#   order        = demand - carry-over   (never below 0)
# ============================================================
def order_quantity(facts: pd.DataFrame) -> pd.Series:
    demand = facts["avg_daily_sales"] * C.PLAN_DAYS * (1 + C.PLAN_SAFETY_BUFFER)
    carry = np.minimum(facts["stock_now"], facts["avg_daily_sales"] * facts["days_left_avg"])
    return np.ceil((demand - carry).clip(lower=0)).astype(int)

# ------------------------------------------------------------
# WHAT WE DID IN STEP 2:
# - Expected 15 days of sales plus 5%, minus the stock already on the shelf that will sell in time.
# ------------------------------------------------------------


# ============================================================
# STEP 3: SPLIT THE ORDER INTO SAFE DELIVERIES
# ------------------------------------------------------------
# One huge delivery of a short-life product is exactly what creates waste.
#   largest safe delivery = daily sales x batch life x fill share
#   fill share = 90% for safe products, 70% for high-risk products
# A product is HIGH RISK when the model flagged at least half of its shelf-evenings,
# or when it really wasted 20% or more of what it received.
# deliveries = order / largest safe delivery, rounded up.
# ============================================================
def build_order_plan(scores, outcomes, master, rules_cfg) -> pd.DataFrame:
    f = product_facts(scores, outcomes, master, rules_cfg)
    f["order_qty"] = order_quantity(f)
    f["high_risk"] = (f["flagged_share"] >= C.PLAN_HIGH_RISK_SHARE) | (f["waste_pct"] >= C.PLAN_HIGH_WASTE_PCT)
    fill = np.where(f["high_risk"], C.PLAN_FILL_HIGH_RISK, C.PLAN_FILL_LOW_RISK)

    f["largest_safe_delivery"] = np.maximum(1, np.floor(f["avg_daily_sales"] * f["batch_life_days"] * fill)).astype(int)
    f["deliveries"] = [math.ceil(q / c) if q > 0 else 0 for q, c in zip(f["order_qty"], f["largest_safe_delivery"])]
    f["units_per_delivery"] = [math.ceil(q / d) if d else 0 for q, d in zip(f["order_qty"], f["deliveries"])]
    f["deliver_every_days"] = [round(C.PLAN_DAYS / d, 1) if d else 0.0 for d in f["deliveries"]]
    f["order_cost"] = (f["order_qty"] * f["cost_price"]).round(0)
    f["risk_label"] = np.where(f["high_risk"], "High", "Low")
    return f.sort_values("order_cost", ascending=False).reset_index(drop=True)

# ------------------------------------------------------------
# WHAT WE DID IN STEP 3:
# - Marked each product High or Low waste risk (model flag share or real waste).
# - Cut the order into deliveries small enough to sell before expiry.
# - Added the cost of the order, so the manager sees the money involved.
# ------------------------------------------------------------


# ============================================================
# STEP 4: A TABLE FOR THE SCREEN, AND TOTALS
# ============================================================
def display_plan(plan: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Product": plan["product_name"], "Category": plan["category"],
        "Sold per day": plan["avg_daily_sales"].round(1), "Batch life (days)": plan["batch_life_days"].round(0).astype(int),
        "Wasted so far (units)": plan["wasted_units"].round(0).astype(int), "Waste %": plan["waste_pct"].round(0).astype(int),
        "Model flagged %": (plan["flagged_share"] * 100).round(0).astype(int), "Waste risk": plan["risk_label"],
        "In stock now": plan["stock_now"].round(0).astype(int),
        f"Order for {C.PLAN_DAYS} days": plan["order_qty"], "Deliveries": plan["deliveries"],
        "Units per delivery": plan["units_per_delivery"], "Deliver every (days)": plan["deliver_every_days"],
        "Order cost": plan["order_cost"]})


def plan_totals(plan: pd.DataFrame) -> dict:
    return {"cost": float(plan["order_cost"].sum()), "units": int(plan["order_qty"].sum()),
            "high_risk": int(plan["high_risk"].sum()), "products": int(len(plan)),
            "deliveries": int(plan["deliveries"].sum())}

# ------------------------------------------------------------
# WHAT WE DID IN STEP 4:
# - Gave the plan friendly column names and added four totals for the cards on the dashboard.
# ------------------------------------------------------------
