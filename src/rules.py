"""
rules.py - the RULE LAYER (plain business rules, NOT a model)
==============================================================
The model only predicts a RISK %. These rules turn the risk into an action the manager can read.
The thresholds are editable in the app sidebar.
"""
import numpy as np
import pandas as pd
import config as C


def suggest(risk, days_left, rules=None):
    """Return (action_text, markdown_pct, reposition) for one batch."""
    r = rules or C.RULES
    if days_left <= 0:
        return "Discard tonight", 0, False
    if risk is None or np.isnan(risk) or risk < r["low"]:
        return "No action", 0, False
    if risk < r["high"]:
        return f"{r['md_mid']}% markdown", r["md_mid"], False
    return f"{r['md_high']}% markdown + move to front display", r["md_high"], True


def apply_rules(scored: pd.DataFrame, rules=None) -> pd.DataFrame:
    """Add action, markdown, reposition, new price and money-at-stake columns to a scored table."""
    s = scored.copy()
    out = [suggest(r, d, rules) for r, d in zip(s["risk"], s["days_left"])]
    s["action"] = [o[0] for o in out]
    s["suggested_markdown"] = [o[1] for o in out]
    s["reposition"] = [o[2] for o in out]
    s["new_price"] = (s["current_price"] * (1 - s["suggested_markdown"] / 100)).round(2)
    # money at stake = chance it is wasted x units left x what the stock cost us
    s["money_at_stake"] = (s["risk"].fillna(0) * s["qty_left"] * s["cost_price"]).round(2)
    s["discard_cost"] = np.where(s["days_left"] <= 0, s["qty_left"] * s["cost_price"], 0.0).round(2)
    return s


def estimate_effect(qty_left_at_decision, final_waste_units, markdown_pct, price, cost, uplift=None):
    """ASSUMPTION-based estimate of what a markdown would have changed (only for CLOSED batches).

    Baseline (real, from the daily files): R units would still have sold, W units were wasted.
    Assumption: a markdown of m% raises the remaining sales by uplift*m%, capped at the stock left.
    Returns (extra_units_sold, waste_cost_avoided, revenue_recovered, margin_given_away).
      waste_cost_avoided = extra units x cost price     (the write-off that did not happen)
      revenue_recovered  = extra units x discounted price (new cash)
      margin_given_away  = discount handed to the R units that would have sold anyway
    Net CASH effect = revenue_recovered - margin_given_away  (waste_cost_avoided is NOT added: it would double count).
    """
    u = C.UPLIFT_PER_MARKDOWN if uplift is None else uplift
    m = markdown_pct / 100.0
    R = max(qty_left_at_decision - final_waste_units, 0)
    extra = min(qty_left_at_decision, R * (1 + u * m)) - R
    return extra, extra * cost, extra * price * (1 - m), R * price * m


def break_even_uplift(markdown_pct):
    """Smallest uplift (sales increase per 1% of markdown) at which a markdown pays for itself in cash:
    u * (1 - m) >= 1  ->  u >= 1 / (1 - m). Ignores disposal costs and freed shelf space."""
    return 1.0 / (1.0 - markdown_pct / 100.0)


def action_list(day_scores: pd.DataFrame, n: int = 15, rules=None) -> pd.DataFrame:
    """The manager's to-do list for one evening: only batches that still have days left AND whose risk is at
    or above the 'low' threshold, biggest money-at-stake first, limited to the top n."""
    r = rules or C.RULES
    live = day_scores[(day_scores["days_left"] > 0) & (day_scores["risk"] >= r["low"])]
    return live.sort_values("money_at_stake", ascending=False).head(n).reset_index(drop=True)
