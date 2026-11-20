"""
store.py - the app's MEMORY and the Power BI exports
=====================================================
Design: the app saves every cleaned daily file in state/uploads/. Whenever something changes
(new upload, model switch, new decisions) we REBUILD every output from those files. That keeps the
logic simple and repeatable: same inputs -> same outputs, and a browser refresh loses nothing.

  state/uploads/DayXX.csv   cleaned daily files          (the raw material)
  state/decisions.csv       the manager's decisions      (the human input)
  powerbi_data/*.csv        batch_scores, outcomes, daily_summary, decisions_log, calendar, model_comparison
"""
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src import scoring, rules, order_plan
from src import master as M

UP =C.STATE_DIR / "uploads"
DEC = C.STATE_DIR / "decisions.csv"
LOG = C.STATE_DIR / "upload_log.csv"
DEC_COLS = ["day_no", "date", "batch_id", "product_name", "category", "risk", "suggested_markdown", "decision",
            "final_markdown", "reposition", "model_used", "decided_by", "decided_at"]


def uploaded_days():
    UP.mkdir(parents=True, exist_ok=True)
    return sorted(int(p.stem[3:]) for p in UP.glob("Day*.csv"))


def save_upload(clean: pd.DataFrame):
    """Save one cleaned day. Uploading the same day again REPLACES it (no duplicates)."""
    UP.mkdir(parents=True, exist_ok=True)
    d = int(clean["day_no"].iloc[0])
    clean.to_csv(UP / f"Day{d:02d}.csv", index=False)
    return d


def reset_all():
    for p in UP.glob("Day*.csv"):
        p.unlink()
    for p in [DEC, LOG] + list(C.POWERBI_DIR.glob("*.csv")):
        if p.name != "model_comparison.csv" and p.exists():
            p.unlink()


def log_upload(day_no, filename, rows, warnings):
    """Keep a small history of uploads (shown in Tab 1)."""
    row = pd.DataFrame([{"day_no": day_no, "file": filename, "rows": rows, "quality_notes": "; ".join(warnings) or "OK",
                         "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}])
    old = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame()
    pd.concat([old, row], ignore_index=True).to_csv(LOG, index=False)


def read_upload_log() -> pd.DataFrame:
    return pd.read_csv(LOG) if LOG.exists() else pd.DataFrame(columns=["day_no", "file", "rows", "quality_notes", "uploaded_at"])


def make_decision_rows(actions: pd.DataFrame, decision, final_markdown, reposition, decided_by="manager") -> pd.DataFrame:
    """Turn an action list + the manager's choices into rows for the decisions log."""
    return pd.DataFrame({
        "day_no": actions["day_no"].values, "date": actions["date"].values, "batch_id": actions["batch_id"].values,
        "product_name": actions["product_name"].values, "category": actions["category"].values,
        "risk": actions["risk"].round(4).values, "suggested_markdown": actions["suggested_markdown"].values,
        "decision": list(decision), "final_markdown": list(final_markdown), "reposition": list(reposition),
        "model_used": actions["model_used"].values, "decided_by": decided_by,
        "decided_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})


def auto_approve(day_scores: pd.DataFrame, n: int = 15, rules_cfg=None) -> pd.DataFrame:
    """DEMO SHORTCUT: approve every suggestion on the day's top-n list. Marked decided_by='auto-demo'."""
    a = rules.action_list(day_scores, n, rules_cfg)
    if a.empty:
        return pd.DataFrame(columns=DEC_COLS)
    return make_decision_rows(a, ["Approve"] * len(a), a["suggested_markdown"], a["reposition"], "auto-demo")


def load_scores(model_name=None, rules_cfg=None) -> pd.DataFrame:
    """Read every uploaded day, score with the chosen model, add the rule-layer columns."""
    parts = [pd.read_csv(UP / f"Day{d:02d}.csv", parse_dates=["date", "delivered_date", "expiry_date"])
             for d in uploaded_days()]
    if not parts:
        return pd.DataFrame()
    s = scoring.score(pd.concat(parts, ignore_index=True), model_name)
    return rules.apply_rules(s, rules_cfg)


def load_decisions() -> pd.DataFrame:
    if DEC.exists():
        return pd.read_csv(DEC, parse_dates=["date"])
    return pd.DataFrame(columns=DEC_COLS)


def save_decisions(new: pd.DataFrame):
    """Replace the decisions of the days contained in `new`, keep all other days."""
    old = load_decisions()
    old = old[~old["day_no"].isin(new["day_no"].unique())]
    pd.concat([old, new], ignore_index=True)[DEC_COLS].to_csv(DEC, index=False)


def make_outcomes(scores: pd.DataFrame) -> pd.DataFrame:
    """A batch is CLOSED on the evening its days_left hits 0: the stock left that evening is its real waste.
    Predicted-vs-actual compares that real result with the risk shown on the first evening (the honest one)
    and on the last evening before expiry."""
    if scores.empty:
        return pd.DataFrame()
    closed = scores[scores["days_left"] == 0][["batch_id", "product_name", "category", "day_no", "date", "qty_received",
                                                 "qty_left", "cost_price", "base_price"]].rename(
        columns={"day_no": "closed_day", "date": "closed_date", "qty_left": "units_wasted"})
    first = scores[scores["selling_days_elapsed"] == 1][["batch_id", "risk"]].rename(columns={"risk": "risk_first_evening"})
    prev = scores[scores["days_left"] == 1][["batch_id", "risk"]].rename(columns={"risk": "risk_last_evening"})
    o = closed.merge(first, on="batch_id", how="left").merge(prev, on="batch_id", how="left")
    o["waste_pct"] = (o["units_wasted"] / o["qty_received"] * 100).round(1)
    o["waste_cost"] = (o["units_wasted"] * o["cost_price"]).round(2)
    o["actually_at_risk"] = (o["waste_pct"] >= C.WASTE_AT_RISK_PCT).astype(int)
    return o


def decision_effects(scores, outcomes, decisions, uplift=None) -> pd.DataFrame:
    """Estimated effect of each decision on batches that are already CLOSED (assumption-based, see rules.py)."""
    if decisions.empty or outcomes.empty:
        return pd.DataFrame()
    d = decisions.merge(outcomes[["batch_id", "units_wasted", "actually_at_risk", "closed_day"]], on="batch_id")
    d = d.merge(scores[["batch_id", "day_no", "qty_left", "cost_price", "current_price"]], on=["batch_id", "day_no"], how="left")
    d = d[d["final_markdown"] > 0].copy()
    if d.empty:
        return d
    eff = [rules.estimate_effect(q, w, m, p, c, uplift) for q, w, m, p, c in
           zip(d["qty_left"], d["units_wasted"], d["final_markdown"], d["current_price"], d["cost_price"])]
    d["extra_units_sold"], d["waste_cost_avoided"], d["revenue_recovered"], d["margin_given_away"] = zip(*eff)
    d["net_cash_effect"] = d["revenue_recovered"] - d["margin_given_away"]
    return d


def daily_summary(scores, outcomes, decisions, effects) -> pd.DataFrame:
    rows = []
    for d, g in scores.groupby("day_no"):
        live = g[g["days_left"] > 0]
        o = outcomes[outcomes["closed_day"] == d] if not outcomes.empty else outcomes
        dd = decisions[decisions["day_no"] == d]
        e = effects[effects["day_no"] == d] if not effects.empty else effects
        rows.append({"day_no": d, "date": g["date"].iloc[0], "batches_on_shelf": len(g),
                     "at_risk_flagged": int((live["risk"] >= C.RULES["low"]).sum()),
                     "money_at_stake": live["money_at_stake"].sum(), "discard_tonight": int((g["days_left"] == 0).sum()),
                     "discard_cost": g["discard_cost"].sum(), "reposition_flags": int(live["reposition"].sum()),
                     "closed_batches": len(o), "waste_cost_closed": o["waste_cost"].sum() if len(o) else 0.0,
                     "decisions_saved": len(dd), "approved": int((dd["decision"] == "Approve").sum()),
                     "changed": int((dd["decision"] == "Change").sum()), "rejected": int((dd["decision"] == "Reject").sum()),
                     "est_waste_cost_avoided": e["waste_cost_avoided"].sum() if len(e) else 0.0,
                     "est_margin_given_away": e["margin_given_away"].sum() if len(e) else 0.0,
                     "est_revenue_recovered": e["revenue_recovered"].sum() if len(e) else 0.0,
                     "model_used": g["model_used"].iloc[0]})
    out = pd.DataFrame(rows)
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(2)
    return out


def export_powerbi(scores, outcomes, decisions, effects, summary):
    """Write the clean CSVs that Power BI reads. Called after every upload / decision / model switch."""
    C.POWERBI_DIR.mkdir(exist_ok=True)
    keep = ["date", "day_no", "batch_id", "product_name", "category", "delivered_date", "expiry_date",
            "selling_days_elapsed", "days_left", "qty_received", "qty_left", "units_sold_today", "units_sold_total",
            "current_price", "cost_price", "risk", "action", "suggested_markdown", "reposition", "new_price",
            "money_at_stake", "discard_cost", "reason", "model_used"]
    out = scores[keep].copy()
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(4)
    out.to_csv(C.POWERBI_DIR / "batch_scores.csv", index=False)
    outcomes.to_csv(C.POWERBI_DIR / "outcomes.csv", index=False)
    decisions.to_csv(C.POWERBI_DIR / "decisions_log.csv", index=False)
    effects.to_csv(C.POWERBI_DIR / "decision_effects.csv", index=False)
    summary.to_csv(C.POWERBI_DIR / "daily_summary.csv", index=False)
    start = pd.Timestamp(C.DEMO_START)
    cal = pd.DataFrame({"day_no": range(1, C.N_DAYS + 1)})
    cal["date"] = start + pd.to_timedelta(cal["day_no"] - 1, unit="D")
    cal["weekday"] = cal["date"].dt.strftime("%A")
    cal["week"] = (cal["day_no"] - 1) // 7 + 1
    cal["delivery"] = cal["day_no"].map(lambda d: "Sunday delivery" if d in C.DELIVERY_SUNDAYS else
                                        ("Wednesday top-up" if d in C.DELIVERY_WEDNESDAYS else ""))
    cal.to_csv(C.POWERBI_DIR / "calendar.csv", index=False)


def refresh(model_name=None, rules_cfg=None, uplift=None):
    """Rebuild everything from the saved uploads + decisions. Returns a dict of tables for the app."""
    scores = load_scores(model_name, rules_cfg)
    if scores.empty:
        return {"scores": scores}
    decisions = load_decisions()
    outcomes = make_outcomes(scores)
    effects = decision_effects(scores, outcomes, decisions, uplift)
    summary = daily_summary(scores, outcomes, decisions, effects)
    export_powerbi(scores, outcomes, decisions, effects, summary)
    # order plan for the next 15 days (per product); uses the master file for batch life when it is saved
    plan = order_plan.build_order_plan(scores, outcomes, M.load_master(), rules_cfg or C.RULES)
    order_plan.display_plan(plan).to_csv(C.POWERBI_DIR / "order_plan.csv", index=False)
    return {"scores": scores, "outcomes": outcomes, "decisions": decisions, "effects": effects, "summary": summary,
            "plan": plan}
