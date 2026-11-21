# ====================================================================
# DEMO - FINAL DASHBOARD numbers (pure pandas, no Streamlit)
# Built from the 8 saved days.
# ====================================================================
import numpy as np
import pandas as pd

from src.data_prep import ROOT
from src.evaluation import evaluate

KEY_FILE = ROOT / "data" / "demo" / "answer_key_hidden.csv"


def combine(hist):
    frames = []
    for d, df in hist.items():
        x = df.copy(); x["day"] = d; frames.append(x)
    return pd.concat(frames, ignore_index=True)


def week_summary(all_rows):
    high = all_rows[all_rows["level"] == "High"]
    return {"days": all_rows["day"].nunique(), "rows": len(all_rows), "batches": all_rows["batch_id"].nunique(),
            "high_batches": high["batch_id"].nunique(), "high_alerts": len(high),
            "units_at_risk_today_avg": float(high.groupby("day")["units_left"].sum().mean()) if len(high) else 0.0}


def daily_counts(all_rows):
    t = all_rows.groupby(["day", "level"]).size().unstack(fill_value=0)
    return t.reindex(columns=["High", "Medium", "Low"], fill_value=0)


def by_category(all_rows):
    g = all_rows.groupby("category")
    t = pd.DataFrame({"batches": g["batch_id"].nunique(),
                      "high-risk batches": all_rows[all_rows["level"] == "High"].groupby("category")["batch_id"].nunique(),
                      "average chance of waste": g["chance"].mean()}).fillna(0)
    t["high-risk batches"] = t["high-risk batches"].astype(int)
    return t.sort_values("average chance of waste", ascending=False).reset_index()


def repeat_risk(all_rows, min_days=3):
    h = all_rows[all_rows["level"] == "High"]
    g = h.groupby("batch_id").agg(category=("category", "first"), sub_category=("sub_category", "first"),
                                  days_high=("day", "nunique"), last_chance=("chance", "last"))
    return g[g["days_high"] >= min_days].sort_values("days_high", ascending=False).reset_index()


def actions_summary(all_rows):
    h = all_rows[all_rows["level"] == "High"]
    t = h["action"].value_counts().rename_axis("action taken").reset_index(name="High-risk rows")
    decided = float((h["action"] != "Not decided").mean()) if len(h) else 0.0
    return t, decided


def prediction_check(all_rows, threshold=0.5):
    """Compare predictions with what really happened - only for batches whose life ended inside the week."""
    key = pd.read_csv(KEY_FILE)
    done = key[key["batch_ends_in_week"]][["batch_id", "at_risk", "units_wasted_final"]]
    m = all_rows.merge(done, on="batch_id", how="inner")
    res = {"finished": int(done["batch_id"].nunique()), "open": int(key["batch_id"].nunique() - done["batch_id"].nunique()),
           "rows": len(m)}
    if len(m) and m["at_risk"].nunique() == 2:
        res["metrics"] = evaluate(m["at_risk"], m["chance"], threshold)
    batch = m.groupby("batch_id").agg(category=("category", "first"), sub_category=("sub_category", "first"),
                                      highest_chance=("chance", "max"), really_at_risk=("at_risk", "first"),
                                      units_wasted=("units_wasted_final", "first")).reset_index()
    batch["flagged High at least once"] = batch["highest_chance"] >= 0.70
    res["batch_table"] = batch.sort_values("highest_chance", ascending=False)
    return res
