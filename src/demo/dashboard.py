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
    """Day 0 is only the loaded inventory, so actions are counted from Day 1."""
    h = all_rows[(all_rows["level"] == "High") & (all_rows["day"] >= 1)].copy()
    h["family"] = h["action"].map(lambda a: "Discount" if str(a).startswith("Discount") else a)
    t = h["family"].value_counts().rename_axis("action taken").reset_index(name="High-risk rows")
    decided = float((h["action"] != "Not decided").mean()) if len(h) else 0.0
    disc = h[h["family"] == "Discount"]["action"].value_counts().rename_axis("discount given").reset_index(name="rows")
    return t, decided, disc


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


# ====================================================================
# SIMPLE KPIs for the final dashboard (all as % where possible)
# ====================================================================
HIGH = 0.70          # same line as the "High" risk level in scoring.py


def _pct(a, b):
    return a / b if b else 0.0


def kpis(all_rows):
    batches = all_rows["batch_id"].nunique()
    high_ids = set(all_rows.loc[all_rows["level"] == "High", "batch_id"])
    pc = prediction_check(all_rows)
    bt = pc["batch_table"]                                     # batches whose life ended inside the week
    flagged = bt["highest_chance"] >= HIGH
    real = bt["really_at_risk"] == 1
    caught, correct = int((flagged & real).sum()), int((flagged == real).sum())
    t, decided, _ = actions_summary(all_rows)
    return {
        "batches": batches, "high_batches": len(high_ids), "high_pct": _pct(len(high_ids), batches),
        "finished": pc["finished"], "open": pc["open"], "open_pct": _pct(pc["open"], pc["open"] + pc["finished"]),
        "really_at_risk": int(real.sum()), "caught": caught, "caught_pct": _pct(caught, int(real.sum())),
        "flagged": int(flagged.sum()), "correct_alerts_pct": _pct(caught, int(flagged.sum())),
        "accuracy_pct": _pct(correct, len(bt)), "action_pct": decided,
        "wasted_units": float(bt.loc[real, "units_wasted"].sum()),
    }


def daily_high_pct(all_rows):
    g = all_rows.groupby("day")
    return (g.apply(lambda d: (d["level"] == "High").mean(), include_groups=False)).rename("pct")


def category_high_pct(all_rows):
    """% of each category's batches that were flagged High at least once."""
    n = all_rows.groupby("category")["batch_id"].nunique()
    h = all_rows[all_rows["level"] == "High"].groupby("category")["batch_id"].nunique().reindex(n.index, fill_value=0)
    return pd.DataFrame({"batches": n, "flagged High": h, "pct": h / n}).sort_values("pct", ascending=False).reset_index()


# ====================================================================
# WASTE OUTCOMES AND NEXT-WEEK ORDER RECOMMENDATIONS
# Rule is learned from the demo week and TESTED on the following week
# (real batches the rule has never seen), so the saving is not hindsight.
# ====================================================================
BUFFER = 0.10                              # safety stock kept on top of what really sold
TEST_FILE = ROOT / "data" / "store" / "store_test_30.csv"
WEEK = ("2024-11-02", "2024-11-09")        # Day 0 .. Day 7 (Saturday to Saturday)
NEXT_WEEK = ("2024-11-10", "2024-11-16")   # the week after Day 7


def _batches():
    """One row per batch of the demo store: input (received) and final (wasted) units."""
    store = pd.read_csv(KEY_FILE)["store_id"].iloc[0]
    t = pd.read_csv(TEST_FILE)
    t = t[t["store_id"] == store]
    b = t.groupby("batch_id").agg(category=("category", "first"), product=("sub_category", "first"),
                                  first=("date", "min"), last=("date", "max"), received=("units_received", "first"),
                                  wasted=("units_wasted_final", "first"), cost_price=("cost_price", "first"),
                                  waste_cost=("waste_cost", "first")).reset_index()
    b["first"], b["last"] = pd.to_datetime(b["first"]), pd.to_datetime(b["last"])
    b["used"] = b["received"] - b["wasted"]
    b["waste_pct"] = b["wasted"] / b["received"]
    return b


def finished_batches(week=WEEK):
    """Batches whose life ended inside the week - their result is known."""
    b = _batches()
    return b[(b["last"] >= week[0]) & (b["last"] <= week[1])].reset_index(drop=True)


def next_week_batches(week=NEXT_WEEK):
    b = _batches()
    return b[(b["first"] >= week[0]) & (b["first"] <= week[1])].reset_index(drop=True)


def waste_headline(fin):
    rec, was = float(fin["received"].sum()), float(fin["wasted"].sum())
    return {"finished": len(fin), "received": rec, "wasted": was, "waste_rate": was / rec if rec else 0.0,
            "waste_cost": float(fin["waste_cost"].sum()), "batches_with_waste": int((fin["wasted"] > 0).sum())}


def category_input_waste(fin):
    g = fin.groupby("category").agg(batches=("batch_id", "count"), received=("received", "sum"),
                                    wasted=("wasted", "sum"), waste_cost=("waste_cost", "sum"))
    g["waste_pct"] = g["wasted"] / g["received"]
    return g.sort_values("waste_pct", ascending=False).reset_index()


# ---- the recommendation rule -------------------------------------------------
def order_rules(fin, buffer=BUFFER, min_batches=2):
    """Order factor per product = (units that really sold x (1 + buffer)) / units ordered, never above 1.
    Products with fewer than `min_batches` finished batches use their category's factor."""
    gp = fin.groupby("product").agg(category=("category", "first"), batches=("batch_id", "count"),
                                    received=("received", "sum"), used=("used", "sum"), wasted=("wasted", "sum"))
    gc = fin.groupby("category").agg(received=("received", "sum"), used=("used", "sum"))
    fcat = (gc["used"] * (1 + buffer) / gc["received"]).clip(upper=1.0)
    fprod = (gp["used"] * (1 + buffer) / gp["received"]).clip(upper=1.0)
    gp["basis"] = ["product" if n >= min_batches else "category" for n in gp["batches"]]
    gp["factor"] = [fprod[p] if gp.loc[p, "basis"] == "product" else fcat[gp.loc[p, "category"]] for p in gp.index]
    gp["waste_pct"] = gp["wasted"] / gp["received"]
    gp["avg_ordered"] = gp["received"] / gp["batches"]
    gp["avg_recommended"] = gp["avg_ordered"] * gp["factor"]
    gp["change_pct"] = gp["factor"] - 1
    return gp.reset_index().sort_values("change_pct").reset_index(drop=True), fcat


def apply_rules(rules, fcat, batches):
    """Replay a week: what would the order, the waste and the lost sales have been?"""
    f = rules.set_index("product")["factor"]
    b = batches.copy()
    b["factor"] = [f[p] if p in f.index else fcat.get(c, 1.0) for p, c in zip(b["product"], b["category"])]
    b["recommended"] = np.ceil(b["received"] * b["factor"]).astype(int)
    b["waste_after"] = np.maximum(0, b["recommended"] - b["used"])
    b["lost_sales"] = np.maximum(0, b["used"] - b["recommended"])
    b["waste_avoided"] = b["wasted"] - b["waste_after"]
    b["money_saved"] = b["waste_avoided"] * b["cost_price"]
    return b


def savings_summary(b):
    w, wa = float(b["wasted"].sum()), float(b["waste_after"].sum())
    r, rn = float(b["received"].sum()), float(b["recommended"].sum())
    return {"batches": len(b), "wasted_now": w, "waste_after": wa, "waste_saving_pct": 1 - wa / w if w else 0.0,
            "waste_rate_now": w / r, "waste_rate_after": wa / rn if rn else 0.0,
            "less_ordered_pct": 1 - rn / r, "lost_sales_pct": float(b["lost_sales"].sum()) / float(b["used"].sum()),
            "money_saved": float(b["money_saved"].sum()), "ordered_now": r, "ordered_new": rn}


def rolling_check(buffer=BUFFER):
    """Same rule, learned on each week and tested on the next one - across the whole test period."""
    b = _batches(); out = []
    for w0 in pd.date_range("2024-06-24", "2024-12-16", freq="7D"):
        tr = b[(b["last"] >= w0) & (b["last"] <= w0 + pd.Timedelta(days=7))]
        te = b[(b["first"] > w0 + pd.Timedelta(days=7)) & (b["first"] <= w0 + pd.Timedelta(days=14))]
        if len(tr) < 30 or len(te) < 30:
            continue
        rules, fcat = order_rules(tr, buffer)
        sm = savings_summary(apply_rules(rules, fcat, te))
        out.append({"week_start": w0.date(), "waste_saving_pct": sm["waste_saving_pct"], "lost_sales_pct": sm["lost_sales_pct"]})
    return pd.DataFrame(out)
