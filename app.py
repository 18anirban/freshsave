"""
FreshSave - app.py   (run with:  streamlit run app.py)
=======================================================
A proof-of-concept for ONE store. The flow in four steps, one per tab:

   Tab 1  Evening upload      -> upload today's closing-stock file (Day 1 ... Day 14)
   Tab 2  Daily decisions     -> the store manager sees what to discount / move / discard, and decides
   Tab 3  Two-week dashboard  -> waste, savings, and whether the predictions came true
   Tab 4  Model comparison    -> how the Decision Tree and ANN were tested + choose the active model

How to read this file: the app only DISPLAYS things. The real work is in src/
   scoring.py (validate + predict)  rules.py (risk -> action)  store.py (memory + Power BI exports)
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from src import rules, scoring, store

st.set_page_config(page_title="FreshSave", layout="wide")
MODELS = list(scoring.MODEL_FILES)
LABELS = {"Approve": "Approve", "Change": "Change", "Reject": "Reject"}


# ----------------------------------------------------------------------------------------------
# Small helpers (they hide Streamlit version differences so the app runs on old and new versions)
# ----------------------------------------------------------------------------------------------
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


def models_ready():
    """True only when train.py has produced every file the app needs."""
    need = ["active_model.json", "decision_tree.joblib", "ann.joblib", "feature_names.json", "data_profile.json",
            "metrics_summary.csv", "curves.json", "test_predictions.csv.gz"]
    return all((C.MODELS_DIR / n).exists() for n in need)


def read_json(name):
    return json.load(open(C.MODELS_DIR / name))


def daily_file(day):
    """Find DayNN_*.csv in the project folder (or the Windows Downloads folder)."""
    for folder in (C.DAILY_DIR, C.WINDOWS_DAILY_DIR):
        try:
            hits = sorted(Path(folder).glob(f"Day{day:02d}_*.csv"))
        except OSError:
            hits = []
        if hits:
            return hits[0]
    return None


def ingest(df, filename):
    """Validate -> clean -> save one day. Returns (ok, message)."""
    clean, errors, warnings = scoring.validate_and_clean(df)
    if errors:
        return False, f"{filename}: " + " ".join(errors)
    day = store.save_upload(clean)
    store.log_upload(day, filename, len(clean), warnings)
    return True, f"{filename}: Day {day} saved ({len(clean)} batches)." + (f" Notes: {' '.join(warnings)}" if warnings else "")


# ----------------------------------------------------------------------------------------------
# Guard: the models must exist (train.py creates them)
# ----------------------------------------------------------------------------------------------
if not models_ready():
    st.title("FreshSave")
    st.error("The models have not been trained yet. Open a terminal in this folder and run:  python src/train.py")
    st.stop()

# ----------------------------------------------------------------------------------------------
# Sidebar: settings the manager (or you, in the demo) can change
# ----------------------------------------------------------------------------------------------
if "flash" not in st.session_state:
    st.session_state["flash"] = []

with st.sidebar:
    st.header("FreshSave")
    st.caption(f"Store: **{C.STORE_ID}**  |  Proof of concept")
    active = scoring.active_model_name()
    st.markdown(f"**Active model:** {active}")
    st.caption("Change it in Tab 4.")
    st.subheader("Rules")
    low = st.slider("Act when risk is above (%)", 10, 90, int(C.RULES["low"] * 100), 5) / 100
    high = st.slider("Strong action above (%)", 30, 95, int(C.RULES["high"] * 100), 5) / 100
    md_mid = st.slider("Medium markdown (%)", 5, 50, C.RULES["md_mid"], 5)
    md_high = st.slider("Strong markdown (%)", 10, 70, C.RULES["md_high"], 5)
    rules_cfg = {"low": low, "high": max(high, low), "md_mid": md_mid, "md_high": md_high}
    st.subheader("Assumption")
    uplift = st.slider("Sales response to a markdown", 0.25, 4.0, float(C.UPLIFT_PER_MARKDOWN), 0.25,
                       help="Sales rise by this number x the markdown %. 0.75 means 20% off gives +15% sales. "
                            "This is an ASSUMPTION, not measured. It only affects the 'estimated savings'.")
    st.caption(f"20% off gives about +{uplift*20:.0f}% sales.")

# Rebuild every table from the saved uploads + decisions (also refreshes the Power BI CSVs)
data = store.refresh(active, rules_cfg, uplift)
scores = data["scores"]
have_data = not scores.empty
days = store.uploaded_days()

st.title("FreshSave: decide today what to discount, move or discard")
st.caption("Each evening: upload the closing stock, see what is at risk, decide, and track the waste.")
for kind, msg in st.session_state["flash"]:
    getattr(st, kind)(msg)
st.session_state["flash"] = []

tab1, tab2, tab3, tab4 = st.tabs(["1. Evening upload", "2. Daily decisions", "3. Two-week dashboard",
                                  "4. Model comparison"])

# ==============================================================================================
# TAB 1 - EVENING UPLOAD
# ==============================================================================================
with tab1:
    st.subheader("Upload tonight's closing stock")
    st.write("In a real store this file arrives automatically from the inventory system each evening. "
             "Here you upload it yourself (Day 1, then Day 2, ... up to Day 14).")
    files = st.file_uploader("Choose one or more daily files (CSV)", type="csv", accept_multiple_files=True)
    c1, c2, c3 = st.columns(3)
    if c1.button("Check, clean and score the file(s)", disabled=not files):
        for f in files:
            try:
                ok, msg = ingest(pd.read_csv(f), f.name)
            except Exception as ex:                                   # unreadable file
                ok, msg = False, f"{f.name}: could not be read ({ex})"
            st.session_state["flash"].append(("success" if ok else "error", msg))
        st.rerun()
    nxt = next((d for d in range(1, C.N_DAYS + 1) if d not in days), None)
    if c2.button(f"Load next day from folder (Day {nxt})" if nxt else "All 14 days loaded", disabled=nxt is None):
        p = daily_file(nxt)
        if p is None:
            st.session_state["flash"].append(("error", f"Day {nxt} file not found. Run: python src/make_daily_files.py"))
        else:
            ok, msg = ingest(pd.read_csv(p), p.name)
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
                    ingest(pd.read_csv(p), p.name)
            fresh = store.load_scores(active, rules_cfg)
            for d in store.uploaded_days():
                if store.load_decisions().query("day_no == @d").empty:
                    store.save_decisions(store.auto_approve(fresh[fresh.day_no == d], 15, rules_cfg))
            st.session_state["flash"].append(("success", "All days loaded. Suggestions were auto-approved (tagged 'auto-demo')."))
            st.rerun()

    st.markdown("**Days uploaded**")
    log = store.read_upload_log()
    rows = []
    for d in range(1, C.N_DAYS + 1):
        n = int(len(scores[scores.day_no == d])) if have_data else 0
        rows.append({"Day": d, "Date": (pd.Timestamp(C.DEMO_START) + pd.Timedelta(days=d - 1)).strftime("%a %d %b"),
                     "Status": "Uploaded" if d in days else "Waiting", "Batches on shelf": n or "",
                     "Delivery": "Sunday delivery" if d in C.DELIVERY_SUNDAYS else
                                 "Wednesday top-up" if d in C.DELIVERY_WEDNESDAYS else ""})
    show_df(pd.DataFrame(rows))
    st.markdown("**What the checks look for:** all columns present, one day per file, this store only, no duplicate "
                "batches, no negative quantities, no stock left above stock received.")
    if not log.empty:
        st.markdown("**Upload history and data-quality notes**")
        show_df(log.sort_values("uploaded_at", ascending=False))
    with st.expander("Start again (clears uploads, decisions and Power BI files)"):
        if st.checkbox("Yes, I want to clear everything") and st.button("Reset"):
            store.reset_all()
            st.session_state["flash"].append(("success", "Everything was cleared."))
            st.rerun()

# ==============================================================================================
# TAB 2 - DAILY DECISIONS (store manager)
# ==============================================================================================
with tab2:
    if not have_data:
        st.info("Upload a day in Tab 1 to see the manager's list.")
    else:
        top_l, top_r = st.columns([1, 1])
        day = top_l.selectbox("Day", days, index=len(days) - 1,
                              format_func=lambda d: f"Day {d} - {(pd.Timestamp(C.DEMO_START) + pd.Timedelta(days=d-1)).strftime('%A %d %b')}")
        n_show = top_r.slider("How many batches to show", 10, 15, 15)
        g = scores[scores.day_no == day]
        live = g[g.days_left > 0]
        flagged = live[live.risk >= rules_cfg["low"]]
        disc = g[g.days_left == 0]

        a, b, c, d_, e = st.columns(5)
        a.metric("Batches on shelf", len(g))
        b.metric("At risk", len(flagged))
        c.metric("Money at stake", money(flagged.money_at_stake.sum()))
        d_.metric("Discard tonight", len(disc), money(disc.discard_cost.sum()) + " lost", delta_color="off")
        e.metric("Expected waste (units)", f"{int((live.risk * live.qty_left).sum()):,}")
        st.caption("Money at stake = chance of waste x units left x cost price. The model suggests; **you decide**.")

        # ---- the action list ------------------------------------------------------------
        st.subheader("Today's action list")
        acts = rules.action_list(g, n_show, rules_cfg)
        if acts.empty:
            st.success("Nothing is above the risk threshold today.")
        else:
            saved = data["decisions"]
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
                tbl, hide_index=True, key=f"ed_{day}_{n_show}_{active}_{low}_{high}_{md_mid}_{md_high}",
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
                repo = [bool(r) if d == "Approve" else (False if d == "Reject" else bool(r))
                        for d, r in zip(edited["Decision"], acts.reposition)]
                store.save_decisions(store.make_decision_rows(acts, edited["Decision"], final, repo))
                st.session_state["flash"].append(("success", f"Saved {len(acts)} decisions for Day {day}."))
                st.rerun()
            n_repo = int(acts.reposition.sum())
            if n_repo:
                st.info(f"{n_repo} batch(es) are suggested for the front display / eye level / a bundle.")

        with st.expander(f"Discard tonight ({len(disc)} batches, {money(disc.discard_cost.sum())} of stock)"):
            show_df(disc[["batch_id", "product_name", "category", "qty_left", "cost_price", "discard_cost"]]
                    .sort_values("discard_cost", ascending=False)
                    .rename(columns={"qty_left": "Units to discard", "discard_cost": "Cost lost"}))

        # ---- scorecard --------------------------------------------------------------------
        if day >= 2 and not data["outcomes"].empty:
            st.subheader(f"Scorecard: what happened up to Day {day}")
            o = data["outcomes"]
            closed_now = o[o.closed_day == day]
            dec_prev = data["decisions"][data["decisions"].day_no == day - 1] if not data["decisions"].empty else pd.DataFrame()
            s1, s2 = st.columns(2)
            with s1:
                st.markdown(f"**Batches that finished tonight ({len(closed_now)})**")
                if len(closed_now):
                    f_ = closed_now.risk_first_evening >= rules_cfg["low"]
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

# ==============================================================================================
# TAB 3 - TWO-WEEK DASHBOARD
# ==============================================================================================
with tab3:
    if not have_data:
        st.info("Upload a day in Tab 1. This page fills in as days arrive.")
    else:
        o, dec, ef, sm = data["outcomes"], data["decisions"], data["effects"], data["summary"]
        st.warning("**Read this first.** Waste, flags and prediction accuracy come from the uploaded files. "
                   "The daily sales path inside each batch is SIMULATED, and every figure marked *estimated* uses the "
                   "assumed sales response to a markdown. Nothing here is a measured saving.")
        k = st.columns(5)
        k[0].metric("Days uploaded", len(days))
        k[1].metric("Batches finished", len(o))
        k[2].metric("Actual waste cost", money(o.waste_cost.sum()) if len(o) else "$0")
        k[3].metric("Waste % of stock", f"{o.units_wasted.sum() / max(o.qty_received.sum(), 1):.1%}" if len(o) else "-")
        k[4].metric("Decisions saved", len(dec))
        if o.empty:
            st.info("Batches finish when their days left reach 0. Upload more days to see waste results.")
        else:
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

            st.subheader("Can we trust the predictions? (predicted vs actual)")
            st.caption("Each finished batch is compared with the risk it showed on its FIRST evening (the honest test) "
                       "and on its last evening before expiry (easier, because most of the answer is already visible).")
            m1, m2 = st.columns(2)
            for col, key, title in [(m1, "risk_first_evening", "First evening"), (m2, "risk_last_evening", "Last evening")]:
                x = o.dropna(subset=[key])
                pred = x[key] >= rules_cfg["low"]
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

            st.subheader("Actions taken per day")
            show_fig(px.bar(sm.melt("day_no", ["at_risk_flagged", "reposition_flags", "discard_tonight"]), x="day_no", y="value",
                            color="variable", barmode="group", labels={"day_no": "Day", "value": "Batches", "variable": ""}))
            st.download_button("Download daily summary (CSV)", sm.to_csv(index=False), "daily_summary.csv")
        st.caption("These tables are also saved for Power BI in the powerbi_data folder after every change.")

# ==============================================================================================
# TAB 4 - MODEL SELECTION AND COMPARISON
# ==============================================================================================
with tab4:
    info = read_json("active_model.json")
    prof = read_json("data_profile.json")
    met = pd.read_csv(C.MODELS_DIR / "metrics_summary.csv")
    st.subheader("Choose the model that scores the batches")

    def _switch():
        scoring.set_active_model(st.session_state["model_pick"])
    st.selectbox("Active model (drives Tabs 2 and 3)", MODELS, index=MODELS.index(info["active"]),
                 key="model_pick", on_change=_switch)
    st.success(f"Default winner: **{info['winner']}**. {info['why']}")
    st.caption("Both models were trained on 2023 data from all 50 stores and tested on 2024. The demo store's 2024 "
               "batches were never used for training.")

    view = st.radio("Test data", ["All stores 2024", f"{C.STORE_ID} only (2024)"], horizontal=True)
    mv = met[met.view == view].set_index("model")

    with st.expander("a) Data profile", expanded=False):
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Raw rows", f"{prof['raw_rows']:,}")
        p2.metric("Columns", prof["raw_columns"])
        p3.metric("Missing cells", prof["raw_missing_cells"])
        p4.metric("Fresh batches kept", f"{prof['fresh_batches']:,}")
        st.write(f"Class balance (batches at risk = {prof['waste_threshold_pct']:.0f}%+ wasted): train "
                 f"**{prof['at_risk_rate_train_batches']:.1%}**, test **{prof['at_risk_rate_test_batches']:.1%}**. "
                 f"Evening snapshots used for training: {prof['train_snapshots']:,}, testing: {prof['test_snapshots']:,}.")
        show_fig(px.bar(x=list(prof["category_counts"]), y=list(prof["category_counts"].values()),
                        labels={"x": "Category", "y": "Batches"}, title="Batches per category"))
        sk = pd.read_csv(C.MODELS_DIR / "skewness.csv")
        st.markdown("**Skewness and kurtosis before and after the log transform** (closer to 0 = more symmetric)")
        show_df(sk.round(2))
        smp = pd.read_csv(C.MODELS_DIR / "profile_sample.csv")
        col = st.selectbox("Show the histogram of", list(smp.columns))
        h1, h2 = st.columns(2)
        with h1:
            show_fig(px.histogram(smp, x=col, nbins=40, title=f"{col} (before)"))
        with h2:
            show_fig(px.histogram(x=np.log1p(smp[col].clip(lower=0)), nbins=40, title=f"{col} (after log1p)"))

    with st.expander("b) How the data was prepared"):
        st.markdown(f"""
- **Split by time:** training = dates before {prof['split_date']} (all stores); testing = {prof['split_date']} onward. No shuffling across time.
- **Fresh food only:** Pharmaceuticals and Frozen_Meals are dropped (shelf life of months).
- **Label:** a batch is *at risk* if {prof['waste_threshold_pct']:.0f}% or more of it ended up wasted.
- **One row per batch per evening** (simulated path - see the README). The last evening is a discard, not a prediction.
- **Transformations:** log1p on skewed columns (both models), category one-hot encoded, scaling for the ANN only.
- **{len(prof['features'])} features:** {', '.join(prof['features'])}
""")
        st.markdown("**Columns deliberately left out, and why**")
        show_df(pd.DataFrame({"Column": list(prof["excluded"]), "Reason": list(prof["excluded"].values())}))

    with st.expander("c) Metrics side by side", expanded=True):
        order = ["accuracy", "precision", "recall", "f1", "roc_auc", "specificity", "balanced_accuracy", "mcc", "log_loss", "brier"]
        nice = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall (at-risk caught)", "f1": "F1", "roc_auc": "ROC-AUC",
                "specificity": "Specificity", "balanced_accuracy": "Balanced accuracy", "mcc": "MCC", "log_loss": "Log loss (lower is better)",
                "brier": "Brier score (lower is better)"}
        t = mv.loc[MODELS, order].T.rename(index=nice).round(3).reset_index().rename(columns={"index": "Metric"})
        show_df(t)
        show_fig(px.bar(mv.loc[MODELS, ["accuracy", "precision", "recall", "f1", "roc_auc"]].reset_index().melt("model"),
                        x="variable", y="value", color="model", barmode="group", labels={"variable": "", "value": "Score"},
                        title=f"Main scores - {view}"))
        st.caption("Recall is the number to watch: a missed at-risk batch becomes waste.")

    with st.expander("d) Visuals"):
        r1, r2 = st.columns(2)
        for col, m in zip((r1, r2), MODELS):
            x = mv.loc[m]
            with col:
                st.markdown(f"**{m}: what happened to {int(x.tp + x.fn + x.fp + x.tn):,} evening checks**")
                show_df(pd.DataFrame({"Result": ["At-risk batches caught", "At-risk batches missed (become waste)",
                                                  "False alarms (needless markdown)", "Correctly left alone"],
                                      "Count": [int(x.tp), int(x.fn), int(x.fp), int(x.tn)]}))
        cv = read_json("curves.json")
        roc, pr = go.Figure(), go.Figure()
        for m in MODELS:
            roc.add_scatter(x=cv[m]["roc_fpr"], y=cv[m]["roc_tpr"], name=m, mode="lines")
            pr.add_scatter(x=cv[m]["pr_recall"], y=cv[m]["pr_precision"], name=m, mode="lines")
        roc.add_scatter(x=[0, 1], y=[0, 1], name="Guessing", mode="lines", line=dict(dash="dot"))
        roc.update_layout(title="ROC curve (all stores)", xaxis_title="False alarm rate", yaxis_title="Caught rate")
        pr.update_layout(title="Precision-recall curve (all stores)", xaxis_title="Recall", yaxis_title="Precision")
        q1, q2 = st.columns(2)
        with q1:
            show_fig(roc)
        with q2:
            show_fig(pr)
        cal = go.Figure()
        for m in MODELS:
            cal.add_scatter(x=cv[m]["calib_pred"], y=cv[m]["calib_true"], name=m, mode="lines+markers")
        cal.add_scatter(x=[0, 1], y=[0, 1], name="Perfect", mode="lines", line=dict(dash="dot"))
        cal.update_layout(title="Calibration: when the model says X% risk, how often is it right?",
                          xaxis_title="Predicted risk", yaxis_title="Actual share at risk")
        fi = pd.read_csv(C.MODELS_DIR / "feature_importance.csv")
        f1_, f2_ = st.columns(2)
        with f1_:
            show_fig(cal)
        with f2_:
            top = fi.sort_values("decision_tree", ascending=False).head(12)
            show_fig(px.bar(top, x="decision_tree", y="feature", orientation="h", title="What drives the Decision Tree",
                            labels={"decision_tree": "Importance"}).update_layout(yaxis=dict(autorange="reversed")))
        topa = fi.sort_values("ann_permutation", ascending=False).head(12)
        show_fig(px.bar(topa, x="ann_permutation", y="feature", orientation="h", title="What drives the ANN (permutation importance)",
                        labels={"ann_permutation": "Drop in F1 when shuffled"}).update_layout(yaxis=dict(autorange="reversed")))
        ev = pd.read_csv(C.MODELS_DIR / "metrics_by_evening.csv")
        show_fig(px.line(ev, x="evening", y="accuracy", color="model", markers=True,
                         labels={"evening": "Evening since delivery", "accuracy": "Accuracy"}, title="Accuracy by evening"))
        show_fig(px.line(ev, x="evening", y="recall", color="model", markers=True,
                         labels={"evening": "Evening since delivery", "recall": "Recall"}, title="Recall by evening"))
        st.warning("**Be careful with the later evenings.** The simulated sales path is built from each batch's real final "
                   "sales, so by the last evenings most of the answer is already visible and accuracy looks very high. "
                   "The **first evening** is the honest number.")

    with st.expander("e) Threshold slider and cost view"):
        tp_ = pd.read_csv(C.MODELS_DIR / "test_predictions.csv.gz")
        if view != "All stores 2024":
            tp_ = tp_[tp_.store_id == C.STORE_ID]
        first = tp_[tp_.evening == 1]                     # one row per batch = the decision made on arrival
        thr = st.slider("Flag a batch when risk is above", 0.1, 0.9, 0.5, 0.05)
        cost_rows = []
        for m, colp in zip(MODELS, ["p_dt", "p_ann"]):
            flag = first[colp] >= thr
            miss = first[(~flag) & (first.y == 1)]
            fa = first[flag & (first.y == 0)]
            cost_rows.append({"Model": m, "Caught": int((flag & (first.y == 1)).sum()), "Missed": len(miss),
                              "False alarms": len(fa),
                              "Recall": (flag & (first.y == 1)).sum() / max((first.y == 1).sum(), 1),
                              "Cost of missed batches": miss.final_waste_cost.sum(),
                              "Cost of false alarms": (C.FALSE_ALARM_MARKDOWN_PCT / 100 * fa.base_price * fa.qty_left).sum()})
        cdf = pd.DataFrame(cost_rows)
        cdf["Total cost"] = cdf["Cost of missed batches"] + cdf["Cost of false alarms"]
        show_df(cdf.round({"Recall": 3, "Cost of missed batches": 0, "Cost of false alarms": 0, "Total cost": 0}))
        st.caption(f"Measured on the first evening of every batch. A missed batch costs the stock that is wasted at its cost price. "
                   f"A false alarm is assumed to cost {C.FALSE_ALARM_MARKDOWN_PCT}% of the price on the stock marked down. "
                   "Both are simple assumptions.")
        ths = np.arange(0.1, 0.91, 0.05)
        curve = []
        for m, colp in zip(MODELS, ["p_dt", "p_ann"]):
            for t_ in ths:
                flag = first[colp] >= t_
                curve.append({"Model": m, "Threshold": t_,
                              "Total cost": first[(~flag) & (first.y == 1)].final_waste_cost.sum() +
                                            (C.FALSE_ALARM_MARKDOWN_PCT / 100 * first[flag & (first.y == 0)].base_price *
                                             first[flag & (first.y == 0)].qty_left).sum()})
        show_fig(px.line(pd.DataFrame(curve), x="Threshold", y="Total cost", color="Model", title="Total cost by threshold"))

    with st.expander("f) Speed"):
        sp = mv.loc[MODELS, ["train_seconds", "predict_ms_per_1000_rows"]].round(3).reset_index()
        show_df(sp.rename(columns={"model": "Model", "train_seconds": "Training time (s)",
                                   "predict_ms_per_1000_rows": "Prediction time (ms per 1,000 rows)"}))
        st.write("The Decision Tree trains faster and can explain a prediction. The ANN is usually a little more accurate.")

    with st.expander("g) About the two test views"):
        show_df(met[["view", "model", "rows", "accuracy", "recall", "f1", "roc_auc"]].round(3))
        st.write("The all-stores view is large and reliable. The single-store view is smaller, so its numbers move around more.")
