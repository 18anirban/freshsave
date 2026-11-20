"""
tab_models.py - TAB 5: MODEL COMPARISON  (for the analyst and for sir)
=======================================================================
Shows how KNN and the Decision Tree were prepared, tested and compared, and explains how each one reaches an
answer. The model that scores the batches is chosen on the Setup tab.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config as C
from src import features as F, scoring
from src.models import REGISTRY, decision_tree_model, knn_model
from ui.helpers import read_json, show_df, show_fig

MODELS = list(scoring.MODEL_FILES)


# ============================================================
# STEP 1: DATA PROFILE AND HOW THE DATA WAS PREPARED
# ============================================================
def _profile_block(prof):
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
- **Transformations:** log1p on skewed columns (both models), category one-hot encoded, 0-1 scaling for KNN only (it measures distances).
- **{len(prof['features'])} features:** {', '.join(prof['features'])}
""")
        st.markdown("**Columns deliberately left out, and why**")
        show_df(pd.DataFrame({"Column": list(prof["excluded"]), "Reason": list(prof["excluded"].values())}))

# ------------------------------------------------------------
# WHAT WE DID IN STEP 1:
# - Kept the data profile and the preparation notes; the only text change is "scaling for KNN only".
# ------------------------------------------------------------


# ============================================================
# STEP 2: METRICS SIDE BY SIDE
# ============================================================
def _metrics_block(mv, view):
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

# ------------------------------------------------------------
# WHAT WE DID IN STEP 2:
# - Showed every test metric for the two models in one table and one chart.
# ------------------------------------------------------------


# ============================================================
# STEP 3: VISUALS (WHAT HAPPENED, CURVES, WHAT DRIVES EACH MODEL)
# ============================================================
def _visuals_block(mv):
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
        show_fig(cal)

        fi = pd.read_csv(C.MODELS_DIR / "feature_importance.csv")
        for m in MODELS:                                   # one importance chart per model, named by the model file
            mod = REGISTRY[m]
            top = fi.sort_values(mod.IMPORTANCE_COLUMN, ascending=False).head(12)
            show_fig(px.bar(top, x=mod.IMPORTANCE_COLUMN, y="feature", orientation="h", title=f"What drives the {m}",
                            labels={mod.IMPORTANCE_COLUMN: mod.IMPORTANCE_LABEL}).update_layout(yaxis=dict(autorange="reversed")))

        ev = pd.read_csv(C.MODELS_DIR / "metrics_by_evening.csv")
        show_fig(px.line(ev, x="evening", y="accuracy", color="model", markers=True,
                         labels={"evening": "Evening since delivery", "accuracy": "Accuracy"}, title="Accuracy by evening"))
        show_fig(px.line(ev, x="evening", y="recall", color="model", markers=True,
                         labels={"evening": "Evening since delivery", "recall": "Recall"}, title="Recall by evening"))
        st.warning("**Be careful with the later evenings.** The simulated sales path is built from each batch's real final "
                   "sales, so by the last evenings most of the answer is already visible and accuracy looks very high. "
                   "The **first evening** is the honest number.")

# ------------------------------------------------------------
# WHAT WE DID IN STEP 3:
# - Kept the charts. The "what drives the model" charts now loop over the model registry, so each model's
#   chart and axis text come from its own file (src/models/).
# ------------------------------------------------------------


# ============================================================
# STEP 4: THRESHOLD SLIDER AND COST VIEW
# ------------------------------------------------------------
# "Flag a batch when risk is above X": how many at-risk batches are caught, how many are missed,
# and what the misses and the false alarms cost. Measured on the FIRST evening of every batch.
# ============================================================
def _threshold_block(view):
    pairs = [(m, REGISTRY[m].PROB_COLUMN) for m in MODELS]
    with st.expander("e) Threshold slider and cost view"):
        tp_ = pd.read_csv(C.MODELS_DIR / "test_predictions.csv.gz")
        if view != "All stores 2024":
            tp_ = tp_[tp_.store_id == C.STORE_ID]
        first = tp_[tp_.evening == 1]                     # one row per batch = the decision made on arrival
        thr = st.slider("Flag a batch when risk is above", 0.1, 0.9, 0.5, 0.05)
        cost_rows = []
        for m, colp in pairs:
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
        for m, colp in pairs:
            for t_ in ths:
                flag = first[colp] >= t_
                curve.append({"Model": m, "Threshold": t_,
                              "Total cost": first[(~flag) & (first.y == 1)].final_waste_cost.sum() +
                                            (C.FALSE_ALARM_MARKDOWN_PCT / 100 * first[flag & (first.y == 0)].base_price *
                                             first[flag & (first.y == 0)].qty_left).sum()})
        show_fig(px.line(pd.DataFrame(curve), x="Threshold", y="Total cost", color="Model", title="Total cost by threshold"))

# ------------------------------------------------------------
# WHAT WE DID IN STEP 4:
# - Kept the cost view. The probability column of each model is read from its own file (p_dt, p_knn).
# ------------------------------------------------------------


# ============================================================
# STEP 5: SPEED AND THE TWO TEST VIEWS
# ============================================================
def _speed_and_views(mv, met):
    with st.expander("f) Speed"):
        sp = mv.loc[MODELS, ["train_seconds", "predict_ms_per_1000_rows"]].round(3).reset_index()
        show_df(sp.rename(columns={"model": "Model", "train_seconds": "Training time (s)",
                                   "predict_ms_per_1000_rows": "Prediction time (ms per 1,000 rows)"}))
        st.write("The Decision Tree trains and predicts quickly and can explain a prediction. KNN needs almost no training, "
                 "but is slower to predict, because it compares every new batch with thousands of stored cases.")

    with st.expander("g) About the two test views"):
        show_df(met[["view", "model", "rows", "accuracy", "recall", "f1", "roc_auc"]].round(3))
        st.write("The all-stores view is large and reliable. The single-store view is smaller, so its numbers move around more.")

# ------------------------------------------------------------
# WHAT WE DID IN STEP 5:
# - Kept both blocks; the speed text now describes KNN instead of the ANN.
# ------------------------------------------------------------


# ============================================================
# STEP 6: HOW THE MODELS WORK (FOR EXPLAINING TO SIR)  (NEW)
# ------------------------------------------------------------
# Decision Tree : the first levels of the learned yes/no rules, printed as text.
# KNN           : pick a batch from the latest uploaded day and see the cases that voted on it.
# ============================================================
def _explain_block(ctx):
    with st.expander("h) How the models work (explain this to sir)"):
        names = read_json("feature_names.json")

        st.markdown("**Decision Tree: the first questions it asks**")
        st.write("Read from the top: if the condition is true, take the first branch; otherwise the second. "
                 "`class: 1` means at risk of waste, `class: 0` means not at risk.")
        tree = scoring.load_model("Decision Tree")
        st.code(decision_tree_model.rules_text(tree, names))

        st.markdown("**KNN: the past cases that voted on one batch**")
        live = ctx.scores[ctx.scores.days_left > 0] if ctx.have_data else pd.DataFrame()
        if live.empty:
            st.caption("Upload a day first, then pick a batch here.")
            return
        live = live[live.day_no == live.day_no.max()]
        labels = [f"{p} ({b})" for p, b in zip(live.product_name, live.batch_id)]
        pick = st.selectbox("Pick a batch from the latest day", labels)
        row = live.iloc[[labels.index(pick)]]
        X, _ = F.build_X(row)
        X = X[names]
        model = scoring.load_model("KNN")
        risk = float(model.predict_proba(X)[0, 1])
        st.write(f"KNN looks at the closest remembered cases. Their votes give this batch a risk of **{risk:.0%}**.")
        show_df(knn_model.explain_neighbours(model, X, names))

# ------------------------------------------------------------
# WHAT WE DID IN STEP 6:
# - Added a section that opens the "black box": printed rules for the tree, and the neighbours behind
#   one KNN prediction, so each model can be explained with a real example.
# ------------------------------------------------------------


# ============================================================
# STEP 7: THE WHOLE MODEL COMPARISON TAB
# ============================================================
def render(ctx):
    info = read_json("active_model.json")
    prof = read_json("data_profile.json")
    met = pd.read_csv(C.MODELS_DIR / "metrics_summary.csv")
    st.subheader("How the two models were tested")
    st.info(f"Model scoring the batches now: **{ctx.active}**. Change it on the Setup tab.")
    st.success(f"Default winner: **{info['winner']}**. {info['why']}")
    st.caption("Both models were trained on 2023 data from all 50 stores and tested on 2024. The demo store's 2024 "
               "batches were never used for training.")

    view = st.radio("Test data", ["All stores 2024", f"{C.STORE_ID} only (2024)"], horizontal=True)
    mv = met[met.view == view].set_index("model")

    _profile_block(prof)
    _metrics_block(mv, view)
    _visuals_block(mv)
    _threshold_block(view)
    _speed_and_views(mv, met)
    _explain_block(ctx)

# ------------------------------------------------------------
# WHAT WE DID IN STEP 7:
# - The tab no longer has the model chooser (it moved to Setup). Everything else is split into blocks.
# ------------------------------------------------------------
