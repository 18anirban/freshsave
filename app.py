# ====================================================================
# FreshSave - Model Train Details  (one tab, built for explaining to sir)
# Run:  streamlit run app.py
# ====================================================================
import pandas as pd
import streamlit as st

from src import charts as C
from src.model_deployment_train.model_comparison import load_results, train_all

st.set_page_config(page_title="FreshSave - Model Train Details", layout="wide")
(tab,) = st.tabs(["Model Train Details"])

pct = lambda x: f"{x * 100:.1f}%"


def table(df, fmt=None):
    st.dataframe(df.style.format(fmt) if fmt else df, hide_index=True)


# ---------------------------------------------------------------- sections
def headline(r):
    mt, w = r["metrics"], r["winner"]
    st.success(f"Best model: **{w}**  -  it catches {pct(mt.loc[w, 'recall'])} of batches that really end up wasted, "
               f"with {pct(mt.loc[w, 'accuracy'])} accuracy on data it has never seen.")
    c = st.columns(5)
    for col, (label, m) in zip(c, [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"),
                                   ("F1", "f1"), ("ROC-AUC", "roc_auc")]):
        col.metric(label, f"{mt.loc[w, m]:.3f}", f"KNN {mt.loc['KNN', m]:.3f}", delta_color="off")
    st.caption("Big number = winner. Small line = KNN, for comparison. All scores are on the 30% test data only.")
    with st.expander("Talk track for sir (read top to bottom)"):
        st.markdown(f"""
1. **Problem** - a store receives fresh products with a short life. If too much is left at the end, it is thrown away. We predict, every evening, which batches will end up wasted.
2. **Data** - {r['profile']['rows_total']:,} daily rows from {r['profile']['batches_total']:,} batches in {r['profile']['stores']} stores (simulated, built to behave like a real retailer).
3. **Split** - train on the older 70% of batches, test on the newer 30%. No batch is in both.
4. **Preparation** - fill blanks, add 3 helpful columns, one-hot the category; KNN also gets log + scaling because it measures distance.
5. **Two models** - KNN (votes of similar past cases) and Decision Tree (a flowchart of rules).
6. **Result** - {r['winner']} wins. Both beat the simple guess; scores are in section 5.
7. **Limits** - data is simulated, so real store data is the next step (section 8).""")


def sec_data(r):
    p = r["profile"]
    st.header("1. The data")
    c = st.columns(4)
    c[0].metric("Rows", f"{p['rows_total']:,}"); c[1].metric("Batches", f"{p['batches_total']:,}")
    c[2].metric("Stores", p["stores"]); c[3].metric("Categories", p["categories"])
    st.markdown("**One row = one batch on one evening of its shelf life.** The label `at_risk` = 1 when 20% or more of the batch ends up wasted. "
                "Simulated data, built to behave like a real retailer (weekday effects, late deliveries, demand shocks, stock-count errors, missing values).")
    st.subheader("Train / test split (by time, by batch)")
    c1, c2 = st.columns([3, 2])
    c1.pyplot(C.split_timeline(p))
    c2.markdown(f"""
| | Train | Test |
|---|---|---|
| Rows | {p['rows_train']:,} | {p['rows_test']:,} |
| Batches | {p['batches_train']:,} | {p['batches_test']:,} |
| From | {p['train_dates'][0]} | {p['test_dates'][0]} |
| To | {p['train_dates'][1]} | {p['test_dates'][1]} |

Batches in both: **{p['batch_overlap']}**""")
    st.caption("Test = the future. Random row splitting would leak: yesterday and today of the same batch look alike. Splitting by time and batch avoids that.")
    st.subheader("Columns")
    table(pd.DataFrame(r["column_info"], columns=["Column", "Role", "Meaning"]))
    st.subheader("Sample rows")
    st.dataframe(r["sample_rows"], hide_index=True)


def sec_structure(r):
    p = r["profile"]
    st.header("2. How the data is structured and prepared")
    c1, c2 = st.columns(2)
    c1.markdown("""
**Pipeline (same for both models)**
1. Load train and test files
2. Remove ID columns and outcome columns from the inputs
3. Add 3 derived columns
4. Fill missing values with the **train** median
5. One-hot encode `category` (8 columns)
6. KNN only: log-squeeze skewed columns, then scale 0-1
7. Tune settings with cross-validation on **train only**
8. Score on test **once**
""")
    c2.markdown("""
**Derived columns**
- `sold_so_far` = units received - units left
- `cover_gap` = units left / daily demand - days left  (positive = more stock than days can sell)
- `margin_pct` = (price - cost) / price

**Never used as inputs**
- `units_wasted_final`, `waste_cost` - they are the future
- `batch_id`, `store_id`, `date`, `sub_category` - identifiers
""")
    st.subheader("Class balance")
    c1, c2 = st.columns([2, 3])
    c1.pyplot(C.class_balance(p))
    c2.markdown(f"About **{pct(p['risk_rate_train'])}** of rows are at risk, so the classes are fairly balanced and plain accuracy is meaningful. "
                f"Per batch, {pct(p['risk_rate_batch_train'])} (train) and {pct(p['risk_rate_batch_test'])} (test) of batches are at risk. "
                f"A model that always says 'safe' would score {pct(1 - p['risk_rate_test'])} - that is the floor to beat.")
    if p["missing"]:
        st.caption("Missing values found: " + ", ".join(f"{k} = {v:,}" for k, v in p["missing"].items()))


def sec_skew(r):
    st.header("3. Skewness")
    st.markdown("**Skewness** tells whether a column has a long tail (a few very large values). |skew| above 1 is called strongly skewed. "
                "KNN uses distances, so a long tail lets a few big values dominate; we squeeze it with `log1p`. A decision tree only compares thresholds, so it ignores skew.")
    sk = r["skew"].copy()
    sk["skew after log1p"] = sk["skew after log1p"].round(2)
    table(sk)
    st.markdown("**Log applied to (KNN only):** " + ", ".join(f"`{c}`" for c in r["log_cols"]) +
                "  - chosen only when |skew| > 1 and log really reduces it. Binary `is_promoted` is left alone.")
    col = st.selectbox("See before / after for a column", r["log_cols"])
    st.pyplot(C.skew_before_after(r, col))


def sec_models(r):
    k, t = r["knn"], r["tree"]
    st.header("4. The two models and their parameters")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("KNN")
        st.markdown("Finds the **k most similar** past batch-days and lets them vote.")
        st.markdown(f"- Chosen **k = {k['best_k']}**, weights = **{k['best_weights']}** (closer neighbours count more)\n"
                    f"- Searched k in {r['k_range']} x weights in [uniform, distance]\n"
                    f"- Training time {k['train_seconds']:.2f}s, prediction time on test {k['predict_seconds']:.1f}s")
        st.pyplot(C.knn_k_chart(k["grid"]))
        st.caption("Red circle = best. Bars show spread across the 5 folds. Very small k is noisy; very large k blurs.")
    with c2:
        st.subheader("Decision Tree")
        st.markdown("Learns a **flowchart of yes/no questions**.")
        st.markdown(f"- Chosen **max_depth = {t['best_depth']}**, **min_samples_leaf = {t['best_leaf']}**\n"
                    f"- Searched depth in {r['tree_depths']} x leaf size in [10, 30, 60]\n"
                    f"- Training time {t['train_seconds']:.2f}s, prediction time on test {t['predict_seconds']:.2f}s")
        st.pyplot(C.tree_depth_chart(t["grid"], t["best_leaf"]))
        st.caption("Grey keeps rising (memorising the past) while orange flattens - that gap is overfitting, so we stop at the best orange point.")
    with st.expander("Full search tables"):
        table(k["grid"].sort_values("cv_f1", ascending=False), {"cv_f1": "{:.4f}", "cv_std": "{:.4f}"})
        table(t["grid"].sort_values("cv_f1", ascending=False), {"cv_f1": "{:.4f}", "train_f1": "{:.4f}"})
    st.caption("Settings were chosen by 5-fold GroupKFold (all days of one batch stay in the same fold) on the training data only.")


def sec_results(r):
    mt = r["metrics"]
    st.header("5. Results on the test data")
    st.pyplot(C.metrics_bars(mt))
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "specificity", "balanced_accuracy", "mcc", "log_loss", "brier"]
    t = mt[cols].astype(float).reset_index().rename(columns={"index": "model"})
    table(t, {c: "{:.3f}" for c in cols})
    st.subheader("Confusion matrices")
    c = st.columns([1, 1, 2])
    c[0].pyplot(C.confusion(mt, "KNN")); c[1].pyplot(C.confusion(mt, "Decision Tree"))
    c[2].pyplot(C.roc(r["curves"], mt))
    st.subheader("Overfitting check")
    gap = pd.DataFrame(r["fit_gap"]).T.reset_index().rename(columns={"index": "model"})
    table(gap, {"train_accuracy": "{:.3f}", "test_accuracy": "{:.3f}"})
    st.caption("KNN with distance weights scores 100% on its own training rows (each row is its own nearest neighbour), so only the test column is honest.")
    st.subheader("Accuracy by evening / category / store")
    t1, t2, t3 = st.tabs(["Evening of shelf life", "Category", "Store"])
    with t1:
        st.pyplot(C.group_bar(r["breakdowns"]["evening_group"], "evening_group"))
        st.caption("Later evenings are easier: the stock left shows the outcome more clearly. Evening 1 is the hardest and the one that matters most.")
    with t2:
        st.pyplot(C.group_bar(r["breakdowns"]["category"], "category"))
    with t3:
        st.pyplot(C.group_bar(r["breakdowns"]["store_id"], "store_id"))
    with st.expander("What the words mean"):
        st.markdown("""
- **Accuracy** - out of all rows, how many were right.
- **Precision** - when it warns 'at risk', how often it is right (low = many false alarms).
- **Recall** - of the batches that really end up wasted, how many it caught (low = missed waste). Matters most here.
- **F1** - balance of precision and recall.
- **ROC-AUC** - how well it ranks risky above safe over all thresholds (1 = perfect, 0.5 = guess).
- **Specificity** - of safe batches, how many it left alone.
- **MCC / Balanced accuracy** - fair scores when classes are uneven.
- **Log loss / Brier** - quality of the probability itself (lower is better).
- **TP / FP / TN / FN** - caught / false alarm / correctly left / missed.""")


def sec_compare(r):
    mt = r["metrics"]
    st.header("6. Detailed model comparison")
    st.info(f"Winner: **{r['winner']}**  (rule: highest average of recall and F1; scores {r['winner_scores']}). "
            "Recall counts because a missed at-risk batch is real money lost.")
    st.subheader("Against simple baselines")
    cols = ["accuracy", "precision", "recall", "f1"]
    t = mt[cols].astype(float).reset_index().rename(columns={"index": "model"})
    table(t, {c: "{:.3f}" for c in cols})
    st.caption("The simple rule (stock cover longer than days left) already catches almost everything but raises many false alarms. "
               "The models are better at being right about both sides: higher accuracy and precision.")
    st.subheader("Money view - first evening of each batch (when the manager can still act)")
    st.pyplot(C.money_bar(r["money"]))
    m = r["money"].copy()
    table(m, {"accuracy": "{:.3f}", "recall": "{:.3f}", "share caught": "{:.1%}", "waste $ caught": "${:,.0f}", "waste $ missed": "${:,.0f}"})
    st.subheader("Do the models agree?")
    table(r["agreement"])
    st.subheader("Threshold trade-off")
    st.pyplot(C.threshold_chart(r["thresholds"]))
    st.caption("Lower threshold = warn more often: recall goes up, precision down. The store can choose where to sit.")
    c = st.columns(2)
    for col, n in zip(c, ["KNN", "Decision Tree"]):
        col.markdown(f"**{n}**")
        with col:
            table(r["thresholds"][n], {x: "{:.3f}" for x in ["accuracy", "precision", "recall", "f1"]})
    st.subheader("Speed and practicality")
    table(pd.DataFrame([
        {"model": "KNN", "train s": r["knn"]["train_seconds"], "predict s (test set)": r["knn"]["predict_seconds"], "explains itself by": "showing similar past cases"},
        {"model": "Decision Tree", "train s": r["tree"]["train_seconds"], "predict s (test set)": r["tree"]["predict_seconds"], "explains itself by": "readable rules"}]),
        {"train s": "{:.2f}", "predict s (test set)": "{:.2f}"})


def sec_learned(r):
    st.header("7. What the models learned")
    st.subheader("Decision Tree - top rules")
    st.code(r["tree"]["rules"])
    st.pyplot(C.importance_bar(r["tree"]["importance"]))
    st.subheader("Four example batches (first evening), explained by both models")
    for ex in r["examples"]:
        with st.expander(f"{ex['label']}  -  truth: {'AT RISK' if ex['truth'] else 'safe'}"):
            row = ex["row"]
            st.write(f"{row['category']} / {row['sub_category']}, days left {row['days_left']}, units left {row['units_left']}, daily demand {row['avg_daily_demand']}")
            c1, c2 = st.columns(2)
            c1.markdown(f"**KNN** says {pct(ex['knn_prob'])} risk. Its nearest past cases:")
            c1.dataframe(ex["neighbors"], hide_index=True)
            c2.markdown(f"**Tree** says {pct(ex['tree_prob'])} risk. Questions it asked:")
            c2.code("\n".join(ex["tree_path"]))


def sec_limits(r):
    st.header("8. Honest limits")
    st.markdown("""
- The data is **simulated** to behave like a retailer; scores on real store data will differ. Real data is the next step.
- Test set was used **once**; nothing was tuned on it.
- Accuracy is higher on later evenings than on evening 1, so judge the model on evening 1.
- The simple rule already catches most waste; the models add fewer false alarms, not magic.
- Outcome columns are never inputs, so the model cannot cheat by seeing the future.
""")


# ---------------------------------------------------------------- page
with tab:
    st.title("Model Train Details")
    st.caption("FreshSave - predicting which fresh batches will be wasted, using KNN and Decision Tree")
    res = load_results()
    if res is None:
        st.warning("Models are not trained yet.")
        if st.button("Train now (about 1 minute)"):
            with st.spinner("Training KNN and Decision Tree..."):
                train_all()
            st.rerun()
        st.stop()
    for fn in (headline, sec_data, sec_structure, sec_skew, sec_models, sec_results, sec_compare, sec_learned, sec_limits):
        fn(res)
