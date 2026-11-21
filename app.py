# ====================================================================
# FreshSave - Model Train Details  (one tab, built for explaining to sir)
# Run:  streamlit run app.py
# ====================================================================
import pandas as pd
import streamlit as st

from src import charts as C
from src.model_deployment_train.model_comparison import load_results, train_all

st.set_page_config(page_title="FreshSave - Model Train Details", layout="wide")

pct = lambda x: f"{x * 100:.1f}%"


def table(df, fmt=None):
    st.dataframe(df.style.format(fmt) if fmt else df, hide_index=True)


# ---------------------------------------------------------------- sections
def talk_track(r):
    with st.expander("Talk track for sir (read top to bottom)"):
        st.markdown(f"""
1. **Problem** - a store receives fresh products with a short life. If too much is left at the end, it is thrown away. We predict, every evening, which batches will end up wasted.
2. **Data** - {r['profile']['rows_total']:,} daily rows from {r['profile']['batches_total']:,} batches in {r['profile']['stores']} stores (simulated, built to behave like a real retailer).
3. **Split** - train on the older 70% of batches, test on the newer 30%. No batch is in both.
4. **Preparation** - fill blanks, add 3 helpful columns, one-hot the category; KNN also gets log + scaling because it measures distance.
5. **Two models** - KNN (votes of similar past cases) and Decision Tree (a flowchart of rules).
6. **Result** - {r['winner']} wins. Both beat the simple guess; scores are in section 5.""")


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
    p, pi = r["profile"], r["prep_info"]
    st.header("2. How the data is structured and prepared")
    st.markdown("Every step below is done in the same order for both models. Steps marked **KNN only** are not needed by the Decision Tree.")

    st.subheader("Column funnel: from the file to the model input")
    nrow = lambda a, b, c: {"stage": a, "columns": b, "what happens": c}
    table(pd.DataFrame([
        nrow("Raw file", pi["n_input_cols"], "All columns of store_train_70.csv"),
        nrow("Remove 3 outcome columns", pi["n_input_cols"] - 3, "units_wasted_final, waste_cost, at_risk (at_risk is kept aside as the label y)"),
        nrow("Remove 4 ID columns", pi["n_input_cols"] - 7, "batch_id, store_id, date, sub_category"),
        nrow("Add 3 derived columns", pi["n_input_cols"] - 4, "sold_so_far, cover_gap, margin_pct"),
        nrow("One-hot the category", pi["n_features"], "category (1 column) becomes 8 yes/no columns"),
        nrow("Final model input X", pi["n_features"], "10 original numbers + 3 derived + 8 one-hot")]))

    st.subheader("Step by step")
    with st.expander("Step 1 - Load the data", expanded=True):
        st.markdown(f"Two files are read: **train** ({p['rows_train']:,} rows) and **test** ({p['rows_test']:,} rows). "
                    "The split was made by time and by batch, so the test set is the future and the model never sees it while learning.")
    with st.expander("Step 2 - Define the label (what we predict)", expanded=True):
        st.markdown(f"`at_risk` = 1 when 20% or more of a batch is wasted at the end of its life. It is the label **y**. "
                    f"In train {pct(p['risk_rate_train'])} of rows are at risk, in test {pct(p['risk_rate_test'])}.")
        st.pyplot(C.class_balance(p))
    with st.expander("Step 3 - Remove outcome columns (no cheating)", expanded=True):
        st.markdown("`units_wasted_final` and `waste_cost` are only known **after** the batch is over. If the model saw them it would simply copy the answer. They are never inputs.")
    with st.expander("Step 4 - Remove ID columns", expanded=True):
        st.markdown("`batch_id`, `store_id`, `date` are labels, not behaviour: a model could memorise them instead of learning patterns. "
                    "`sub_category` has 40 values and would add 40 columns for little gain; `category` (8 values) is enough.")
    with st.expander("Step 5 - Keep the original input columns", expanded=True):
        st.markdown("`shelf_life_days, days_left, units_received, units_left, avg_daily_demand, units_sold_today, cost_price, selling_price, handling_score, is_promoted`")
    with st.expander("Step 6 - Create 3 derived columns", expanded=True):
        st.markdown("These turn raw numbers into something close to the business question *'will this stock sell in time?'*")
        table(pd.DataFrame([
            {"new column": "sold_so_far", "formula": "units_received - units_left", "meaning": "How much of the batch has already sold"},
            {"new column": "cover_gap", "formula": "units_left / avg_daily_demand - days_left", "meaning": "Days of stock minus days of life. Above 0 = more stock than days left to sell it"},
            {"new column": "margin_pct", "formula": "(selling_price - cost_price) / selling_price", "meaning": "Profit share of the price"}]))
    with st.expander("Step 7 - One-hot encode category", expanded=True):
        st.markdown("Models need numbers, not words. `category` becomes 8 columns (Bakery, Beverages, Dairy, Deli, Meat, Produce, Ready_to_Eat, Seafood). "
                    "A Dairy row has `cat_Dairy = 1` and the other seven are 0. We do **not** give categories numbers like 1, 2, 3, because that would wrongly say Meat is 'bigger' than Dairy.")
    with st.expander("Step 8 - Fill missing values", expanded=True):
        miss = pi["missing_train"]
        st.markdown("Missing in train: " + (", ".join(f"`{k}` = {v:,} rows" for k, v in miss.items()) or "none") +
                    ". Each blank is filled with the **median** of that column in the **train** data only (the median is not pulled by extreme values). "
                    "The same train medians are reused for the test data, so nothing from test leaks in.")
        if miss:
            for k in miss:
                st.caption(f"Median used for {k}: {pi['medians'][k]}")
    with st.expander("Step 9 - KNN only: check skewness and apply log", expanded=True):
        st.markdown("Skewed columns (long tail) are squeezed with `log1p` so a few huge values do not dominate distances. Details and charts are in section 3. "
                    "Applied to: " + ", ".join(f"`{c}`" for c in r["log_cols"]) + ".")
    with st.expander("Step 10 - KNN only: scale every column to 0-1", expanded=True):
        st.markdown("KNN measures distance. `units_received` can be in hundreds while `is_promoted` is 0 or 1, so without scaling the big numbers would decide everything. "
                    "MinMax scaling gives each column the same 0-1 range: `(value - min) / (max - min)`. Min and max are learned from train only.")
        st.dataframe(pi["scaler_ranges"].round(3), hide_index=True)
    with st.expander("Step 11 - Choose settings with cross-validation (train only)", expanded=True):
        st.markdown("Train is cut into 5 folds, **keeping all days of one batch in the same fold**. The model learns on 4 folds and is checked on the 5th, rotating. "
                    "This picks k (KNN) and depth / leaf size (Tree) without touching the test data.")
    with st.expander("Step 12 - Train the final model", expanded=True):
        st.markdown("The chosen settings are used to train once on all train rows.")
    with st.expander("Step 13 - Test once", expanded=True):
        st.markdown(f"The trained model predicts the {p['rows_test']:,} test rows. Scores are computed once. Nothing is changed afterwards to improve them.")

    st.subheader("All 21 model inputs")
    table(pi["feature_table"])

    st.subheader("Follow one row through the pipeline")
    st.markdown("Original row from the training file:")
    st.dataframe(pi["row_example"], hide_index=True)
    c1, c2 = st.columns(2)
    c1.markdown("**After steps 3-8** (what the Decision Tree gets)")
    c1.dataframe(pi["row_features"].round(3))
    c2.markdown("**After steps 9-10** (what KNN gets, all between 0 and 1)")
    c2.dataframe(pi["row_scaled"].round(3))


def sec_skew(r):
    st.header("3. Skewness")
    st.subheader("What is skewness?")
    st.markdown("Skewness says whether the values of a column are balanced around the middle or have a **long tail on one side**. "
                "Example: most products have 20-100 units left, but a few have 2,000. Those few big values stretch the tail to the right.")
    st.pyplot(C.skew_shapes())
    st.markdown("""
- **Skew near 0** - balanced, nothing to do.
- **Skew above +1** - long right tail (a few very large values). Common for prices and quantities.
- **Skew below -1** - long left tail.
- **Rule used here:** |skew| above 1 counts as strongly skewed.

**Why it matters for our two models**
- **KNN** compares distances, so a few huge values dominate the distance and the rest of the data looks squeezed together. We apply `log1p` (log of 1 + value), which pulls long tails in.
- **Decision Tree** only asks 'is value above a number?', so skew does not hurt it. It uses raw values and the rules stay readable.
""")
    st.subheader("Overall skewness - all numeric columns")
    st.pyplot(C.skew_overview(r["skew"]))
    st.caption("Grey = raw data. Blue = what KNN sees after log1p (equal to grey where log was not applied). Bars outside the red lines are strongly skewed.")
    st.subheader("Every column's distribution (raw train data)")
    cols = list(r["skew"]["column"])
    st.pyplot(C.hist_grid(r["X_train_raw_all"], cols))
    st.subheader("Skewness table")
    sk = r["skew"].rename(columns={"skewness": "skew (raw)"})[["column", "skew (raw)", "skewed (|skew|>1)", "log applied", "skew used by KNN"]]
    table(sk)
    st.markdown("**Log is applied only when** the column is strongly skewed, has no negative values, and log really reduces the skew. "
                "`is_promoted` is a 0/1 column, so it is left alone, and `sold_so_far` has a few negative values (stock-count errors), so log cannot be used on it. "
                "Log can slightly overshoot to the other side (`units_left`, `units_sold_today` go a little negative because many values are 0), but it is still closer to balanced than before. Applied to: " + ", ".join(f"`{c}`" for c in r["log_cols"]) + ".")
    st.subheader("Before and after for one column")
    col = st.selectbox("Pick a column", r["log_cols"])
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
    mt, w = r["metrics"], r["winner"]
    lo = "KNN" if w == "Decision Tree" else "Decision Tree"
    st.header("6. Detailed model comparison")
    st.subheader("Final comparison table")
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    t = mt.loc[["KNN", "Decision Tree"], cols].astype(float)
    t["train time (s)"] = [r["knn"]["train_seconds"], r["tree"]["train_seconds"]]
    t["predict time (s)"] = [r["knn"]["predict_seconds"], r["tree"]["predict_seconds"]]
    t["best model"] = ["Winner" if n == w else "" for n in t.index]
    t = t.reset_index().rename(columns={"index": "model"})
    table(t, {**{c: "{:.3f}" for c in cols}, "train time (s)": "{:.2f}", "predict time (s)": "{:.2f}"})

    st.subheader("Conclusion of findings")
    rule = mt.loc["Simple rule (cover_gap > 0)"]; maj = mt.loc["Majority guess"]
    d = lambda m: (mt.loc[w, m] - mt.loc[lo, m]) * 100
    st.markdown(f"""
- **{w} is the better model.** It scores higher than {lo} on accuracy (+{d('accuracy'):.1f} points), recall (+{d('recall'):.1f}) and F1 (+{d('f1'):.1f}).
- **Recall decides the winner**, because a missed at-risk batch is waste the store could have avoided. {w} catches {pct(mt.loc[w, 'recall'])} of at-risk rows, {lo} catches {pct(mt.loc[lo, 'recall'])}.
- **Both models clearly beat a plain guess.** Always saying 'safe' is right {pct(maj['accuracy'])} of the time and catches nothing.
- **Both models are more accurate than the one-line rule** (stock cover longer than days left): {pct(rule['accuracy'])} accuracy and {pct(rule['precision'])} precision, against {pct(mt.loc[w, 'accuracy'])} and {pct(mt.loc[w, 'precision'])} for {w}.
- **Speed:** KNN trains in {r['knn']['train_seconds']:.2f}s but needs {r['knn']['predict_seconds']:.1f}s to predict the test set; the Tree trains in {r['tree']['train_seconds']:.2f}s and predicts in {r['tree']['predict_seconds']:.2f}s.
- **Explainability:** the Tree gives readable rules; KNN explains by showing similar past cases.
""")
    st.success(f"Best model: **{w}**  -  it catches {pct(mt.loc[w, 'recall'])} of batches that really end up wasted, "
               f"with {pct(mt.loc[w, 'accuracy'])} accuracy on data it has never seen.")
    c = st.columns(5)
    for col, (label, m) in zip(c, [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"),
                                   ("F1", "f1"), ("ROC-AUC", "roc_auc")]):
        col.metric(label, f"{mt.loc[w, m]:.3f}", f"{lo} {mt.loc[lo, m]:.3f}", delta_color="off")
    st.caption("Big number = winner. Small line = the other model. All scores are on the 30% test data only.")


# ---------------------------------------------------------------- pages
def landing():
    st.title("FreshSave")
    st.caption("Predicting which fresh batches will be wasted, using KNN and Decision Tree")
    st.write("")
    if st.button("Model Train Details", type="primary"):
        st.session_state["open_details"] = True
        st.rerun()


def details():
    if st.button("< Back"):
        st.session_state["open_details"] = False
        st.rerun()
    st.title("Model Train Details")
    res = load_results()
    if res is None:
        st.warning("Models are not trained yet.")
        if st.button("Train now (about 1 minute)"):
            with st.spinner("Training KNN and Decision Tree..."):
                train_all()
            st.rerun()
        st.stop()
    for fn in (talk_track, sec_data, sec_structure, sec_skew, sec_models, sec_results, sec_compare):
        fn(res)


if st.session_state.get("open_details", False):
    details()
else:
    landing()
