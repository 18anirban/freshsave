# ====================================================================
# FreshSave - Model Train Details  (one tab, built for explaining to sir)
# Run:  streamlit run app.py
# ====================================================================
import pandas as pd
import streamlit as st

from src import charts as C
from src.model_deployment_train.model_comparison import load_results, train_all
from src.demo.page import demo_page
from src.landing import landing

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
    st.info("In simple words: this is the history of products we use to teach the models, and the part we hide to test them.")
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
    st.info("In simple words: we clean the data and turn it into numbers the models can use. The table below lists all 13 steps and why each is done.")
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
    miss = pi["missing_train"]
    mtxt = ", ".join(f"{k} ({v:,} blanks, median {pi['medians'][k]})" for k, v in miss.items()) or "none"
    table(pd.DataFrame([
        {"Step": 1, "What we did": "Loaded train and test files", "Why": f"Train {p['rows_train']:,} rows to learn, test {p['rows_test']:,} rows kept hidden to check"},
        {"Step": 2, "What we did": "Set the label `at_risk`", "Why": "This is what the models predict: 1 = 20% or more of the batch is wasted"},
        {"Step": 3, "What we did": "Removed outcome columns (units_wasted_final, waste_cost)", "Why": "They are only known after the batch ends; using them would be cheating"},
        {"Step": 4, "What we did": "Removed ID columns (batch_id, store_id, date, sub_category)", "Why": "They name things, they do not describe behaviour"},
        {"Step": 5, "What we did": "Kept 10 original input columns", "Why": "Shelf life, days left, units, demand, prices, handling score, promotion"},
        {"Step": 6, "What we did": "Added 3 derived columns (sold_so_far, cover_gap, margin_pct)", "Why": "They express 'will this stock sell in time?' directly"},
        {"Step": 7, "What we did": "One-hot encoded category into 8 yes/no columns", "Why": "Models need numbers, and category has no natural order"},
        {"Step": 8, "What we did": f"Filled blanks with the train median: {mtxt}", "Why": "Models cannot read empty cells; median is not pulled by extreme values"},
        {"Step": 9, "What we did": "KNN only: log on skewed columns (section 3)", "Why": "Stops a few huge values from dominating distances"},
        {"Step": 10, "What we did": "KNN only: scaled every column to 0-1", "Why": "Gives every column equal weight in the distance"},
        {"Step": 11, "What we did": "Chose settings with 5-fold cross-validation on train only", "Why": "Picks k (KNN) and depth (Tree) fairly, without touching test"},
        {"Step": 12, "What we did": "Trained the final models", "Why": "Used the chosen settings on all train rows"},
        {"Step": 13, "What we did": "Tested once on the hidden test data", "Why": "Gives honest scores on data the models never saw"}]))
    with st.expander("See the min and max used for scaling (step 10)"):
        st.dataframe(pi["scaler_ranges"].round(3), hide_index=True)
    with st.expander("See all 21 model inputs"):
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
    st.info("In simple words: some columns have a few very big values. KNN gets confused by them, so we shrink them. The Decision Tree does not need this.")
    st.markdown("**Skewness** shows whether a column is balanced or has a long tail of a few very large values. "
                "Skew near 0 is balanced; above +1 or below -1 is strongly skewed.")
    st.pyplot(C.skew_shapes())
    st.markdown("- **KNN** works on distances, so big values dominate. We shrink skewed columns with `log1p`.\n"
                "- **Decision Tree** only asks 'is it above a number?', so skew does not matter. It uses raw values.")
    st.subheader("Skewness of all numeric columns")
    st.pyplot(C.skew_overview(r["skew"]))
    st.caption("Grey = raw data. Blue = what KNN sees after log. Bars outside the red lines are strongly skewed.")
    sk = r["skew"].rename(columns={"skewness": "skew (raw)"})[["column", "skew (raw)", "log applied", "skew used by KNN"]]
    table(sk)
    st.markdown("Log is used only on columns with skew above 1 and no negative values, where it really helps: " +
                ", ".join(f"`{c}`" for c in r["log_cols"]) + ".")
    st.subheader("Before and after")
    col = st.selectbox("Pick a column", r["log_cols"])
    st.pyplot(C.skew_before_after(r, col))


def sec_models(r):
    k, t = r["knn"], r["tree"]
    st.header("4. The two models and their parameters")
    st.info("In simple words: KNN asks 'which past cases look like this one?'. The Decision Tree asks a chain of yes/no questions. We tried several settings and kept the best.")
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
    st.caption("Settings were chosen by 5-fold GroupKFold (all days of one batch stay in the same fold) on the training data only.")


def sec_results(r):
    mt = r["metrics"]
    st.header("5. Results on the test data")
    st.info("In simple words: we ask both models about batches they have never seen, and check how often they are right.")
    st.pyplot(C.metrics_bars(mt))
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    t = mt.loc[["KNN", "Decision Tree"], cols].astype(float).reset_index().rename(columns={"index": "model"})
    table(t, {c: "{:.3f}" for c in cols})
    with st.expander("What the words mean", expanded=True):
        st.markdown("""
- **Accuracy** - out of all rows, how many were right.
- **Precision** - when the model warns 'at risk', how often it is right.
- **Recall** - of the batches that really end up wasted, how many the model caught.
- **F1** - one number that balances precision and recall.
- **ROC-AUC** - how well the model puts risky batches above safe ones (1 = perfect, 0.5 = guessing).""")
    st.subheader("Confusion matrices")
    st.caption("Each box counts rows. Top-left and bottom-right are correct. The other two are mistakes: a false alarm, or a missed at-risk batch.")
    c = st.columns(2)
    c[0].pyplot(C.confusion(mt, "KNN")); c[1].pyplot(C.confusion(mt, "Decision Tree"))
    st.subheader("Accuracy by evening of shelf life")
    st.pyplot(C.group_bar(r["breakdowns"]["evening_group"], "evening_group"))
    st.caption("Evening 1 = the day after delivery. Later evenings are easier because the stock left shows more clearly what will happen.")


def sec_compare(r):
    mt, w = r["metrics"], r["winner"]
    lo = "KNN" if w == "Decision Tree" else "Decision Tree"
    st.header("6. Detailed model comparison")
    st.info("In simple words: the two models side by side, and which one we pick.")
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
def details():
    if st.button("< Back"):
        st.session_state["page"] = "home"
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


page = st.session_state.get("page", "home")
if page == "details":
    details()
elif page == "demo":
    demo_page()
else:
    landing()
