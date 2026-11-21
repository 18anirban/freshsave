# ====================================================================
# MODEL 3 : DETAILED MODEL COMPARISON  (+ the "train everything" entry)
# Runs KNN and Decision Tree, scores both on the SAME untouched test
# set, compares them with two simple baselines, and saves the results.
# Run:  python -m src.model_deployment_train.model_comparison
# ====================================================================
import joblib
import numpy as np
import pandas as pd

from src.data_prep import (feature_table, load_data, profile_data, skewness_table, build_features, apply_imputer,
                           fit_imputer, MODEL_DIR, TARGET, COLUMN_INFO, FEATURE_NAMES, NUMERIC_COLS)
from src.evaluation import evaluate, threshold_table, group_metrics, calibration
from src.model_deployment_train import knn_model, decision_tree_model

THRESHOLD = 0.5


# ============================================================
# STEP 1: Baselines - the bar a model must beat
# ============================================================
def baseline_probs(train_df, test_df):
    majority = int(train_df[TARGET].mean() >= 0.5)
    cover_gap = build_features(test_df)["cover_gap"]          # simple rule: stock lasts longer than shelf life
    return {"Majority guess": np.full(len(test_df), float(majority)),
            "Simple rule (cover_gap > 0)": (cover_gap > 0).astype(float).values}
# WHAT WE DID IN STEP 1: a "always say no" guess and a one-line business rule.


# ============================================================
# STEP 2: One metrics table for every model
# ============================================================
def metrics_table(y, probs):
    rows = {}
    for name, p in probs.items():
        r = evaluate(y, p, THRESHOLD)
        rows[name] = {k: v for k, v in r.items() if not k.endswith("_curve")}
    return pd.DataFrame(rows).T


# ============================================================
# STEP 3: Where is each model strong or weak?
# ============================================================
def breakdowns(test_df, probs):
    d = test_df.copy()
    d["evening"] = d["shelf_life_days"] - d["days_left"]
    d["evening_group"] = d["evening"].clip(upper=5).map(lambda e: "5+" if e >= 5 else str(e))
    out = {}
    for by in ["evening_group", "category", "store_id"]:
        frames = []
        for name, p in probs.items():
            g = group_metrics(d, p, by, THRESHOLD)
            g.insert(1, "model", name)
            frames.append(g)
        out[by] = pd.concat(frames, ignore_index=True)
    return out


# ============================================================
# STEP 4: Pick the winner (average of recall and F1)
# ============================================================
# Recall matters most - a missed at-risk batch is real money lost.
def pick_winner(mt):
    s = (mt.loc[["KNN", "Decision Tree"], "recall"] + mt.loc[["KNN", "Decision Tree"], "f1"]) / 2
    return s.idxmax(), s.round(4).to_dict()


# ============================================================
# STEP 5: Four example rows explained by both models
# ============================================================
def pick_examples(test_df, p_knn, p_tree, seed=7):
    d = test_df.reset_index(drop=True)
    first = ((d["shelf_life_days"] - d["days_left"]) == 1).values
    a, b, y = p_knn >= THRESHOLD, p_tree >= THRESHOLD, d[TARGET].values == 1
    cases = {"Both right - at risk": first & (a == y) & (b == y) & y,
             "Both right - safe": first & (a == y) & (b == y) & ~y,
             "Models disagree": first & (a != b),
             "Both wrong": first & (a != y) & (b != y)}
    rng, out = np.random.RandomState(seed), {}
    for label, mask in cases.items():
        idx = np.where(mask)[0]
        if len(idx):
            out[label] = int(rng.choice(idx))
    return out


# ============================================================
# STEP 6: Train everything, compare, save
# ============================================================
def train_all(save=True):
    train_df, test_df = load_data()
    knn, tree = knn_model.run(train_df, test_df), decision_tree_model.run(train_df, test_df)
    y = test_df[TARGET]
    probs = {"KNN": knn["prob_test"], "Decision Tree": tree["prob_test"], **baseline_probs(train_df, test_df)}
    mt = metrics_table(y, probs)
    winner, scores = pick_winner(mt)

    # train-vs-test accuracy shows overfitting (KNN with distance weights always scores 100% on its own rows)
    fit_gap = {n: {"train_accuracy": float(((r["prob_train"] >= THRESHOLD) == train_df[TARGET].values).mean()),
                   "test_accuracy": float(mt.loc[n, "accuracy"])} for n, r in [("KNN", knn), ("Decision Tree", tree)]}

    X_tr_tree = tree["X_train"]
    skew = skewness_table(X_tr_tree)
    used = knn["X_train"][NUMERIC_COLS].skew()          # skew KNN really sees (after log, scaling does not change skew)
    skew["skew used by KNN"] = skew["column"].map(used).round(2)
    skew["log applied"] = skew["column"].isin(knn["prep"]["log_cols"])
    imp = knn["prep"]["imputer"]
    prep_info = {
        "n_input_cols": len(train_df.columns), "n_features": len(FEATURE_NAMES),
        "missing_train": {c: int(v) for c, v in build_features(train_df).isna().sum().items() if v},
        "medians": dict(zip(FEATURE_NAMES, [round(float(x), 3) for x in imp.statistics_])),
        "feature_table": feature_table(),
        "scaler_ranges": pd.DataFrame({"feature": FEATURE_NAMES, "min (after log)": knn["prep"]["scaler"].data_min_,
                                       "max (after log)": knn["prep"]["scaler"].data_max_}),
        "row_example": train_df.iloc[[0]],
        "row_features": build_features(train_df.iloc[[0]]).T.rename(columns={0: "value"}),
        "row_scaled": knn_model.transform(train_df.iloc[[0]], knn["prep"]).T.rename(columns={0: "value"}),
    }
    res = {
        "profile": profile_data(train_df, test_df), "column_info": COLUMN_INFO,
        "skew": skew, "prep_info": prep_info, "log_cols": knn["prep"]["log_cols"],
        "X_train_raw": X_tr_tree[NUMERIC_COLS].sample(8000, random_state=1),
        "X_train_logged": knn["X_train"][NUMERIC_COLS].sample(8000, random_state=1),
        "X_train_raw_all": X_tr_tree[NUMERIC_COLS].sample(8000, random_state=2),
        "sample_rows": train_df.head(8),
        "knn": {k: v for k, v in knn.items() if k not in ("model", "prep", "X_test", "X_train", "prob_train", "prob_test")},
        "tree": {k: v for k, v in tree.items() if k not in ("model", "prep", "X_test", "X_train", "prob_train", "prob_test")},
        "metrics": mt, "winner": winner, "winner_scores": scores, "fit_gap": fit_gap,
        "breakdowns": breakdowns(test_df, probs),
        
        "thresholds": {n: threshold_table(y, probs[n]) for n in ["KNN", "Decision Tree"]},
        "curves": {n: (evaluate(y, probs[n])["fpr_curve"], evaluate(y, probs[n])["tpr_curve"]) for n in ["KNN", "Decision Tree"]},
        "calibration": {n: calibration(y, probs[n]) for n in ["KNN", "Decision Tree"]},
        "y_test": y.values, "prob": {n: probs[n] for n in ["KNN", "Decision Tree"]},
        "k_range": knn_model.K_VALUES, "tree_depths": decision_tree_model.DEPTHS,
    }
    # examples
    ex = pick_examples(test_df, knn["prob_test"], tree["prob_test"])
    res["examples"] = []
    for label, i in ex.items():
        row = test_df.iloc[[i]]
        res["examples"].append({
            "label": label, "row": row[["category", "sub_category", "days_left", "units_left", "avg_daily_demand"]].iloc[0].to_dict(),
            "truth": int(row[TARGET].iloc[0]),
            "knn_prob": float(knn["prob_test"][i]), "tree_prob": float(tree["prob_test"][i]),
            "neighbors": knn_model.explain_neighbors(knn["model"], knn["X_test"].iloc[[i]], train_df),
            "tree_path": decision_tree_model.explain_path(tree["model"], tree["X_test"].iloc[[i]])})
    if save:
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump({"model": knn["model"], "prep": knn["prep"]}, MODEL_DIR / "knn.joblib")
        joblib.dump({"model": tree["model"], "prep": tree["prep"]}, MODEL_DIR / "decision_tree.joblib")
        joblib.dump(res, MODEL_DIR / "train_results.joblib")
    return res


def load_results():
    f = MODEL_DIR / "train_results.joblib"
    return joblib.load(f) if f.exists() else None


if __name__ == "__main__":
    r = train_all()
    print(r["metrics"][["accuracy", "precision", "recall", "f1", "roc_auc"]].round(3))
    print("Winner:", r["winner"], r["winner_scores"])
