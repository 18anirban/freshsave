"""
STEP 1 of 3 - TRAIN THE MODELS   (run once:  python src/train.py)
===================================================================
In plain words:
  1. Read the Kaggle history file (all 50 stores).
  2. Keep fresh food only; label = "at risk" if 20%+ of the batch ended up wasted.
  3. Simulate each batch's evening-by-evening stock (simulate.py).
  4. Train TWO models on 2023: Decision Tree and KNN (the code of each is in src/models/).
  5. Test both on 2024 (the "future") and save every metric the app shows.
  6. Pick the winner (best recall + F1) as the default model.
The app never reads the Kaggle file - it only reads what this script saves in models/.

Adding a model later: write one more file in src/models/ and add it to REGISTRY. This script
loops over REGISTRY, so nothing else here needs to change.
"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             balanced_accuracy_score, matthews_corrcoef, log_loss, brier_score_loss,
                             confusion_matrix, roc_curve, precision_recall_curve)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src import features as F
from src.models import REGISTRY
from src.simulate import build_long


def evening_snapshots(batches, seed):
    """One row per batch per evening that still has days left. The final expiry evening is excluded:
    that evening the batch is simply discarded, there is nothing left to predict."""
    long = build_long(batches, seed)
    return long[long["days_left"] > 0].reset_index(drop=True)


def all_metrics(y, p, thr=0.5):
    """Every number in the Tab 4 metrics table."""
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat), "f1": f1_score(y, yhat), "roc_auc": roc_auc_score(y, p),
            "specificity": tn / (tn + fp), "balanced_accuracy": balanced_accuracy_score(y, yhat),
            "mcc": matthews_corrcoef(y, yhat), "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
            "brier": brier_score_loss(y, p), "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def thin(n, keep=150):
    return np.unique(np.linspace(0, n - 1, min(keep, n)).astype(int))


def main():
    t0 = time.time()
    C.MODELS_DIR.mkdir(exist_ok=True); C.POWERBI_DIR.mkdir(exist_ok=True)

    print("1/6 Loading Kaggle file ...")
    raw = F.load_raw()
    batches = F.prepare_batches(raw)
    # Strict time-based split. The demo store's 2024 batches fall in the TEST period, so they are never trained on.
    train_b = batches[batches["transaction_date"] < C.SPLIT_DATE]
    test_b = batches[batches["transaction_date"] >= C.SPLIT_DATE]
    print(f"    raw rows {len(raw):,} | fresh batches {len(batches):,} | train {len(train_b):,} | test {len(test_b):,}")

    print("2/6 Simulating evening snapshots ...")
    tr = evening_snapshots(train_b, C.SEED)
    te = evening_snapshots(test_b, C.SEED + 1)
    Xtr, names = F.build_X(tr); ytr = tr["at_risk"].to_numpy()
    Xte, _ = F.build_X(te);     yte = te["at_risk"].to_numpy()
    print(f"    snapshot rows: train {len(tr):,} | test {len(te):,} | features {len(names)}")

    # data profile for Tab 4
    F.skew_table(tr).to_csv(C.MODELS_DIR / "skewness.csv", index=False)
    F.add_engineered(tr)[F.LOG_COLS].sample(3000, random_state=C.SEED).to_csv(C.MODELS_DIR / "profile_sample.csv", index=False)
    json.dump({"raw_rows": int(len(raw)), "raw_columns": int(raw.shape[1]), "raw_missing_cells": int(raw.isna().sum().sum()),
               "fresh_batches": int(len(batches)), "train_batches": int(len(train_b)), "test_batches": int(len(test_b)),
               "train_snapshots": int(len(tr)), "test_snapshots": int(len(te)),
               "at_risk_rate_train_batches": float(train_b["at_risk"].mean()),
               "at_risk_rate_test_batches": float(test_b["at_risk"].mean()),
               "at_risk_rate_train_snapshots": float(ytr.mean()), "at_risk_rate_test_snapshots": float(yte.mean()),
               "features": names, "excluded": F.EXCLUDED, "waste_threshold_pct": C.WASTE_AT_RISK_PCT,
               "split_date": C.SPLIT_DATE, "demo_store": C.STORE_ID,
               "category_counts": batches["category"].value_counts().to_dict()},
              open(C.MODELS_DIR / "data_profile.json", "w"), indent=2)

    print("3/6 Tuning + training both models (a few minutes) ...")
    # Tuning runs on a SAMPLE of batches (fast). Folds are grouped by batch so one batch never sits in
    # both the training fold and the validation fold (that would make scores look too good).
    rng = np.random.default_rng(C.SEED)
    sub = tr["batch_id"].isin(rng.choice(tr["batch_id"].unique(), size=6000, replace=False)).to_numpy()
    cv, groups = GroupKFold(n_splits=3), tr.loc[sub, "batch_id"]
    # Each model file in src/models/ says which settings to try (PARAM_GRID) and how to build the final model.
    best, models, train_time = {}, {}, {}
    for name, m in REGISTRY.items():
        search = GridSearchCV(m.base_estimator(), m.PARAM_GRID, cv=cv, scoring="f1")
        search.fit(Xtr[sub], ytr[sub], groups=groups)
        best[name] = search.best_params_
    print("    best settings:", " | ".join(f"{n}: {p}" for n, p in best.items()))

    for name, m in REGISTRY.items():
        X_fit, y_fit = m.training_rows(Xtr, ytr, rng)        # the tree uses every row, KNN a sample (see config.py)
        est = m.build(best[name])
        t = time.time(); est.fit(X_fit, y_fit); train_time[name] = time.time() - t
        models[name] = est
        joblib.dump(est, C.MODELS_DIR / m.FILE)
    json.dump(names, open(C.MODELS_DIR / "feature_names.json", "w"))

    print("4/6 Evaluating on 2024 ...")
    probs, pred_ms = {}, {}
    for name, m in models.items():
        t = time.time(); probs[name] = m.predict_proba(Xte)[:, 1]
        pred_ms[name] = (time.time() - t) / len(Xte) * 1e6                      # ms per 1000 rows
    rows, by_day, curves = [], [], {}
    store_mask = (te["store_id"] == C.STORE_ID).to_numpy()
    for view, mask in [("All stores 2024", np.ones(len(te), bool)), (f"{C.STORE_ID} only (2024)", store_mask)]:
        for name in models:
            r = all_metrics(yte[mask], probs[name][mask])
            r.update(model=name, view=view, rows=int(mask.sum()), train_seconds=train_time[name],
                     predict_ms_per_1000_rows=pred_ms[name]); rows.append(r)
    for name in models:                                                          # accuracy per evening number
        for k in sorted(te["selling_days_elapsed"].unique()):
            m = (te["selling_days_elapsed"] == k).to_numpy()
            if m.sum() >= 200 and len(np.unique(yte[m])) == 2:
                r = all_metrics(yte[m], probs[name][m]); r.update(model=name, evening=int(k), rows=int(m.sum())); by_day.append(r)
    for name in models:
        fpr, tpr, _ = roc_curve(yte, probs[name]); i = thin(len(fpr))
        pr, rc, _ = precision_recall_curve(yte, probs[name]); j = thin(len(pr))
        cy, cx = calibration_curve(yte, probs[name], n_bins=10)
        curves[name] = {"roc_fpr": fpr[i].tolist(), "roc_tpr": tpr[i].tolist(), "pr_precision": pr[j].tolist(),
                        "pr_recall": rc[j].tolist(), "calib_pred": cx.tolist(), "calib_true": cy.tolist()}
    metrics = pd.DataFrame(rows)
    metrics.to_csv(C.MODELS_DIR / "metrics_summary.csv", index=False)
    pd.DataFrame(by_day).to_csv(C.MODELS_DIR / "metrics_by_evening.csv", index=False)
    json.dump(curves, open(C.MODELS_DIR / "curves.json", "w"))
    test_out = pd.DataFrame({"store_id": te["store_id"], "evening": te["selling_days_elapsed"], "y": yte})
    for name, m in REGISTRY.items():
        test_out[m.PROB_COLUMN] = probs[name]                  # p_dt, p_knn
    test_out = test_out.assign(qty_left=te["qty_left"], cost_price=te["cost_price"], base_price=te["base_price"],
                               final_waste_cost=te["final_waste_cost"])
    test_out.to_csv(C.MODELS_DIR / "test_predictions.csv.gz", index=False, compression="gzip")  # for threshold slider + cost view

    print("5/6 Feature importance ...")
    s = rng.choice(len(Xte), size=3000, replace=False)
    imp = pd.DataFrame({"feature": names})
    for name, m in REGISTRY.items():
        imp[m.IMPORTANCE_COLUMN] = m.importance(models[name], names, Xte.iloc[s], yte[s]).to_numpy()
    imp.to_csv(C.MODELS_DIR / "feature_importance.csv", index=False)

    allv = metrics[metrics["view"] == "All stores 2024"].set_index("model")
    winner = ((allv["recall"] + allv["f1"]) / 2).idxmax()   # recall matters most: a missed batch becomes waste
    why = (f"{winner} has the best average of recall ({allv.loc[winner,'recall']:.1%}) and F1 "
           f"({allv.loc[winner,'f1']:.1%}) on the 2024 test data. Recall matters most because a missed "
           f"at-risk batch turns into waste.")
    json.dump({"active": winner, "winner": winner, "why": why}, open(C.MODELS_DIR / "active_model.json", "w"))
    metrics.to_csv(C.POWERBI_DIR / "model_comparison.csv", index=False)
    print("\n6/6 RESULT (all stores, 2024 test)\n",
          metrics[metrics["view"] == "All stores 2024"][["model", "accuracy", "precision", "recall", "f1", "roc_auc", "train_seconds"]].round(3).to_string(index=False))
    print("\nWinner:", winner, "-", why, f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
