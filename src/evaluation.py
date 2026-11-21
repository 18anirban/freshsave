# ====================================================================
# SHARED EVALUATION  - one scoring function so every model is judged
# in exactly the same way.
# ====================================================================
import numpy as np
import pandas as pd
from sklearn import metrics as M
from sklearn.calibration import calibration_curve


def evaluate(y, prob, threshold=0.5):
    """All the scores for one model. prob = predicted chance of at_risk."""
    y = np.asarray(y); prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = M.confusion_matrix(y, pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = M.roc_curve(y, prob) if len(set(prob)) > 1 else ([0, 1], [0, 1], None)
    try:
        auc = M.roc_auc_score(y, prob)
        ll = M.log_loss(y, np.clip(prob, 1e-6, 1 - 1e-6))
    except ValueError:
        auc, ll = float("nan"), float("nan")
    return {
        "accuracy": M.accuracy_score(y, pred),
        "precision": M.precision_score(y, pred, zero_division=0),
        "recall": M.recall_score(y, pred, zero_division=0),
        "f1": M.f1_score(y, pred, zero_division=0),
        "roc_auc": auc,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "balanced_accuracy": M.balanced_accuracy_score(y, pred),
        "mcc": M.matthews_corrcoef(y, pred),
        "log_loss": ll,
        "brier": M.brier_score_loss(y, prob),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "fpr_curve": list(fpr), "tpr_curve": list(tpr),
    }


def calibration(y, prob, bins=10):
    frac, mean = calibration_curve(y, prob, n_bins=bins, strategy="quantile")
    return mean, frac


def threshold_table(y, prob, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)):
    rows = []
    for t in thresholds:
        r = evaluate(y, prob, t)
        rows.append({"threshold": t, "accuracy": r["accuracy"], "precision": r["precision"],
                     "recall": r["recall"], "f1": r["f1"], "missed (FN)": r["fn"], "false alarms (FP)": r["fp"]})
    return pd.DataFrame(rows)


def group_metrics(df, prob, by, threshold=0.5):
    """Accuracy / recall per group (evening, category, store)."""
    d = df[[by, "at_risk"]].copy()
    d["pred"] = (np.asarray(prob) >= threshold).astype(int)
    rows = []
    for k, g in d.groupby(by):
        rows.append({by: k, "rows": len(g),
                     "accuracy": M.accuracy_score(g["at_risk"], g["pred"]),
                     "recall": M.recall_score(g["at_risk"], g["pred"], zero_division=0)})
    return pd.DataFrame(rows)
