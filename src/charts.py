# ====================================================================
# CHARTS for the app. Each function returns a matplotlib figure.
# Colours: KNN = blue, Decision Tree = orange (same everywhere).
# ====================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GREY, RED, GREEN = "#2a6fdb", "#e8833a", "#9aa0a6", "#d64545", "#2e9e6b"
COL = {"KNN": BLUE, "Decision Tree": ORANGE}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25})


def _fig(w=6, h=3.4):
    return plt.subplots(figsize=(w, h))


def class_balance(profile):
    fig, ax = _fig(5, 2.6)
    names = ["Train", "Test"]
    risk = [profile["risk_rate_train"] * 100, profile["risk_rate_test"] * 100]
    ax.barh(names, [100, 100], color="#dfe7f3")
    ax.barh(names, risk, color=RED)
    for i, r in enumerate(risk):
        ax.text(r / 2, i, f"At risk {r:.1f}%", ha="center", va="center", color="white", fontweight="bold")
        ax.text(r + (100 - r) / 2, i, f"Safe {100 - r:.1f}%", ha="center", va="center", color="#333")
    ax.set_xlim(0, 100); ax.set_xlabel("% of rows"); ax.grid(False)
    fig.tight_layout(); return fig


def split_timeline(profile):
    fig, ax = _fig(7, 1.9)
    import pandas as pd
    a0, a1 = pd.to_datetime(profile["train_dates"]); b0, b1 = pd.to_datetime(profile["test_dates"])
    ax.barh(["Train"], [(a1 - a0).days], left=[a0.toordinal()], color=BLUE)
    ax.barh(["Test"], [(b1 - b0).days], left=[b0.toordinal()], color=ORANGE)
    ticks = pd.date_range(a0, b1, freq="6MS")
    ax.set_xticks([t.toordinal() for t in ticks]); ax.set_xticklabels([t.strftime("%b %y") for t in ticks])
    ax.grid(False); fig.tight_layout(); return fig


def skew_before_after(res, col):
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
    for ax, key, title, c in [(axes[0], "X_train_raw", "Before (raw)", GREY), (axes[1], "X_train_logged", "After log1p (KNN)", BLUE)]:
        v = res[key][col]
        ax.hist(v, bins=35, color=c); ax.set_title(f"{title}  skew={v.skew():.2f}", fontsize=9)
    fig.suptitle(col, fontsize=10); fig.tight_layout(); return fig


def knn_k_chart(grid):
    fig, ax = _fig(6, 3.2)
    for w, ls in [("uniform", "--"), ("distance", "-")]:
        g = grid[grid["weights"] == w]
        ax.errorbar(g["k"], g["cv_f1"], yerr=g["cv_std"], marker="o", ls=ls, color=BLUE, label=w, capsize=3)
    best = grid.sort_values("cv_f1").iloc[-1]
    ax.scatter([best["k"]], [best["cv_f1"]], s=160, facecolors="none", edgecolors=RED, lw=2, zorder=5)
    ax.set_xlabel("k (number of neighbours)"); ax.set_ylabel("cross-validated F1"); ax.legend(title="vote weighting")
    fig.tight_layout(); return fig


def tree_depth_chart(grid, leaf):
    fig, ax = _fig(6, 3.2)
    g = grid[grid["min_samples_leaf"] == leaf]
    ax.plot(g["max_depth"], g["train_f1"], marker="o", color=GREY, label="train F1 (memorising)")
    ax.plot(g["max_depth"], g["cv_f1"], marker="o", color=ORANGE, label="validation F1 (real)")
    ax.set_xlabel("max depth"); ax.set_ylabel("F1"); ax.legend(); ax.set_title(f"min_samples_leaf = {leaf}", fontsize=9)
    fig.tight_layout(); return fig


def metrics_bars(mt):
    names = ["KNN", "Decision Tree"]; cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    fig, ax = _fig(7, 3.4); x = np.arange(len(cols)); w = .38
    for i, n in enumerate(names):
        v = [mt.loc[n, c] for c in cols]
        b = ax.bar(x + (i - .5) * w, v, w, color=COL[n], label=n)
        for r, val in zip(b, v):
            ax.text(r.get_x() + r.get_width() / 2, val + .008, f"{val:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
    ax.set_ylim(0.5, 1.1); ax.legend(ncol=2, loc="upper left"); fig.tight_layout(); return fig


def confusion(mt, name):
    fig, ax = _fig(3.3, 3)
    m = np.array([[mt.loc[name, "tn"], mt.loc[name, "fp"]], [mt.loc[name, "fn"], mt.loc[name, "tp"]]], dtype=int)
    ax.imshow(m, cmap="Blues"); ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i, j]:,}", ha="center", va="center", fontsize=12,
                    color="white" if m[i, j] > m.max() / 2 else "#222")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred safe", "Pred risk"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["Safe", "At risk"])
    ax.set_title(name, fontsize=10); fig.tight_layout(); return fig


def roc(curves, mt):
    fig, ax = _fig(4.2, 3.6)
    for n, (f, t) in curves.items():
        ax.plot(f, t, color=COL[n], label=f"{n} (AUC {mt.loc[n, 'roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=GREY); ax.set_xlabel("False alarm rate"); ax.set_ylabel("Recall")
    ax.legend(loc="lower right"); fig.tight_layout(); return fig


def group_bar(df, by, metric="accuracy"):
    models = ["KNN", "Decision Tree"]
    keys = list(dict.fromkeys(df[by])); x = np.arange(len(keys)); w = .38
    fig, ax = _fig(7, 3.2)
    for i, m in enumerate(models):
        g = df[df["model"] == m].set_index(by).loc[keys]
        ax.bar(x + (i - .5) * w, g[metric], w, color=COL[m], label=m)
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=25 if len(keys) > 5 else 0)
    ax.set_ylim(0.5, 1.1); ax.set_ylabel(metric); ax.legend(ncol=2, loc="upper left"); fig.tight_layout(); return fig


def importance_bar(imp, top=10):
    t = imp.head(top)[::-1]
    fig, ax = _fig(6, 3.4); ax.barh(t["feature"], t["importance"], color=ORANGE); ax.set_xlabel("importance")
    fig.tight_layout(); return fig


def threshold_chart(tables):
    fig, ax = _fig(6, 3.2)
    for n, t in tables.items():
        ax.plot(t["threshold"], t["recall"], marker="o", color=COL[n], label=f"{n} recall")
        ax.plot(t["threshold"], t["precision"], marker="s", ls="--", color=COL[n], label=f"{n} precision")
    ax.set_xlabel("alarm threshold"); ax.legend(fontsize=8, ncol=2); fig.tight_layout(); return fig


def skew_shapes():
    """Teaching picture: what skew looks like."""
    rng = np.random.RandomState(0)
    data = [("Symmetric  (skew ~ 0)", rng.normal(50, 10, 5000)), ("Right-skewed  (skew > 1)", rng.lognormal(3, .7, 5000)),
            ("Left-skewed  (skew < -1)", 100 - rng.lognormal(3, .7, 5000))]
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.3))
    for ax, (t, v) in zip(axes, data):
        ax.hist(v, bins=40, color=GREY); ax.set_title(t, fontsize=9); ax.set_yticks([]); ax.grid(False)
    fig.tight_layout(); return fig


def skew_overview(skew):
    """All numeric columns: skewness before vs what KNN sees after log."""
    d = skew.sort_values("skewness", key=abs, ascending=True)
    y = np.arange(len(d)); fig, ax = _fig(7, 0.42 * len(d) + 1.2)
    ax.barh(y + .2, d["skewness"], .38, color=GREY, label="raw (Decision Tree sees this)")
    ax.barh(y - .2, d["skew used by KNN"], .38, color=BLUE, label="after log1p where applied (KNN sees this)")
    for x in (-1, 1):
        ax.axvline(x, color=RED, ls="--", lw=1)
    ax.axvline(0, color="#444", lw=.8)
    ax.set_yticks(y); ax.set_yticklabels(d["column"]); ax.set_xlabel("skewness   (red lines = +/-1 limit)")
    ax.legend(loc="lower right", fontsize=8); fig.tight_layout(); return fig


def hist_grid(df, cols, title_skew=True, ncols=4):
    n = len(cols); rows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(rows, ncols, figsize=(9, 1.9 * rows))
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    for ax, c in zip(np.ravel(axes), cols):
        ax.hist(df[c], bins=30, color=BLUE); ax.set_yticks([]); ax.grid(False)
        ax.set_title(f"{c}\nskew {df[c].skew():.2f}", fontsize=8)
    fig.tight_layout(); return fig
