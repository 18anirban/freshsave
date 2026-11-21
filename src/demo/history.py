# ====================================================================
# DEMO - HISTORY  (each saved day is a CSV in demo_runs/, so a browser
# refresh does not lose the week)
# ====================================================================
from pathlib import Path
import pandas as pd
from src.data_prep import ROOT

HIST_DIR = ROOT / "demo_runs"
LAST_DAY = 7


def save_day(day, df, model_name):
    HIST_DIR.mkdir(exist_ok=True)
    df.to_csv(HIST_DIR / f"day_{day}.csv", index=False)
    (HIST_DIR / "model.txt").write_text(model_name)


def load_all():
    """{day: DataFrame} for the days saved so far, in order."""
    out = {}
    for d in range(LAST_DAY + 1):
        f = HIST_DIR / f"day_{d}.csv"
        if not f.exists():
            break
        out[d] = pd.read_csv(f)
    return out


def saved_model():
    f = HIST_DIR / "model.txt"
    return f.read_text().strip() if f.exists() else None


def reset():
    if HIST_DIR.exists():
        for f in HIST_DIR.glob("*"):
            f.unlink()
