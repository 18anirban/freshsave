"""
simulate.py - the day-by-day sales SIMULATOR  (an ASSUMPTION, labelled as such everywhere)

Why we need it: the Kaggle file only gives each batch's START and its FINAL result. A real store would
report stock every evening, so we fill in the evenings between.

How: the batch's real final sales S are spread over its L selling days with random weights (seeded).
The running total ends EXACTLY on S, so the stock left on the last evening equals the real waste.
"""
import numpy as np
import pandas as pd


def sim_path(S: int, L: int, rng) -> np.ndarray:
    """Cumulative units sold at the end of selling day 1..L. Never decreases, last value == S."""
    w = rng.dirichlet(np.full(L, 6.0))           # random daily share of the sales
    cum = np.round(np.cumsum(w) * S).astype(int)
    cum[-1] = S                                   # force the real final number
    return np.maximum.accumulate(cum)


def build_long(b: pd.DataFrame, seed: int, life_col: str = "life") -> pd.DataFrame:
    """Expand batches into one row per batch per selling day (k = 1..L)."""
    rng = np.random.default_rng(seed)
    L = b[life_col].to_numpy(int)
    S = b["units_sold_final"].to_numpy(int)
    cums = [sim_path(s, l, rng) for s, l in zip(S, L)]
    rep = np.repeat(np.arange(len(b)), L)
    k = np.concatenate([np.arange(1, l + 1) for l in L])
    cum = np.concatenate(cums)
    prev = np.concatenate([np.concatenate([[0], c[:-1]]) for c in cums])
    out = b.iloc[rep].reset_index(drop=True)
    out["selling_days_elapsed"] = k
    out["units_sold_total"] = cum
    out["units_sold_today"] = cum - prev
    out["qty_left"] = out["qty_received"] - cum
    out["days_left"] = out[life_col].to_numpy() - k   # life is already expanded per row
    return out
