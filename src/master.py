"""
master.py - THE MASTER FILE (uploaded once, on the Setup tab)
==============================================================
The master file is the fortnight's batch list: every batch the store receives in the 14 days,
with its delivery date, expiry date (= batch life), quantity received and prices.

What it is used for:
  1. A check on every daily file: batches that are not in the master are reported.
  2. Batch life and cost per product in the 15-day order plan (src/order_plan.py).

The daily files (Day01 ... Day14) still carry the evening-by-evening stock. The master file does not
replace them; it is the "total list" they must agree with.
"""
import pandas as pd

import config as C
from src import features as F

MASTER_COLS = ["batch_id", "product_name", "category", "delivered_date", "expiry_date",
               "qty_received", "cost_price", "base_price"]
NUMERIC = ["qty_received", "cost_price", "base_price"]
DATES = ["delivered_date", "expiry_date"]


# ============================================================
# STEP 1: CHECK AND CLEAN THE UPLOADED MASTER FILE
# ------------------------------------------------------------
# Same idea as scoring.validate_and_clean for the daily files:
#   errors   -> the file is refused
#   warnings -> the file is accepted, the note is shown
# ============================================================
def validate_and_clean_master(df: pd.DataFrame):
    errors, warnings = [], []
    missing = [c for c in MASTER_COLS if c not in df.columns]
    if missing:
        return None, [f"Missing columns: {', '.join(missing)}"], warnings

    d = df[MASTER_COLS + (["store_id"] if "store_id" in df.columns else [])].copy()
    for c in ["batch_id", "product_name", "category"]:
        d[c] = d[c].astype(str).str.strip()
    for c in NUMERIC:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in DATES:
        d[c] = pd.to_datetime(d[c], errors="coerce")

    bad = d[NUMERIC + DATES].isna().any(axis=1)
    if bad.any():
        warnings.append(f"{int(bad.sum())} row(s) had empty or invalid values and were removed.")
        d = d[~bad]
    if d.empty:
        return None, ["The file has no usable rows."], warnings

    if "store_id" in d.columns and (d["store_id"] != C.STORE_ID).any():
        errors.append(f"This app is for {C.STORE_ID} only; the master file has other store IDs.")
    if (d[NUMERIC] < 0).any().any():
        errors.append("Negative quantities or prices found - please check the file.")
    if (d["expiry_date"] < d["delivered_date"]).any():
        errors.append("Some batches expire before they are delivered.")

    dup = d.duplicated("batch_id")
    if dup.any():
        warnings.append(f"{int(dup.sum())} duplicate batch row(s) removed (kept the first).")
        d = d[~dup]
    unknown = sorted(set(d["category"]) - set(F.CATEGORIES))
    if unknown:
        warnings.append(f"Unknown categories {unknown}: scored as 'none of the known categories'.")

    return d[MASTER_COLS].reset_index(drop=True), errors, warnings

# ------------------------------------------------------------
# WHAT WE DID IN STEP 1:
# - Checked that all 8 columns exist, numbers and dates are valid, nothing is negative,
#   no batch expires before it is delivered and (if present) the store is STORE_015.
# - Removed duplicate batch IDs with a warning instead of stopping.
# ------------------------------------------------------------


# ============================================================
# STEP 2: THE APP'S MEMORY FOR THE MASTER FILE
# ------------------------------------------------------------
# The cleaned master file is saved in state/master.csv, so a browser refresh loses nothing.
# ============================================================
def master_exists() -> bool:
    return C.MASTER_FILE.exists()


def save_master(clean: pd.DataFrame):
    C.STATE_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(C.MASTER_FILE, index=False, date_format="%Y-%m-%d")


def load_master() -> pd.DataFrame:
    if not master_exists():
        return pd.DataFrame(columns=MASTER_COLS)
    return pd.read_csv(C.MASTER_FILE, parse_dates=DATES)


def remove_master():
    if master_exists():
        C.MASTER_FILE.unlink()

# ------------------------------------------------------------
# WHAT WE DID IN STEP 2:
# - Four small functions: does it exist, save it, load it, remove it.
# ------------------------------------------------------------


# ============================================================
# STEP 3: BATCH LIFE AND A SHORT SUMMARY
# ------------------------------------------------------------
# batch life (days) = expiry date - delivered date + 1  (the same rule make_daily_files.py uses)
# ============================================================
def with_batch_life(master: pd.DataFrame) -> pd.DataFrame:
    m = master.copy()
    m["batch_life_days"] = (m["expiry_date"] - m["delivered_date"]).dt.days + 1
    return m


def summary(master: pd.DataFrame) -> dict:
    m = with_batch_life(master)
    return {"batches": len(m), "products": int(m["product_name"].nunique()), "categories": int(m["category"].nunique()),
            "units": float(m["qty_received"].sum()), "stock_value": float((m["qty_received"] * m["cost_price"]).sum()),
            "avg_life": float(m["batch_life_days"].mean()),
            "deliveries": m.groupby("delivered_date").size().rename("batches").reset_index()}

# ------------------------------------------------------------
# WHAT WE DID IN STEP 3:
# - Added the batch life (in days) and a summary the Setup tab shows after the upload.
# ------------------------------------------------------------


# ============================================================
# STEP 4: COMPARE A DAILY FILE WITH THE MASTER FILE
# ------------------------------------------------------------
# These are warnings only. The upload still goes through.
# ============================================================
def check_day_against_master(day: pd.DataFrame, master: pd.DataFrame):
    if master is None or master.empty:
        return []
    notes = []
    unknown = set(day["batch_id"]) - set(master["batch_id"])
    if unknown:
        notes.append(f"{len(unknown)} batch(es) in this file are not in the master file.")
    both = day.drop_duplicates("batch_id").merge(master[["batch_id", "qty_received", "expiry_date"]],
                                                  on="batch_id", suffixes=("", "_master"))
    wrong_qty = int((both["qty_received"] != both["qty_received_master"]).sum())
    wrong_exp = int((both["expiry_date"] != both["expiry_date_master"]).sum())
    if wrong_qty:
        notes.append(f"{wrong_qty} batch(es) show a different quantity received than the master file.")
    if wrong_exp:
        notes.append(f"{wrong_exp} batch(es) show a different expiry date than the master file.")
    return notes

# ------------------------------------------------------------
# WHAT WE DID IN STEP 4:
# - Reported batches missing from the master and batches whose quantity or expiry date disagrees.
# ------------------------------------------------------------


# ============================================================
# STEP 5: BUILD A MASTER FILE FROM THE DAILY FILES (FOR THE DEMO)
# ------------------------------------------------------------
# A real store would export the master file from its ordering system. In the demo we derive it
# from the 14 daily files: one row per batch.
# ============================================================
def master_from_days(frames) -> pd.DataFrame:
    all_days = pd.concat(list(frames), ignore_index=True)
    first = all_days.sort_values(["batch_id", "day_no"]).drop_duplicates("batch_id")
    out = first[MASTER_COLS].sort_values(["delivered_date", "batch_id"]).reset_index(drop=True)
    return out

# ------------------------------------------------------------
# WHAT WE DID IN STEP 5:
# - Took the first row of every batch from the daily files. make_daily_files.py uses this to write
#   DailyFiles-Retail/Master_Fortnight_Batches.csv, the sample you can upload on the Setup tab.
# ------------------------------------------------------------
