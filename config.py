"""
config.py - ALL settings live here. Change a value here, not inside the code.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_CSV = ROOT / "data" / "raw" / "perishable_goods_management.csv"   # Kaggle history file (training only)
MODELS_DIR = ROOT / "models"            # saved models + metrics written by train.py
POWERBI_DIR = ROOT / "powerbi_data"     # clean CSVs that Power BI reads
STATE_DIR = ROOT / "state"              # the app's own memory (uploaded days) so a refresh loses nothing
DAILY_DIR = ROOT / "DailyFiles-Retail"  # the 14 daily files (copy kept inside the project)
WINDOWS_DAILY_DIR = Path(r"C:\Users\18ani\Downloads\DailyFiles-Retail")  # also written when run on Windows

STORE_ID = "STORE_015"            # the ONE store this app serves
WASTE_AT_RISK_PCT = 20.0          # label: a batch is "at risk" if >= this % of it ends up wasted
SPLIT_DATE = "2024-01-01"         # train on dates before this (all stores), test on dates from this
DEMO_START = "2024-09-01"         # a Sunday = Day 1 of the demo
N_DAYS = 14
DELIVERY_SUNDAYS = (1, 8)         # day numbers of the Sunday deliveries
DELIVERY_WEDNESDAYS = (4, 11)     # day numbers of the Wednesday top-ups
DROP_CATEGORIES = ["Pharmaceuticals", "Frozen_Meals"]            # shelf life of months: different problem
SHORT_LIFE_CATEGORIES = ["Bakery", "Ready_to_Eat", "Seafood", "Meat"]   # these also get a Wednesday delivery
LIFE_MIN, LIFE_MAX = 2, 10        # demo shelf life is compressed into this many days
SEED = 42                         # fixed seed = same results every run

# Rule layer (also editable live in the app sidebar)
RULES = {"low": 0.40, "high": 0.70, "md_mid": 20, "md_high": 40}
# ASSUMPTION (not measured): a markdown of m% raises the remaining daily sales by UPLIFT * m%
UPLIFT_PER_MARKDOWN = 0.75        # e.g. 20% off -> 15% more sales
FALSE_ALARM_MARKDOWN_PCT = 20     # cost view: a false alarm costs this % of the price on the stock marked down

# ---- Models (the two the app offers; code lives in src/models/) ----
KNN_MAX_TRAIN_ROWS = 60000        # KNN compares every new batch with its stored cases, so we keep at most this many

# ---- Master file (Setup tab): the list of every batch delivered in the fortnight ----
MASTER_FILE = STATE_DIR / "master.csv"                        # the cleaned copy the app remembers
MASTER_SAMPLE = DAILY_DIR / "Master_Fortnight_Batches.csv"    # sample master file made by make_daily_files.py

# ---- Order plan for the next days (Manager dashboard) ----
PLAN_DAYS = 15                    # how many days ahead the manager orders for
PLAN_SAFETY_BUFFER = 0.05         # 5% extra so the shelf does not run empty
PLAN_FILL_LOW_RISK = 0.90         # one delivery = 90% of what can be sold before expiry (safe products)
PLAN_FILL_HIGH_RISK = 0.70        # one delivery = 70% of it for products that waste a lot (smaller, more often)
PLAN_HIGH_RISK_SHARE = 0.50       # a product is "high risk" if the model flags >= this share of its shelf-evenings
PLAN_HIGH_WASTE_PCT = 20.0        # ... or if it really wasted this % of what it received
