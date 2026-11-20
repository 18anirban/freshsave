"""
Automated checks. Run:  python -m pytest tests -q      (or:  python tests/test_pipeline.py)
Each test states, in its name, the promise it protects.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C
from src import features as F, simulate as S, scoring, rules, store, order_plan
from src import master as M
from src.models import REGISTRY, MODEL_FILES, knn_model

_batches = None
def batches():
    global _batches
    if _batches is None:
        _batches = F.prepare_batches(F.load_raw())
    return _batches


def test_only_fresh_categories_are_kept():
    assert set(batches().category) == set(F.CATEGORIES)
    assert not set(C.DROP_CATEGORIES) & set(batches().category)


def test_label_rate_is_reasonable_and_matches_threshold():
    b = batches()
    assert 0.3 < b.at_risk.mean() < 0.55
    assert ((b.final_waste_pct >= C.WASTE_AT_RISK_PCT) == (b.at_risk == 1)).all()


def test_no_outcome_column_is_a_model_input():
    names = json.load(open(C.MODELS_DIR / "feature_names.json"))
    banned = set(F.OUTCOME_COLS) | {"spoilage_risk", "quality_grade", "markdown_applied", "discount_pct", "selling_price"}
    assert not banned & set(names), banned & set(names)


def test_time_split_demo_store_2024_never_in_training():
    b = batches()
    train = b[b.transaction_date < C.SPLIT_DATE]
    assert train.transaction_date.max() < pd.Timestamp(C.SPLIT_DATE)
    demo = b[(b.store_id == C.STORE_ID) & (b.transaction_date >= C.SPLIT_DATE)]
    assert len(demo) > 0 and not set(demo.batch_id) & set(train.batch_id)


def test_simulator_cumulative_sales_equal_real_final_sales_and_stock_never_negative():
    l = S.build_long(batches().head(3000), 1)
    g = l.groupby("batch_id")
    assert (g.units_sold_total.last() == g.units_sold_final.first()).all()
    assert (l.qty_left >= 0).all() and (l.units_sold_today >= 0).all()


def test_simulator_days_left_counts_down_by_one_to_zero():
    l = S.build_long(batches().head(3000), 1)
    g = l.groupby("batch_id")
    assert (g.days_left.last() == 0).all()
    assert (g.days_left.diff().dropna() == -1).all()          # this check caught a real bug during the build


def test_waste_on_last_evening_equals_real_waste():
    l = S.build_long(batches().head(2000), 1)
    last = l.groupby("batch_id").last()
    assert (last.qty_left == last.qty_received - last.units_sold_final).all()


def test_fourteen_daily_files_exist_with_correct_shape():
    files = sorted(C.DAILY_DIR.glob("Day*.csv"))
    assert len(files) == C.N_DAYS
    for f in files:
        clean, errors, _ = scoring.validate_and_clean(pd.read_csv(f))
        assert not errors, (f.name, errors)
        assert not set(F.OUTCOME_COLS) & set(clean.columns)      # a real store would not know these


def test_every_demo_batch_finishes_inside_the_window():
    parts = [pd.read_csv(f) for f in sorted(C.DAILY_DIR.glob("Day*.csv"))]
    all_ = pd.concat(parts)
    assert all_.groupby("batch_id").days_left.min().eq(0).all()
    assert all_.batch_id.nunique() == 848


def test_validation_blocks_bad_files():
    good = pd.read_csv(sorted(C.DAILY_DIR.glob("Day02*.csv"))[0])
    bad = good.copy(); bad["store_id"] = "STORE_001"
    assert scoring.validate_and_clean(bad)[1]                       # wrong store
    bad = good.copy(); bad.loc[0, "qty_left"] = -5
    assert scoring.validate_and_clean(bad)[1]                       # negative stock
    assert scoring.validate_and_clean(good.drop(columns=["qty_left"]))[1]   # missing column
    dup = pd.concat([good, good.head(3)])
    clean, errors, warnings = scoring.validate_and_clean(dup)
    assert not errors and len(clean) == len(good) and warnings      # duplicates removed with a warning


def test_rules_map_risk_to_actions():
    assert rules.suggest(0.2, 5)[1] == 0
    assert rules.suggest(0.5, 5) == (f"{C.RULES['md_mid']}% markdown", C.RULES["md_mid"], False)
    a, m, r = rules.suggest(0.9, 5); assert m == C.RULES["md_high"] and r is True
    assert rules.suggest(0.9, 0)[0] == "Discard tonight"


def test_upload_flow_day1_then_day2_scores_logs_and_exports():
    store.reset_all()
    for d in (1, 2):
        f = sorted(C.DAILY_DIR.glob(f"Day{d:02d}_*.csv"))[0]
        clean, errors, _ = scoring.validate_and_clean(pd.read_csv(f)); assert not errors
        store.save_upload(clean)
    out = store.refresh("Decision Tree")
    s = out["scores"]
    assert store.uploaded_days() == [1, 2]
    assert s[s.days_left > 0].risk.between(0, 1).all() and s[s.days_left == 0].risk.isna().all()
    store.save_decisions(store.auto_approve(s[s.day_no == 1]))
    out = store.refresh("KNN")                                      # switching model re-scores everything
    assert out["scores"].model_used.eq("KNN").all()
    assert len(store.load_decisions()) > 0
    for name in ["batch_scores", "outcomes", "decisions_log", "daily_summary", "calendar", "order_plan"]:
        assert (C.POWERBI_DIR / f"{name}.csv").exists()
    store.reset_all()


def test_reuploading_a_day_replaces_it_without_duplicates():
    store.reset_all()
    f = sorted(C.DAILY_DIR.glob("Day01*.csv"))[0]
    clean, _, _ = scoring.validate_and_clean(pd.read_csv(f))
    store.save_upload(clean); store.save_upload(clean)
    assert store.uploaded_days() == [1] and len(store.refresh()["scores"]) == len(clean)
    store.reset_all()


def test_markdown_effect_is_capped_by_stock_and_zero_without_markdown():
    assert rules.estimate_effect(100, 40, 0, 5, 2)[0] == 0
    extra, *_ = rules.estimate_effect(100, 90, 40, 5, 2, uplift=50)   # huge uplift cannot sell more than is left
    assert extra <= 90


# ---------------------------------------------------------------------------------------------
# New: the two model files, the master file and the 15-day order plan
# ---------------------------------------------------------------------------------------------
def _clean_days(n=14):
    out = []
    for d in range(1, n + 1):
        f = sorted(C.DAILY_DIR.glob(f"Day{d:02d}_*.csv"))[0]
        clean, errors, _ = scoring.validate_and_clean(pd.read_csv(f))
        assert not errors, errors
        out.append(clean)
    return out


def test_two_models_are_registered_each_with_its_own_file():
    assert list(REGISTRY) == ["Decision Tree", "KNN"]
    assert MODEL_FILES == {"Decision Tree": "decision_tree.joblib", "KNN": "knn.joblib"}
    for m in REGISTRY.values():
        for need in ("NAME", "FILE", "PROB_COLUMN", "IMPORTANCE_COLUMN", "IMPORTANCE_LABEL", "PARAM_GRID",
                     "base_estimator", "build", "training_rows", "importance"):
            assert hasattr(m, need), (m.NAME, need)


def test_master_sample_file_is_valid_and_agrees_with_the_daily_files():
    clean, errors, warnings = M.validate_and_clean_master(pd.read_csv(C.MASTER_SAMPLE))
    assert not errors and not warnings and len(clean) == 848
    days = _clean_days()
    assert set(clean.batch_id) == set(pd.concat(days).batch_id)
    for d in days:
        assert M.check_day_against_master(d, clean) == []          # no notes: the files agree


def test_master_validation_blocks_bad_files():
    good = pd.read_csv(C.MASTER_SAMPLE)
    assert M.validate_and_clean_master(good.drop(columns=["expiry_date"]))[1]          # missing column
    bad = good.copy(); bad.loc[0, "qty_received"] = -1
    assert M.validate_and_clean_master(bad)[1]                                         # negative quantity
    bad = good.copy(); bad.loc[0, "expiry_date"] = "2000-01-01"
    assert M.validate_and_clean_master(bad)[1]                                         # expires before delivery
    clean, errors, warnings = M.validate_and_clean_master(pd.concat([good, good.head(3)]))
    assert not errors and len(clean) == len(good) and warnings                          # duplicates removed with a warning


def test_master_is_saved_and_reloaded_and_can_be_removed():
    M.remove_master()
    assert not M.master_exists()
    clean, _, _ = M.validate_and_clean_master(pd.read_csv(C.MASTER_SAMPLE))
    M.save_master(clean)
    assert M.master_exists() and len(M.load_master()) == len(clean)
    M.remove_master()
    assert not M.master_exists()


def test_a_day_that_disagrees_with_the_master_is_reported():
    master, _, _ = M.validate_and_clean_master(pd.read_csv(C.MASTER_SAMPLE))
    day = _clean_days(2)[1].copy()
    day.loc[day.index[0], "batch_id"] = "B999999"                                       # a batch the master does not know
    day.loc[day.index[1], "qty_received"] += 5                                          # a quantity that differs
    assert len(M.check_day_against_master(day, master)) == 2


def test_order_plan_is_consistent_for_every_product():
    store.reset_all()
    for clean in _clean_days():
        store.save_upload(clean)
    master, _, _ = M.validate_and_clean_master(pd.read_csv(C.MASTER_SAMPLE)); M.save_master(master)
    plan = store.refresh("Decision Tree")["plan"]
    assert plan.product_name.is_unique and len(plan) > 20
    assert (plan.order_qty >= 0).all() and (plan.order_cost >= 0).all()
    assert (plan.units_per_delivery * plan.deliveries >= plan.order_qty).all()          # deliveries cover the order
    assert (plan.units_per_delivery <= plan.largest_safe_delivery).all()                # none is bigger than the safe size
    assert (C.POWERBI_DIR / "order_plan.csv").exists()
    M.remove_master(); store.reset_all()


def test_high_risk_products_get_smaller_deliveries():
    rows = []
    for product, risk in (("A", 0.9), ("B", 0.1)):                                      # same sales and life, different risk
        for day in (1, 2, 3):
            rows.append(dict(day_no=day, product_name=product, category="Dairy", batch_id=product + "1",
                             units_sold_today=10, qty_left=100 - 10 * day, days_left=5 - day, risk=risk,
                             delivered_date=pd.Timestamp("2024-09-01"), expiry_date=pd.Timestamp("2024-09-05"),
                             qty_received=100, cost_price=2.0, base_price=4.0))
    plan = order_plan.build_order_plan(pd.DataFrame(rows), pd.DataFrame(), None, C.RULES).set_index("product_name")
    assert plan.loc["A", "high_risk"] and not plan.loc["B", "high_risk"]
    assert plan.loc["A", "largest_safe_delivery"] < plan.loc["B", "largest_safe_delivery"]
    assert plan.loc["A", "deliveries"] >= plan.loc["B", "deliveries"]
    assert plan.loc["A", "order_qty"] == plan.loc["B", "order_qty"]                     # risk changes delivery size, not demand


def test_knn_neighbour_table_has_one_row_per_neighbour():
    names = json.load(open(C.MODELS_DIR / "feature_names.json"))
    model = scoring.load_model("KNN")
    day = _clean_days(2)[1]
    X, _ = F.build_X(day[day.days_left > 0].head(1))
    table = knn_model.explain_neighbours(model, X[names], names)
    assert len(table) == model.named_steps["knn"].n_neighbors
    assert set(table["What happened"]) <= {"Wasted 20%+", "Sold in time"}


if __name__ == "__main__":
    n = 0
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); n += 1; print("PASS", k)
    print(f"{n} tests passed")
