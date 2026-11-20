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
from src import features as F, simulate as S, scoring, rules, store

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
    out = store.refresh("ANN")                                      # switching model re-scores everything
    assert out["scores"].model_used.eq("ANN").all()
    assert len(store.load_decisions()) > 0
    for name in ["batch_scores", "outcomes", "decisions_log", "daily_summary", "calendar"]:
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


if __name__ == "__main__":
    n = 0
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); n += 1; print("PASS", k)
    print(f"{n} tests passed")
