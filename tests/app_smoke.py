"""
Smoke test for app.py WITHOUT a browser.
Streamlit and Plotly are replaced by stand-ins, so this checks that every tab's data code runs on real data
(it does NOT check how the page looks - for that, run:  streamlit run app.py).
Run:  python tests/app_smoke.py
"""
import runpy, sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import config as C
from src import store


class Rerun(Exception):
    pass


def run_app(press=()):
    """Execute app.py once. `press` = button-label fragments that should return True."""
    st = MagicMock()
    st.session_state = {}
    st.rerun.side_effect = Rerun
    st.slider.side_effect = lambda *a, **k: a[3]
    st.selectbox.side_effect = lambda label, options, index=0, **k: list(options)[index]
    st.radio.side_effect = lambda label, options, **k: list(options)[0]
    st.checkbox.return_value = False
    st.file_uploader.return_value = []
    st.button.side_effect = lambda label, **k: any(p in label for p in press) and not k.get("disabled", False)
    def make_col():
        c = MagicMock()                     # widgets called on a column behave like the normal ones
        c.button, c.slider, c.selectbox = st.button, st.slider, st.selectbox
        return c
    st.columns.side_effect = lambda spec: [make_col() for _ in range(spec if isinstance(spec, int) else len(spec))]
    st.tabs.side_effect = lambda names: [MagicMock() for _ in names]
    st.data_editor.side_effect = lambda df, **k: df.copy()
    sys.modules.update({"streamlit": st, "plotly": MagicMock(), "plotly.express": MagicMock(), "plotly.graph_objects": MagicMock()})
    try:
        runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
    except Rerun:
        pass
    return st


def load_days(n):
    store.reset_all()
    for d in range(1, n + 1):
        f = sorted(C.DAILY_DIR.glob(f"Day{d:02d}_*.csv"))[0]
        from src import scoring
        clean, err, _ = scoring.validate_and_clean(pd.read_csv(f))
        assert not err, err
        store.save_upload(clean)


if __name__ == "__main__":
    store.reset_all(); run_app();                         print("empty state          : OK")
    load_days(1); run_app();                               print("Day 1 only           : OK")
    load_days(4); run_app(press=["Save my decisions"])
    assert len(store.load_decisions()) > 0;                print("save decisions       : OK", len(store.load_decisions()), "rows")
    run_app();                                             print("Day 4 + decisions    : OK")
    store.reset_all(); run_app(press=["Load next day"]);   assert store.uploaded_days() == [1]; print("load next day        : OK")
    store.reset_all(); run_app(press=["auto-approve"])
    assert len(store.uploaded_days()) == 14 and len(store.load_decisions()) > 100; print("load all + auto      : OK", len(store.load_decisions()), "decisions")
    run_app();                                             print("all 14 days          : OK")
    out = sorted(p.name for p in C.POWERBI_DIR.glob("*.csv")); print("powerbi_data files   :", out)
    store.reset_all()
    print("ALL SMOKE TESTS PASSED")
