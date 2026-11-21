# ====================================================================
# DEMO PAGE (Streamlit). Follows the agreed flow:
# set-up -> Day 0 opening stock -> Day 1..7 sales upload -> file check ->
# prediction -> risk list -> record action -> save day -> final dashboard.
# ====================================================================
import pandas as pd
import streamlit as st

from src import charts as C
from src.model_deployment_train.model_comparison import train_all
from src.demo import dashboard as D, history as H, scoring as S

DEMO_DIR = H.ROOT / "data" / "demo"
ICON = {"High": "🔴 High", "Medium": "🟠 Medium", "Low": "🟢 Low"}
pct = lambda x: f"{x * 100:.1f}%"


DAY_NAMES = ["Saturday night", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


def day_file_name(day):
    return "day_0_opening_stock.csv" if day == 0 else f"day_{day}_sales.csv"


def day_tracker(done):
    cols = st.columns(8)
    for d, col in enumerate(cols):
        state = "Done" if d < done else "Next" if d == done else "Waiting"
        col.markdown(f"**Day {d}**  \n{DAY_NAMES[d]}  \n{state}")
    st.progress(done / 8)


def table(df, fmt=None):
    st.dataframe(df.style.format(fmt) if fmt else df, hide_index=True)


# ---------------------------------------------------------------- day step
def day_step(day, hist, model_name):
    title = ("Opening stock: inventory loaded on Saturday night, before any sales" if day == 0
             else f"{DAY_NAMES[day]} evening closing report (sales of the day and stock left)")
    st.subheader(f"Day {day} - upload: {title}")
    st.caption("CSV with the same columns as the training data (no outcome columns). Or use the ready-made demo file.")
    up = st.file_uploader("Upload CSV", type="csv", key=f"up_{day}")
    use_demo = st.checkbox(f"Use demo file ({day_file_name(day)})", key=f"demo_{day}")
    if up is not None:
        df = pd.read_csv(up)
    elif use_demo:
        df = pd.read_csv(DEMO_DIR / day_file_name(day))
    else:
        st.info("Waiting for today's file.")
        return
    prev = hist[day - 1]["date"].iloc[0] if day > 0 else None
    store = hist[0]["store_id"].iloc[0] if day > 0 else None
    errors, warnings = S.check_file(df, prev, store)
    for w in warnings:
        st.warning(w)
    if errors:
        for e in errors:
            st.error(e)
        st.stop()
    st.success(f"File OK: {len(df)} batches, store {df['store_id'].iloc[0]}, date {df['date'].iloc[0]}.")

    scored = S.score(df, model_name, S.load_model(model_name))
    n = scored["level"].value_counts()
    c = st.columns(4)
    c[0].metric("Batches checked", len(scored))
    for col, lv in zip(c[1:], ["High", "Medium", "Low"]):
        col.metric(f"{ICON[lv]} risk", int(n.get(lv, 0)))
    st.caption(f"Units on the shelf in High-risk batches: {int(scored.loc[scored['level'] == 'High', 'units_left'].sum()):,}")

    st.subheader("Today's risk list (highest risk first)")
    show = pd.DataFrame({"Risk": scored["level"].map(ICON), "Chance of waste (%)": (scored["chance"] * 100).round(0),
                         "Batch": scored["batch_id"], "Category": scored["category"], "Product": scored["sub_category"],
                         "Days left": scored["days_left"], "Units left": scored["units_left"],
                         "Why": scored["reason"], "Suggested": scored["suggested"], "Action taken": "Not decided"})
    edited = st.data_editor(show, hide_index=True, disabled=[c for c in show.columns if c != "Action taken"],
                            column_config={"Action taken": st.column_config.SelectboxColumn(options=S.ACTIONS)})
    scored["action"] = edited["Action taken"].values
    if st.button(f"Save Day {day}", type="primary"):
        H.save_day(day, scored, model_name)
        for k in (f"up_{day}", f"demo_{day}"):
            st.session_state.pop(k, None)
        st.rerun()


# ---------------------------------------------------------------- final dashboard
def final_dashboard(hist):
    st.header("Final dashboard - the 7-day week")
    a = D.combine(hist)
    s = D.week_summary(a)
    c = st.columns(4)
    c[0].metric("Days covered", f"{s['days']} (Day 0-7)"); c[1].metric("Batches tracked", s["batches"])
    c[2].metric("Batches flagged High", s["high_batches"]); c[3].metric("High-risk alerts raised", s["high_alerts"])

    st.subheader("Risk trend, day by day")
    st.pyplot(C.risk_trend(D.daily_counts(a)))
    st.caption("Each bar is one evening's file: how many batches were High, Medium or Low risk.")

    st.subheader("Risk by category")
    cat = D.by_category(a)
    c1, c2 = st.columns([3, 2])
    c1.pyplot(C.risk_by_category(cat))
    with c2:
        table(cat, {"average chance of waste": "{:.0%}"})

    st.subheader("Repeat-risk products (High on 3 or more days)")
    rep = D.repeat_risk(a)
    if len(rep):
        table(rep, {"last_chance": "{:.0%}"})
    else:
        st.write("None this week.")

    st.subheader("Actions the manager took on High-risk rows")
    t, decided = D.actions_summary(a)
    st.write(f"Action recorded on {pct(decided)} of High-risk rows.")
    table(t)

    st.subheader("Prediction check - predicted vs what really happened")
    pc = D.prediction_check(a)
    st.write(f"Only batches whose life ended inside the week can be checked: **{pc['finished']} batches** ({pc['rows']} daily rows). "
             f"{pc['open']} batches were still on the shelf at the end of the week, so their result is not known yet.")
    if "metrics" in pc:
        m = pc["metrics"]
        c = st.columns(4)
        for col, (lab, k) in zip(c, [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("F1", "f1")]):
            col.metric(lab, f"{m[k]:.3f}")
        st.caption("Scores use a 50% line on every daily row. This is one small week, so use the full test scores in 'Model Train Details' as the model's accuracy.")
        c1, c2 = st.columns([1, 2])
        mt = pd.DataFrame({"tn": [m["tn"]], "fp": [m["fp"]], "fn": [m["fn"]], "tp": [m["tp"]]}, index=["Model"])
        c1.pyplot(C.confusion(mt, "Model"))
        with c2:
            table(pc["batch_table"].rename(columns={"highest_chance": "highest chance", "really_at_risk": "really at risk",
                                                    "units_wasted": "units wasted"}), {"highest chance": "{:.0%}"})

    st.download_button("Download full week (CSV)", a.to_csv(index=False), "freshsave_demo_week.csv", "text/csv")


# ---------------------------------------------------------------- page
def demo_page():
    if st.button("< Back"):
        st.session_state["page"] = "home"
        st.rerun()
    st.title("Demo")
    st.caption("A week of daily uploads for one store. Each evening: upload, check, predict, act. After Day 7 the final dashboard opens.")

    if not S.models_ready():
        st.warning("Models are not trained yet.")
        if st.button("Train now (about 1 minute)"):
            with st.spinner("Training..."):
                train_all()
            st.rerun()
        st.stop()

    hist = H.load_all()
    done = len(hist)
    locked = H.saved_model()
    day_tracker(done)

    if done == 0:
        model_name = st.radio("Model for the demo", list(S.MODEL_FILES), horizontal=True,
                              help="Decision Tree was the winner in the training comparison.")
    else:
        model_name = locked or "Decision Tree"
        st.caption(f"Model: **{model_name}** (fixed for the week)  |  Store: **{hist[0]['store_id'].iloc[0]}**")

    if done <= H.LAST_DAY:
        day_step(done, hist, model_name)
    else:
        final_dashboard(hist)

    if hist:
        with st.expander("Earlier days"):
            d = st.selectbox("Day", list(hist), format_func=lambda x: f"Day {x}")
            h = hist[d]
            st.dataframe(pd.DataFrame({"Risk": h["level"].map(ICON), "Chance (%)": (h["chance"] * 100).round(0), "Batch": h["batch_id"],
                                       "Category": h["category"], "Product": h["sub_category"], "Units left": h["units_left"],
                                       "Action taken": h["action"]}), hide_index=True)
        if st.button("Reset demo (start again)"):
            H.reset()
            for k in [k for k in st.session_state if k.startswith(("up_", "demo_"))]:
                st.session_state.pop(k)
            st.rerun()
