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


def day_tracker(done, status="waiting"):
    """One column per day. Green tick = day saved. Today shows the upload status."""
    cols = st.columns(8)
    for d, col in enumerate(cols):
        if d < done:
            state = ":green[**✔ Done**]"
        elif d == done:
            state = {"waiting": "▶ Next: upload file", "uploaded": ":green[**✔ File uploaded**]",
                     "error": ":red[**✖ Fix the file**]"}[status]
        else:
            state = ":gray[Waiting]"
        col.markdown(f"**Day {d}**  \n{DAY_NAMES[d]}  \n{state}")
    st.progress(done / 8)


def show_tracker(slot, done, status):
    with slot.container():
        day_tracker(done, status)


def step_line(flags):
    """Mini checklist for the day in progress."""
    names = ["File uploaded", "File checked", "Risk predicted", "Action recorded", "Day saved"]
    return "   ".join((":green[✔ " + n + "]") if f else (":gray[○ " + n + "]") for n, f in zip(names, flags))


@st.cache_resource
def get_model(name):
    return S.load_model(name)


def table(df, fmt=None):
    st.dataframe(df.style.format(fmt) if fmt else df, hide_index=True)


# ---------------------------------------------------------------- day step
def day_step(day, hist, model_name, tracker):
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
        st.markdown(step_line([0, 0, 0, 0, 0]))
        return
    prev = hist[day - 1]["date"].iloc[0] if day > 0 else None
    store = hist[0]["store_id"].iloc[0] if day > 0 else None
    errors, warnings = S.check_file(df, prev, store)
    for w in warnings:
        st.warning(w)
    if errors:
        for e in errors:
            st.error(e)
        show_tracker(tracker, day, "error")
        st.stop()
    show_tracker(tracker, day, "uploaded")
    st.success(f"File OK: {len(df)} batches, store {df['store_id'].iloc[0]}, date {df['date'].iloc[0]}.")

    scored = S.score(df, model_name, get_model(model_name))
    n = scored["level"].value_counts()
    c = st.columns(4)
    c[0].metric("Batches checked", len(scored))
    for col, lv in zip(c[1:], ["High", "Medium", "Low"]):
        col.metric(f"{ICON[lv]} risk", int(n.get(lv, 0)))
    st.caption(f"Units on the shelf in High-risk batches: {int(scored.loc[scored['level'] == 'High', 'units_left'].sum()):,}")

    steps = st.empty()
    st.subheader("Today's risk list (highest risk first)")
    if day == 0:
        st.caption("This is the inventory loaded on Saturday night. Suggestions and actions start from Day 1.")
        scored["suggested"], scored["action"] = "", ""
    scored["action"] = risk_table(day, scored)
    acted = day == 0 or any(a != "Not decided" for a in scored["action"])
    steps.markdown(step_line([1, 1, 1, acted, 0]) + ("" if day else "   (no actions on Day 0)"))
    if st.button(f"Save Day {day}", type="primary"):
        H.save_day(day, scored, model_name)
        for k in [k for k in st.session_state if k.startswith((f"up_{day}", f"demo_{day}", f"act_{day}_"))]:
            st.session_state.pop(k, None)
        st.rerun()


def risk_table(day, scored):
    """Table with one dropdown per row (Day 1 onward). Each box changes colour after a choice."""
    heads = ["Risk", "Chance %", "Batch", "Category", "Product", "Days left", "Units left", "Why"]
    widths = [1, 0.9, 1, 1.2, 1.6, 0.8, 0.9, 4]
    if day > 0:
        heads += ["Suggested by Model", "Action taken"]
        widths += [2, 2]
    for col, h in zip(st.columns(widths), heads):
        col.markdown(f"**{h}**")
    chosen, css = [], []
    for i, r in scored.iterrows():
        cols = st.columns(widths)
        vals = [ICON[r["level"]], f"{r['chance'] * 100:.0f}", r["batch_id"], r["category"], r["sub_category"],
                int(r["days_left"]), int(r["units_left"]), r["reason"]]
        for col, v in zip(cols, vals):
            col.write(v)
        if day > 0:
            cols[8].write(r["suggested"])
            key = f"act_{day}_{i}"
            pick = cols[9].selectbox("Action taken", S.ACTIONS, key=key, label_visibility="collapsed")
            chosen.append(pick)
            css.append(f'.st-key-{key} [data-baseweb="select"] > div {{ background-color: {S.action_colour(pick)} !important; }}'
                       f'.st-key-{key} [data-baseweb="select"] * {{ color: #111 !important; }}')
    if css:
        st.markdown("<style>" + "".join(css) + "</style>", unsafe_allow_html=True)
    return chosen if day > 0 else [""] * len(scored)


# ---------------------------------------------------------------- final dashboard
def final_dashboard(hist):
    st.header("Final dashboard - the 7-day week")
    a = D.combine(hist)
    k = D.kpis(a)
    st.caption("Day 0 (Saturday night) to Day 7 (Saturday evening). One store. Simulated data.")

    st.subheader("Key numbers")
    c = st.columns(4)
    c[0].metric("Batches tracked", k["batches"])
    c[1].metric("Flagged High risk", f"{k['high_pct'] * 100:.0f}%", f"{k['high_batches']} of {k['batches']} batches", delta_color="off")
    c[2].metric("Alerts acted on", f"{k['action_pct'] * 100:.0f}%", "High-risk rows with an action", delta_color="off")
    c[3].metric("Batches still open", f"{k['open_pct'] * 100:.0f}%", f"{k['open']} not finished this week", delta_color="off")
    st.markdown("**How well did the model do?** (only batches whose life ended this week)")
    c = st.columns(3)
    c[0].metric("Waste caught", f"{k['caught_pct'] * 100:.0f}%", f"{k['caught']} of {k['really_at_risk']} wasting batches flagged", delta_color="off")
    c[1].metric("Alerts that were right", f"{k['correct_alerts_pct'] * 100:.0f}%", f"{k['caught']} of {k['flagged']} High alerts", delta_color="off")
    c[2].metric("Overall accuracy", f"{k['accuracy_pct'] * 100:.0f}%", f"on {k['finished']} finished batches", delta_color="off")
    st.caption("One small week of data. The full test scores in 'Model Train Details' are the model's real accuracy.")

    st.subheader("% of the shelf at High risk, day by day")
    daily = D.daily_high_pct(a)
    st.pyplot(C.pct_bars_vertical([f"Day {d}" for d in daily.index], list(daily.values), "% of batches High risk"))

    st.subheader("% of batches flagged High, by category")
    cat = D.category_high_pct(a)
    st.pyplot(C.pct_bars_horizontal(list(cat["category"]), list(cat["pct"]), "% of batches flagged High at least once"))

    with st.expander("How each number is calculated"):
        st.markdown(f"""
- **Batches tracked** - different batches seen in the 8 files.
- **Flagged High risk** - batches that were High (70% chance of waste or more) at least once, divided by batches tracked.
- **Alerts acted on** - High-risk rows from Day 1 to Day 7 where the manager chose an action, divided by all High-risk rows.
- **Batches still open** - batches whose shelf life had not ended by Day 7, so we do not know their result yet.
- **Waste caught** - of the batches that really ended up wasted, how many were flagged High at least once.
- **Alerts that were right** - of the batches flagged High, how many really ended up wasted.
- **Overall accuracy** - finished batches where the model's High flag matched what really happened.
""")
    with st.expander("Batch-by-batch check"):
        pc = D.prediction_check(a)
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
    tracker = st.empty()
    show_tracker(tracker, done, "waiting")

    if done == 0:
        model_name = st.radio("Model for the demo", list(S.MODEL_FILES), horizontal=True,
                              help="Decision Tree was the winner in the training comparison.")
    else:
        model_name = locked or "Decision Tree"
        st.caption(f"Model: **{model_name}** (fixed for the week)  |  Store: **{hist[0]['store_id'].iloc[0]}**")

    if done <= H.LAST_DAY:
        day_step(done, hist, model_name, tracker)
    else:
        final_dashboard(hist)

    if hist:
        with st.expander("Earlier days"):
            d = st.selectbox("Day", list(hist), format_func=lambda x: f"Day {x}")
            h = hist[d]
            view = pd.DataFrame({"Risk": h["level"].map(ICON), "Chance (%)": (h["chance"] * 100).round(0), "Batch": h["batch_id"],
                                 "Category": h["category"], "Product": h["sub_category"], "Units left": h["units_left"]})
            if d > 0:
                view["Suggested by Model"] = h["suggested"]
                view["Action taken"] = h["action"]
            st.dataframe(view, hide_index=True)
        if st.button("Reset demo (start again)"):
            H.reset()
            for k in [k for k in st.session_state if k.startswith(("up_", "demo_"))]:
                st.session_state.pop(k)
            st.rerun()
