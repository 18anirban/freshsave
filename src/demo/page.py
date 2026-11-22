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
def compact_uploader_css(day):
    """Make the upload box small: just a Browse button and the file name."""
    st.markdown(f"""<style>
.st-key-up_{day} [data-testid="stFileUploaderDropzone"] {{ padding: 0.35rem 0.6rem; min-height: 0; }}
.st-key-up_{day} [data-testid="stFileUploaderDropzoneInstructions"] {{ display: none; }}
</style>""", unsafe_allow_html=True)


def apply_bulk(day, ids, bulk_key):
    """Callback: give the chosen action to every batch currently shown."""
    action = st.session_state[bulk_key]
    acts = st.session_state.setdefault(f"acts_{day}", {})
    for b in ids:
        acts[b] = action
        st.session_state[f"act_{day}_{b}"] = action


def day_step(day, hist, model_name, tracker):
    title = ("Opening stock: inventory loaded on Saturday night, before any sales" if day == 0
             else f"{DAY_NAMES[day]} evening closing report (sales of the day and stock left)")
    st.subheader(f"Day {day} - upload: {title}")
    compact_uploader_css(day)
    c_up, c_demo, c_save = st.columns([3, 3, 1.6])
    up = c_up.file_uploader("Upload CSV", type="csv", key=f"up_{day}", label_visibility="collapsed")
    use_demo = c_demo.checkbox(f"Use demo file ({day_file_name(day)})", key=f"demo_{day}")
    st.caption("CSV with the same columns as the training data (no outcome columns).")
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
    acts = st.session_state.setdefault(f"acts_{day}", {})
    if c_save.button(f"Save Day {day}", type="primary", key=f"save_{day}"):          # right beside the upload box
        if day == 0:
            scored["suggested"], scored["action"] = "", ""
        else:
            scored["action"] = [acts.get(b, "Not decided") for b in scored["batch_id"]]
        H.save_day(day, scored, model_name)
        prefixes = (f"up_{day}", f"demo_{day}", f"act_{day}_", f"acts_{day}", f"fcat_{day}", f"flvl_{day}", f"bulk_{day}")
        for k in [k for k in st.session_state if str(k).startswith(prefixes)]:
            st.session_state.pop(k, None)
        st.rerun()

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
    view = filter_bar(day, scored, acts)
    risk_table(day, view, acts)
    acted = day == 0 or any(v != "Not decided" for v in acts.values())
    steps.markdown(step_line([1, 1, 1, acted, 0]) + ("" if day else "   (no actions on Day 0)"))


def filter_bar(day, scored, acts):
    """Dropdowns to show one category / one risk type, and one action for all rows shown."""
    c = st.columns([1.5, 1.5, 2.2, 1.8])
    cat = c[0].selectbox("Category", ["All"] + sorted(scored["category"].unique()), key=f"fcat_{day}")
    lvl = c[1].selectbox("Risk type", ["All", "High", "Medium", "Low"], key=f"flvl_{day}")
    view = scored
    if cat != "All":
        view = view[view["category"] == cat]
    if lvl != "All":
        view = view[view["level"] == lvl]
    if day > 0:
        c[2].selectbox("Action for all rows shown", S.ACTIONS[1:], index=S.ACTIONS[1:].index("Discount 50%"), key=f"bulk_{day}")
        c[3].markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        c[3].button(f"Apply to {len(view)} rows", key=f"apply_{day}", on_click=apply_bulk,
                    args=(day, list(view["batch_id"]), f"bulk_{day}"))
        st.caption(f"Showing {len(view)} of {len(scored)} batches. Pick a category and risk type, choose an action, click Apply "
                   "to give it to all rows shown, or choose an action row by row.")
    else:
        st.caption(f"Showing {len(view)} of {len(scored)} batches.")
    return view


def risk_table(day, view, acts):
    """Table with one dropdown per row (Day 1 onward). Each box changes colour after a choice."""
    heads = ["Risk", "Chance %", "Batch", "Category", "Product", "Days left", "Units left", "Why"]
    widths = [1, 0.9, 1, 1.2, 1.6, 0.8, 0.9, 4]
    if day > 0:
        heads += ["Suggested by Model", "Action taken"]
        widths += [2, 2]
    for col, h in zip(st.columns(widths), heads):
        col.markdown(f"**{h}**")
    css = []
    for _, r in view.iterrows():
        cols = st.columns(widths)
        vals = [ICON[r["level"]], f"{r['chance'] * 100:.0f}", r["batch_id"], r["category"], r["sub_category"],
                int(r["days_left"]), int(r["units_left"]), r["reason"]]
        for col, v in zip(cols, vals):
            col.write(v)
        if day > 0:
            cols[8].write(r["suggested"])
            key = f"act_{day}_{r['batch_id']}"
            if key not in st.session_state:                       # box was hidden by a filter, restore its choice
                st.session_state[key] = acts.get(r["batch_id"], "Not decided")
            pick = cols[9].selectbox("Action taken", S.ACTIONS, key=key, label_visibility="collapsed")
            acts[r["batch_id"]] = pick
            css.append(f'.st-key-{key} [data-baseweb="select"] > div {{ background-color: {S.action_colour(pick)} !important; }}'
                       f'.st-key-{key} [data-baseweb="select"] * {{ color: #111 !important; }}')
    if css:
        st.markdown("<style>" + "".join(css) + "</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------- final dashboard
def box(text):
    """Explanation box: every number and rule on the dashboard is explained in one of these."""
    st.info(text)


def final_dashboard(hist):
    st.header("Final dashboard - the 7-day week")
    st.caption("Day 0 (Saturday night) to Day 7 (Saturday evening). One store. Simulated data.")
    a = D.combine(hist)
    k = D.kpis(a)
    fin = D.finished_batches()
    wh = D.waste_headline(fin)

    # ---------------- 1. headline KPIs
    st.subheader("1. Key numbers for the week")
    c = st.columns(5)
    c[0].metric("Waste rate", f"{wh['waste_rate'] * 100:.0f}%", f"{wh['wasted']:,.0f} of {wh['received']:,.0f} units", delta_color="off")
    c[1].metric("Waste cost", f"${wh['waste_cost']:,.0f}", f"{wh['batches_with_waste']} batches with waste", delta_color="off")
    c[2].metric("Waste caught", f"{k['caught_pct'] * 100:.0f}%", f"{k['caught']} of {k['really_at_risk']} wasting batches", delta_color="off")
    c[3].metric("Alerts that were right", f"{k['correct_alerts_pct'] * 100:.0f}%", f"{k['caught']} of {k['flagged']} High alerts", delta_color="off")
    c[4].metric("Alerts acted on", f"{k['action_pct'] * 100:.0f}%", "High-risk rows with an action", delta_color="off")
    box(f"""**How these are calculated**  (on the {wh['finished']} batches whose life ended this week; {k['open']} batches were still on the shelf, so their result is not known yet)

- **Waste rate** = units wasted / units received = {wh['wasted']:,.0f} / {wh['received']:,.0f} = {wh['waste_rate'] * 100:.1f}%.
- **Waste cost** = units wasted x cost price of each batch, added up = ${wh['waste_cost']:,.0f}.
- **Waste caught** = of the {k['really_at_risk']} batches that really ended up at risk (20% or more wasted), how many the model flagged High (70% chance or more) at least once = {k['caught']} / {k['really_at_risk']}.
- **Alerts that were right** = of the {k['flagged']} batches flagged High, how many really ended up at risk = {k['caught']} / {k['flagged']}.
- **Alerts acted on** = High-risk rows from Day 1 to Day 7 where the manager picked an action / all High-risk rows.""")

    # ---------------- 2. category input vs waste
    st.subheader("2. Input vs waste by category")
    cat = D.category_input_waste(fin)
    st.pyplot(C.category_input_waste(cat))
    table(cat.rename(columns={"received": "input (units received)", "wasted": "final (units wasted)", "waste_pct": "waste %",
                              "waste_cost": "waste cost ($)"}), {"waste %": "{:.0%}", "waste cost ($)": "${:,.0f}"})
    worst = cat.iloc[0]
    box(f"""**How to read it**

- **Input** (blue) = units received in the batches that finished this week. **Final** (red) = units that ended up wasted.
- **Waste %** = wasted / received for the category. Highest this week: **{worst['category']}** at {worst['waste_pct'] * 100:.0f}% ({worst['wasted']:,.0f} of {worst['received']:,.0f} units).
- Each category adds up its batches, so a big batch counts more than a small one.""")
    with st.expander("Every finished batch: input vs final"):
        show = fin[["batch_id", "category", "product", "received", "wasted", "waste_pct", "waste_cost"]].sort_values("waste_pct", ascending=False)
        table(show.rename(columns={"received": "input (units received)", "wasted": "final (units wasted)", "waste_pct": "waste %",
                                   "waste_cost": "waste cost ($)"}), {"waste %": "{:.0%}", "waste cost ($)": "${:,.2f}"})

    # ---------------- 3. next-week orders
    st.subheader("3. Recommended orders for next week")
    rules, fcat = D.order_rules(fin)
    nw = D.apply_rules(rules, fcat, D.next_week_batches())
    sm = D.savings_summary(nw)
    rc = D.rolling_check()
    c = st.columns(4)
    c[0].metric("Waste saved next week", f"{sm['waste_saving_pct'] * 100:.0f}%", f"{sm['wasted_now']:,.0f} to {sm['waste_after']:,.0f} units", delta_color="off")
    c[1].metric("Waste rate", f"{sm['waste_rate_after'] * 100:.0f}%", f"was {sm['waste_rate_now'] * 100:.0f}%", delta_color="off")
    c[2].metric("Money saved", f"${sm['money_saved']:,.0f}", "waste cost avoided", delta_color="off")
    c[3].metric("Sales at risk", f"{sm['lost_sales_pct'] * 100:.0f}%", "units that might not be in stock", delta_color="off")
    st.caption(f"Typical result over {len(rc)} different weeks with the same rule: {rc['waste_saving_pct'].mean() * 100:.0f}% waste saved "
               f"(lowest {rc['waste_saving_pct'].min() * 100:.0f}%, highest {rc['waste_saving_pct'].max() * 100:.0f}%), "
               f"{rc['lost_sales_pct'].mean() * 100:.0f}% sales at risk.")
    top = rules.iloc[0]
    box(f"""**The recommendation logic, step by step**

1. **Take the finished batches of the week** and, for each product, add up the units ordered (received) and the units wasted.
2. **Units that really sold** = received - wasted.
3. **Recommended order = units that sold x 1.10.** The 10% is a safety buffer, so we do not run out. The order is never raised above what was ordered.
4. **Change % = recommended / ordered - 1.**
5. A product with fewer than 2 finished batches has too little history, so it uses its category's change instead (basis column in the table).

*Worked example:* **{top['product']}** - {top['batches']} batches, {top['received']:,.0f} units ordered, {top['wasted']:,.0f} wasted, so {top['used']:,.0f} sold. {top['used']:,.0f} x 1.10 = {top['used'] * 1.1:,.0f}. {top['used'] * 1.1:,.0f} / {top['received']:,.0f} = {top['factor']:.2f}, so order **{abs(top['change_pct']) * 100:.0f}% less**.

**How the saving is worked out.** We did not use this week's own results (that would be hindsight). The rule is applied to the **{sm['batches']} real batches delivered the week after** (10-16 Nov), which it has never seen:
- **New order** = old order x the product's factor, rounded up.
- **Waste after** = new order - units that really sold (not below 0).
- **Waste saved %** = 1 - waste after / waste before = 1 - {sm['waste_after']:,.0f} / {sm['wasted_now']:,.0f}.
- **Money saved** = waste avoided x cost price.
- **Sales at risk** = units that sold last time but would be above the new, smaller order, as a share of units sold. This is the price of ordering less. It is a minimum: where a batch sold out, we cannot see the extra demand.

**Limits.** It assumes next week sells like this week. A big promotion, a holiday or weather would change that. Based on one week of history per product, so treat it as guidance for the manager, not a guarantee.""")
    cols = ["product", "category", "batches", "avg_ordered", "waste_pct", "change_pct", "avg_recommended", "basis"]
    table(rules[cols].rename(columns={"avg_ordered": "avg order per batch now", "waste_pct": "waste %", "change_pct": "recommended change",
                                      "avg_recommended": "recommended order per batch"}),
          {"avg order per batch now": "{:,.0f}", "waste %": "{:.0%}", "recommended change": "{:+.0%}", "recommended order per batch": "{:,.0f}"})
    with st.expander(f"All {sm['batches']} batches of next week: old order, recommended order, waste"):
        t = nw[["batch_id", "category", "product", "received", "recommended", "wasted", "waste_after", "lost_sales", "money_saved"]]
        table(t.rename(columns={"received": "old order", "recommended": "recommended order", "wasted": "waste with old order",
                                "waste_after": "waste with recommended", "lost_sales": "sales at risk", "money_saved": "money saved ($)"}),
              {"money saved ($)": "${:,.0f}"})

    # ---------------- 4. risk during the week
    with st.expander("Risk during the week (% of batches High risk)"):
        daily = D.daily_high_pct(a)
        st.pyplot(C.pct_bars_vertical([f"Day {d}" for d in daily.index], list(daily.values), "% of batches High risk"))
        cat2 = D.category_high_pct(a)
        st.pyplot(C.pct_bars_horizontal(list(cat2["category"]), list(cat2["pct"]), "% of batches flagged High at least once"))
        box("**High risk** = the model gives a 70% chance of waste or more. Daily bar = High-risk batches / batches on the shelf that evening.")
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
