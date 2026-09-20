"""Interactive demo.

Four tabs, in the order the project was actually built:

  Score an application    the model as a working underwriting tool
  The leakage trap        what the same data gives you if you are careless
  Choosing a cutoff       move the threshold, watch the money move
  Monitoring              PSI, discrimination and calibration by vintage

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, serve  # noqa: E402

st.set_page_config(page_title="Credit risk underwriting",
                   page_icon="*", layout="wide")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

@st.cache_resource
def get_bundle():
    return serve.load_bundle()


@st.cache_data
def get_report(name: str):
    path = os.path.join(config.REPORT_DIR, name)
    if not os.path.exists(path):
        return None
    if name.endswith(".csv"):
        return pd.read_csv(path)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def asset(name: str) -> str | None:
    path = os.path.join(config.ASSET_DIR, name)
    return path if os.path.exists(path) else None


# Streamlit sizes a dataframe container independently of its contents, which
# leaves the final row sliced through the middle of the glyphs. It scrolls, so
# nothing is lost, but it reads as a rendering bug. Sizing the container to a
# whole number of rows fixes it.
ROW_PX = 35
HEADER_PX = 38


def show_table(frame: pd.DataFrame, max_rows: int = 12) -> None:
    rows = min(len(frame), max_rows)
    st.dataframe(frame, hide_index=True, use_container_width=True,
                 height=HEADER_PX + ROW_PX * rows)


metrics = get_report("metrics.json")
leak = get_report("leakage_experiment.json")
drift_payload = get_report("drift.json")
sweep = get_report("profit_sweep.csv")

st.title("Consumer credit underwriting")
st.caption("779,025 Lending Club loans, 2007-2016 vintages. Trained only on "
           "columns that existed at the moment of underwriting.")

try:
    bundle = get_bundle()
except FileNotFoundError:
    st.error("No trained model found. Run `make all` first "
             "(or `python -m src.train`).")
    st.stop()

tab_score, tab_leak, tab_cut, tab_mon = st.tabs(
    ["Score an application", "The leakage trap", "Choosing a cutoff",
     "Monitoring"])


# ---------------------------------------------------------------------------
# 1. score
# ---------------------------------------------------------------------------

with tab_score:
    st.subheader("Score an application")
    st.write("Only a handful of fields are asked for. Everything else falls "
             "back to the training median, and the result tells you how many "
             "fields that was -- a score built mostly from defaults deserves "
             "less weight than one built from a full application.")

    left, middle, right = st.columns(3)

    with left:
        st.markdown("**The loan**")
        loan_amnt = st.number_input("Amount requested ($)", 1000, 40000, 15000,
                                    step=500)
        term_months = st.selectbox("Term (months)", [36, 60], index=0)
        int_rate = st.slider("Offered APR (%)", 5.0, 31.0, 12.5, 0.25)
        purpose = st.selectbox(
            "Purpose",
            ["debt_consolidation", "credit_card", "home_improvement", "other",
             "major_purchase", "medical", "small_business", "car",
             "moving", "vacation", "house", "renewable_energy", "wedding"])

    with middle:
        st.markdown("**The borrower**")
        annual_inc = st.number_input("Annual income ($)", 10000, 500000, 65000,
                                     step=1000)
        fico = st.slider("FICO score", 660, 850, 690)
        dti = st.slider("Debt-to-income (%)", 0.0, 45.0, 18.0, 0.5)
        emp_length = st.selectbox(
            "Employment length",
            ["10+ years", "5 years", "2 years", "1 year", "< 1 year",
             "3 years", "4 years", "6 years", "7 years", "8 years", "9 years"])
        home_ownership = st.selectbox("Home ownership",
                                      ["MORTGAGE", "RENT", "OWN"])

    with right:
        st.markdown("**Credit file**")
        credit_history_months = st.slider("Months of credit history", 6, 600, 168)
        revol_util = st.slider("Revolving utilisation (%)", 0.0, 150.0, 45.0, 1.0)
        open_acc = st.number_input("Open accounts", 0, 60, 9)
        total_acc = st.number_input("Total accounts ever", 0, 120, 24)
        delinq_2yrs = st.number_input("Delinquencies, last 2 years", 0, 20, 0)
        inq_last_6mths = st.number_input("Credit enquiries, last 6 months",
                                         0, 20, 1)

    grade = st.select_slider(
        "Lending Club grade (their own assessment, if you have it)",
        options=["A", "B", "C", "D", "E", "F", "G"], value="C")

    application = serve.Application(
        loan_amnt=float(loan_amnt),
        term_months=int(term_months),
        int_rate=float(int_rate),
        annual_inc=float(annual_inc),
        dti=float(dti),
        fico_range_low=float(fico),
        fico_range_high=float(fico) + 4,
        emp_length=emp_length,
        home_ownership=home_ownership,
        purpose=purpose,
        grade=grade,
        revol_util=float(revol_util),
        open_acc=float(open_acc),
        total_acc=float(total_acc),
        delinq_2yrs=float(delinq_2yrs),
        inq_last_6mths=float(inq_last_6mths),
        credit_history_months=float(credit_history_months),
    )

    decision = serve.score(application)

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Probability of charge-off",
              "%.2f%%" % (100 * decision.probability_of_default))
    c2.metric("Break-even for this loan",
              "%.2f%%" % (100 * decision.breakeven_probability),
              help="Approve while the predicted probability sits below this. "
                   "It moves with the rate, term and size of the loan.")
    c3.metric("Expected value", "$%.0f" % decision.expected_value)
    c4.metric("Scorecard points", "%.0f" % decision.scorecard_points)

    if decision.recommendation == "approve":
        st.success("**Approve.** Predicted risk is below this loan's "
                   "break-even, so the expected value is positive.")
    else:
        st.error("**Decline.** Predicted risk exceeds what the interest on "
                 "this loan can cover.")

    st.caption("%d fields supplied, %d filled from training defaults."
               % (decision.fields_supplied, decision.fields_defaulted))

    if decision.adverse_action_reasons:
        st.markdown("**Principal reasons for the score** — the scorecard "
                    "variables costing this applicant the most points. This is "
                    "what goes in an adverse action notice; it comes straight "
                    "from the fitted points table, not from a post-hoc "
                    "explanation of a black box.")
        show_table(
            pd.DataFrame([r.model_dump() for r in decision.adverse_action_reasons])
            .rename(columns={"variable": "Variable", "value": "Band",
                             "points_lost": "Points below best band"}))


# ---------------------------------------------------------------------------
# 2. leakage
# ---------------------------------------------------------------------------

with tab_leak:
    st.subheader("The same dataset, four ways of evaluating it")
    if not leak:
        st.info("Run `python -m src.leakage_experiment` to populate this tab.")
    else:
        dec = leak["decomposition"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Careless setup", "%.4f" % dec["careless_auc"],
                  help="Random split, post-origination columns left in.")
        c2.metric("Honest setup", "%.4f" % dec["honest_auc"],
                  help="Out-of-time split, origination columns only.")
        c3.metric("Overstatement", "%+.4f" % dec["total_overstatement"])

        img = asset("leakage_grid.png")
        if img:
            st.image(img, use_container_width=True)

        st.markdown("""
Both mistakes are usually made together, which makes it impossible to say which
one did the damage. Holding one fixed at a time separates them:

- **post-origination columns: %+.4f.** This is nearly the whole gap.
- **random instead of out-of-time split: %+.4f.** Almost nothing, and the sign
  is the opposite of what I predicted before running it.

The second result is worth being straight about. I expected a random split to
flatter the model. It did not, because the population is filtered to loans that
had their full term to mature, which removes the 2017-18 vintages entirely and
leaves default rates of 14-16%% across every remaining year. There is barely any
drift left for a random split to exploit.
""" % (dec["attributable_to_leakage"], dec["attributable_to_random_split"]))

        st.markdown("**Each post-origination column, used entirely on its own:**")
        singles = pd.DataFrame(leak["single_feature_aucs"])
        show_table(
            singles.rename(columns={"column": "Column", "auc_alone": "AUC alone",
                                    "non_null_share": "Non-null",
                                    "reason": "Why it leaks"}))

        img = asset("single_feature_auc.png")
        if img:
            st.image(img, use_container_width=True)


# ---------------------------------------------------------------------------
# 3. cutoff
# ---------------------------------------------------------------------------

with tab_cut:
    st.subheader("Where to cut, and what it costs")
    if metrics is None or sweep is None:
        st.info("Run `python -m src.train` to populate this tab.")
    else:
        st.write("Drag the cutoff. Every application with a predicted "
                 "probability below it is approved. The numbers are realised "
                 "outcomes on 372,810 out-of-time applications.")

        threshold = st.slider("Decline above this probability", 0.0, 1.0, 0.28,
                              0.005)
        nearest = sweep.iloc[(sweep["threshold"] - threshold).abs().idxmin()]
        approve_all = next(p for p in metrics["policies"]
                           if "approve all" in p["policy"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Approved", "%.1f%%" % (100 * nearest["approval_rate"]))
        c2.metric("Default rate of the book",
                  "%.2f%%" % (100 * nearest["book_default_rate"]))
        c3.metric("Total profit", "$%.1fm" % (nearest["total_profit"] / 1e6))
        c4.metric("vs approving everyone",
                  "$%+.1fm" % ((nearest["total_profit"]
                                - approve_all["total_profit"]) / 1e6))

        st.divider()
        st.markdown("**Named policies, compared on the same book:**")
        pol = pd.DataFrame(metrics["policies"])
        pol = pol[["policy", "approval_rate", "book_default_rate",
                   "total_profit", "profit_per_application"]]
        pol["approval_rate"] = (100 * pol["approval_rate"]).round(1)
        pol["book_default_rate"] = (100 * pol["book_default_rate"]).round(2)
        pol["total_profit"] = pol["total_profit"].round(0)
        pol["profit_per_application"] = pol["profit_per_application"].round(2)
        show_table(
            pol.rename(columns={"policy": "Policy",
                                "approval_rate": "Approved %",
                                "book_default_rate": "Book default %",
                                "total_profit": "Total profit $",
                                "profit_per_application": "Per application $"}))

        img = asset("profit_curve.png")
        if img:
            st.image(img, use_container_width=True)

        thr = metrics["thresholds"]
        st.markdown("""
The break-even probability is **not one number**. The middle 90%% of this test
book breaks even between %.1f%% and %.1f%%, median %.1f%%, because a 60-month
loan at 26%% earns far more interest than a 36-month loan at 7%% and can
therefore carry far more risk. Any single global cutoff is an approximation to
that. (%d loans out of %s earn no interest at all — scheduled payments that do
not exceed principal — so they break even at zero and can never be approved on
expected value.)
""" % (100 * thr["breakeven_p_p05"], 100 * thr["breakeven_p_p95"],
       100 * thr["breakeven_p_median"], thr["n_zero_interest"],
       f"{metrics['population']['n_test']:,}"))


# ---------------------------------------------------------------------------
# 4. monitoring
# ---------------------------------------------------------------------------

with tab_mon:
    st.subheader("Is the model still being asked the question it was built for?")
    if not drift_payload:
        st.info("Run `python -m src.monitor` to populate this tab.")
    else:
        gate = drift_payload["gate"]
        comparison = drift_payload["comparison"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Champion AUC (LightGBM)",
                  "%.4f" % comparison["champion_lightgbm"]["auc"])
        c2.metric("Challenger AUC (scorecard)",
                  "%.4f" % comparison["challenger_scorecard"]["auc"],
                  delta="%+.4f" % (comparison["challenger_scorecard"]["auc"]
                                   - comparison["champion_lightgbm"]["auc"]))
        c3.metric("Gate decision",
                  "PROMOTE" if gate["promoted"] else "KEEP CHAMPION")

        st.markdown("**Promotion gate**")
        checks = pd.DataFrame(gate["checks"])
        checks["passed"] = checks["passed"].map({True: "PASS", False: "FAIL"})
        show_table(
            checks.rename(columns={"name": "Check", "passed": "Result",
                                   "detail": "Detail"}))

        st.caption("The interpretable scorecard loses on all three quality "
                   "checks, but only just. That gap is the price of being able "
                   "to explain a decline, expressed as a number the business "
                   "can accept or reject.")

        img = asset("drift_monitor.png")
        if img:
            st.image(img, use_container_width=True)

        st.markdown("**Most-shifted input variables**")
        charac = pd.DataFrame(drift_payload["characteristic_analysis"]).head(10)
        show_table(
            charac.rename(columns={"variable": "Variable", "kind": "Type",
                                   "psi": "PSI", "verdict": "Verdict"}))

        st.caption("The largest shift is `initial_list_status`, which records "
                   "whether Lending Club listed a loan whole or fractionally. "
                   "That is a change in their own operations, not in borrower "
                   "quality -- which is exactly what per-variable PSI is for.")
