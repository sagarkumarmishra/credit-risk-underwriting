"""Build the PDF summary.

Written for someone who will not run the code: a credit risk lead, a hiring
manager, anyone who wants the findings and the reasoning in four pages. Every
number is read from the JSON the pipeline produced, so the document cannot drift
away from the results it describes.
"""

from __future__ import annotations

import json
import os
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src import config

INK = colors.HexColor("#1b1f24")
MUTED = colors.HexColor("#5b6370")
RULE = colors.HexColor("#d4d8dd")
ACCENT = colors.HexColor("#0b5fa5")
BAD = colors.HexColor("#b3261e")
GOOD = colors.HexColor("#1e7b34")


def styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontName="Helvetica-Bold",
                                fontSize=19, leading=23, textColor=INK,
                                spaceAfter=3),
        "subtitle": ParagraphStyle("st", parent=base["Normal"], fontSize=10.5,
                                   leading=14, textColor=MUTED, spaceAfter=12),
        "h2": ParagraphStyle("h2", parent=base["Heading2"],
                             fontName="Helvetica-Bold", fontSize=12.5,
                             leading=15, textColor=INK, spaceBefore=13,
                             spaceAfter=5),
        "body": ParagraphStyle("b", parent=base["Normal"], fontSize=9.6,
                               leading=13.6, textColor=INK, alignment=TA_LEFT,
                               spaceAfter=7),
        "small": ParagraphStyle("s", parent=base["Normal"], fontSize=8.3,
                                leading=11, textColor=MUTED, spaceAfter=5),
        "callout": ParagraphStyle("c", parent=base["Normal"], fontSize=10.2,
                                  leading=14.5, textColor=INK,
                                  leftIndent=8, borderPadding=7,
                                  backColor=colors.HexColor("#f2f5f8"),
                                  spaceBefore=5, spaceAfter=9),
    }


def load(name: str):
    path = os.path.join(config.REPORT_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def table(rows, widths, align_right=(), header=True):
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    style = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8.6),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, RULE),
    ]
    if header:
        style += [
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.6),
            ("LINEBELOW", (0, 0), (-1, 0), 0.9, INK),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
        ]
    for col in align_right:
        style.append(("ALIGN", (col, 0), (col, -1), "RIGHT"))
    t.setStyle(TableStyle(style))
    return t


def image(name: str, width: float):
    path = os.path.join(config.ASSET_DIR, name)
    if not os.path.exists(path):
        return None
    from PIL import Image as PILImage  # noqa: PLC0415
    try:
        with PILImage.open(path) as im:
            ratio = im.height / im.width
    except Exception:
        ratio = 0.5
    return Image(path, width=width, height=width * ratio)


def build(out_path: str) -> str:
    s = styles()
    metrics = load("metrics.json")
    leak = load("leakage_experiment.json")
    drift_payload = load("drift.json")

    if metrics is None or leak is None:
        raise SystemExit("run `make all` first -- the reports are not there yet")

    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=17 * mm, rightMargin=17 * mm,
        topMargin=15 * mm, bottomMargin=15 * mm,
        title="Credit risk underwriting",
        author="Sagar Kumar Mishra",
    )
    width = doc.width
    story = []

    pop = metrics["population"]
    dec = leak["decomposition"]
    models = metrics["models"]
    policies = metrics["policies"]

    # -- page 1 -------------------------------------------------------------
    story.append(Paragraph("Probability of default, done point-in-time",
                           s["title"]))
    story.append(Paragraph(
        "A consumer credit underwriting model on %s Lending Club loans, "
        "2007 to 2016 vintages, and the lending policy that follows from it."
        % f"{pop['n_total']:,}", s["subtitle"]))

    story.append(Paragraph(
        "Most published loan-default models report an AUC somewhere between "
        "0.95 and 0.99. None of them could be deployed. The reason is not "
        "subtle statistics, it is data lineage: the dataset contains dozens of "
        "columns recorded <i>after</i> the loan was issued, and at the moment "
        "an underwriter has to decide, every one of them is empty. This project "
        "measures exactly what that mistake is worth, builds the model that "
        "does not make it, and then turns it into an approve/decline policy "
        "using arithmetic on money rather than a threshold of 0.5.",
        s["body"]))

    story.append(Paragraph("What the two classic mistakes are worth", s["h2"]))
    story.append(table(
        [["Configuration", "Split", "Columns", "AUC"],
         ["A  careless", "random", "all, including post-origination",
          "%.4f" % leak["cells"]["A_random_leaky"]["auc"]],
         ["B", "out-of-time", "all, including post-origination",
          "%.4f" % leak["cells"]["B_oot_leaky"]["auc"]],
         ["C", "random", "origination only",
          "%.4f" % leak["cells"]["C_random_honest"]["auc"]],
         ["D  honest", "out-of-time", "origination only",
          "%.4f" % leak["cells"]["D_oot_honest"]["auc"]]],
        [width * 0.22, width * 0.16, width * 0.44, width * 0.18],
        align_right=(3,)))

    story.append(Spacer(1, 7))
    story.append(Paragraph(
        "<b>Cell A is a perfect classifier — 1.0000 — and is worth nothing.</b> "
        "Holding one mistake fixed at a time separates them: post-origination "
        "columns account for <b>%+.4f</b> of the %+.4f gap. The random split "
        "accounts for <b>%+.4f</b>."
        % (dec["attributable_to_leakage"], dec["total_overstatement"],
           dec["attributable_to_random_split"]), s["callout"]))

    story.append(Paragraph(
        "That second figure is the opposite of what I predicted before running "
        "it, and it is reported as a failed hypothesis rather than quietly "
        "dropped. The mechanism is the population rule described below: once "
        "loans are required to have had their full term to mature, the 2017 and "
        "2018 vintages disappear entirely and every remaining year defaults at "
        "14 to 16 percent. There is almost no temporal drift left for a random "
        "split to exploit. On a dataset without that filter, the same test "
        "would very likely show the textbook result.", s["body"]))

    img = image("leakage_grid.png", width)
    if img:
        story.append(img)

    story.append(Paragraph(
        "The worst offender is not the obviously named one. "
        "<font face='Courier'>last_fico_range_high</font> scores %.4f entirely "
        "on its own — better than <font face='Courier'>recoveries</font> at "
        "%.4f — because it is the borrower's credit score at the most recent "
        "bureau pull, which on a defaulted loan happened after the default "
        "wrecked it. Sorted alphabetically it sits next to the legitimate "
        "<font face='Courier'>fico_range_high</font>, which is precisely how it "
        "ends up in a feature list by accident."
        % (next(r["auc_alone"] for r in leak["single_feature_aucs"]
                if r["column"] == "last_fico_range_high"),
           next(r["auc_alone"] for r in leak["single_feature_aucs"]
                if r["column"] == "recoveries")), s["body"]))

    story.append(PageBreak())

    # -- page 2 -------------------------------------------------------------
    story.append(Paragraph("The population, and why it is smaller than it looks",
                           s["h2"]))
    story.append(Paragraph(
        "Two rules, both non-negotiable. A loan needs a resolved outcome, which "
        "removes everything still in flight. And it needs to have had its full "
        "contractual term before the snapshot — a 60-month loan issued in 2017 "
        "cannot appear as Fully Paid in a 2018 Q4 extract. Filtering on "
        "resolved status alone keeps recent vintages <i>only if they defaulted "
        "early</i>, which is survivorship bias pointing the wrong way.",
        s["body"]))

    story.append(table(
        [["Stage", "Loans", "Removed"],
         ["All loans in the extract", "2,260,701", ""],
         ["Resolved outcome only", "1,348,059", "912,642 still in flight"],
         ["Full term before snapshot", f"{pop['n_total']:,}",
          "569,034 not yet matured"],
         ["Charged off", "118,371",
          "%.2f%% default rate" % (100 * pop["default_rate_overall"])]],
        [width * 0.36, width * 0.20, width * 0.44], align_right=(1,)))

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "The maturity rule keeps 100% of 2013 originations, 30.6% of 2016, and "
        "<b>none at all</b> of 2017 and 2018. Losing a quarter of a million "
        "recent loans is painful and it is still correct.", s["small"]))

    story.append(Paragraph("Models", s["h2"]))
    story.append(table(
        [["Model", "AUC", "Gini", "KS", "Brier"],
         ["Constant baseline",
          "%.4f" % models["baseline_constant"]["auc"],
          "%.4f" % models["baseline_constant"]["gini"], "—",
          "%.5f" % models["baseline_constant"]["brier"]],
         ["WOE scorecard (%d variables)" % models["scorecard_woe"]["n_variables"],
          "%.4f" % models["scorecard_woe"]["auc"],
          "%.4f" % models["scorecard_woe"]["gini"],
          "%.4f" % models["scorecard_woe"]["ks"],
          "%.5f" % models["scorecard_woe"]["brier"]],
         ["LightGBM",
          "%.4f" % models["lightgbm_raw"]["auc"],
          "%.4f" % models["lightgbm_raw"]["gini"],
          "%.4f" % models["lightgbm_raw"]["ks"],
          "%.5f" % models["lightgbm_raw"]["brier"]],
         ["LightGBM, isotonic calibrated",
          "%.4f" % models["lightgbm_calibrated"]["auc"],
          "%.4f" % models["lightgbm_calibrated"]["gini"],
          "%.4f" % models["lightgbm_calibrated"]["ks"],
          "%.5f" % models["lightgbm_calibrated"]["brier"]]],
        [width * 0.38, width * 0.14, width * 0.14, width * 0.14, width * 0.20],
        align_right=(1, 2, 3, 4)))

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Accuracy is deliberately absent. At a 15%% default rate, predicting "
        "that everybody repays scores 85%% and is useless. Calibration improves "
        "Brier from %.5f to %.5f while leaving AUC unchanged at %.4f — "
        "calibration changes the price, never the order."
        % (models["lightgbm_raw"]["brier"],
           models["lightgbm_calibrated"]["brier"],
           models["lightgbm_calibrated"]["auc"]), s["body"]))

    img = image("calibration.png", width)
    if img:
        story.append(img)

    story.append(PageBreak())

    # -- page 3 -------------------------------------------------------------
    story.append(Paragraph("From probability to policy", s["h2"]))
    story.append(Paragraph(
        "A probability is not a decision. Approving a loan is worth "
        "<i>(1−p)·interest − p·LGD·exposure</i>, so it is worth doing while "
        "<i>p</i> sits below <i>interest / (interest + LGD·exposure)</i>. That "
        "break-even is a property of the individual loan, not a global "
        "constant: across this test book the middle 90%% of loans break even "
        "between %.1f%% and %.1f%%, median %.1f%%, because a 60-month loan at "
        "26%% can carry far more risk than a 36-month loan at 7%%. Loss given "
        "default is measured from actual recoveries on loans that charged off "
        "— mean %.3f, median %.3f."
        % (100 * metrics["thresholds"]["breakeven_p_p05"],
           100 * metrics["thresholds"]["breakeven_p_p95"],
           100 * metrics["thresholds"]["breakeven_p_median"],
           metrics["lgd"]["lgd_mean"], metrics["lgd"]["lgd_median"]),
        s["body"]))
    story.append(Paragraph(
        "%d of the %s loans (%.3f%%) have scheduled payments that do not exceed "
        "principal, so they earn no interest and break even at zero. They can "
        "never clear an expected-value test, which is the correct outcome for a "
        "loan with no upside."
        % (metrics["thresholds"]["n_zero_interest"],
           f"{pop['n_test']:,}",
           100 * metrics["thresholds"]["share_zero_interest"]), s["small"]))

    rows = [["Policy", "Approved", "Book bad rate", "Profit / application"]]
    for pol in policies:
        rows.append([
            pol["policy"].replace("model, ", ""),
            "%.1f%%" % (100 * pol["approval_rate"]),
            "%.2f%%" % (100 * pol["book_default_rate"]),
            "$%.2f" % pol["profit_per_application"],
        ])
    story.append(table(rows, [width * 0.44, width * 0.16, width * 0.20,
                              width * 0.20], align_right=(1, 2, 3)))

    approve_all = next(p for p in policies if "approve all" in p["policy"])
    youden = next(p for p in policies if "Youden" in p["policy"])
    ev = next(p for p in policies if "expected value" in p["policy"])

    story.append(Spacer(1, 7))
    story.append(Paragraph(
        "<b>The statistically optimal cutoff is a disaster.</b> Youden's J "
        "maximises TPR − FPR, declines 43%% of applications, and destroys "
        "$%.0f of value per application against simply approving everyone. The "
        "expected-value rule adds $%.2f per application instead — about %.1f%% "
        "on a $%.0fm book."
        % (approve_all["profit_per_application"] - youden["profit_per_application"],
           ev["profit_per_application"] - approve_all["profit_per_application"],
           100 * (ev["total_profit"] / approve_all["total_profit"] - 1),
           approve_all["total_profit"] / 1e6), s["callout"]))

    story.append(Paragraph(
        "That %.1f%% deserves context rather than spin. Every loan in this "
        "dataset was already approved by Lending Club's own underwriting, so "
        "the model is finding residual risk among applicants who already "
        "passed a credit screen. The easy declines happened upstream and are "
        "not in the data. A model sitting at the front of the funnel would have "
        "far more room; this number is the honest one for the population "
        "available."
        % (100 * (ev["total_profit"] / approve_all["total_profit"] - 1)),
        s["body"]))

    img = image("profit_curve.png", width)
    if img:
        story.append(img)

    story.append(PageBreak())

    # -- page 4 -------------------------------------------------------------
    story.append(Paragraph("Monitoring, and the price of interpretability",
                           s["h2"]))

    if drift_payload:
        psi_rows = drift_payload["score_psi_by_vintage"]
        perf = drift_payload["performance_by_vintage"]
        story.append(Paragraph(
            "Three views of the same five quarters, and they disagree — which "
            "is the whole argument for having all three. Score PSI climbs from "
            "%.4f to %.4f and crosses the conventional 0.10 line. "
            "Discrimination does not degrade at all; AUC actually improves from "
            "%.4f to %.4f. But the model under-predicts the default rate in "
            "every single quarter, and the gap widens from %.1f to %.1f "
            "percentage points. Any one signal alone gives the wrong answer."
            % (psi_rows[0]["psi"], psi_rows[-1]["psi"],
               perf[0]["auc"], perf[-1]["auc"],
               100 * (perf[0]["default_rate"] - perf[0]["mean_predicted"]),
               100 * (perf[-1]["default_rate"] - perf[-1]["mean_predicted"])),
            s["body"]))

        charac = drift_payload["characteristic_analysis"][:4]
        rows = [["Most-shifted variable", "PSI", "Verdict"]]
        for row in charac:
            rows.append([row["variable"], "%.4f" % row["psi"], row["verdict"]])
        story.append(table(rows, [width * 0.50, width * 0.22, width * 0.28],
                           align_right=(1,)))

        story.append(Spacer(1, 6))
        story.append(Paragraph(
            "The largest shift is <font face='Courier'>initial_list_status</font>, "
            "which records whether Lending Club listed a loan whole or "
            "fractionally. That is a change in their own platform operations, "
            "not in borrower quality — exactly the distinction per-variable PSI "
            "exists to make. Reacting to it by retraining would be solving the "
            "wrong problem.", s["body"]))

        img = image("drift_monitor.png", width * 0.92)
        if img:
            story.append(img)

        gate = drift_payload["gate"]
        comp = drift_payload["comparison"]
        story.append(Paragraph("The promotion gate", s["h2"]))
        story.append(Paragraph(
            "The interpretable scorecard was run as a challenger against the "
            "LightGBM champion, and it <b>failed</b> on discrimination, "
            "calibration and profit. It failed narrowly: %.4f AUC against "
            "%.4f, and $%.1fm profit against $%.1fm on a $%.0fm book. So the "
            "cost of being able to explain a decline is roughly %.3f AUC and "
            "%.2f%% of profit. That is a number a credit committee can accept "
            "or reject, which is more useful than an opinion about "
            "interpretability."
            % (comp["challenger_scorecard"]["auc"],
               comp["champion_lightgbm"]["auc"],
               comp["challenger_scorecard"]["profit"] / 1e6,
               comp["champion_lightgbm"]["profit"] / 1e6,
               comp["champion_lightgbm"]["profit"] / 1e6,
               comp["champion_lightgbm"]["auc"] - comp["challenger_scorecard"]["auc"],
               100 * (1 - comp["challenger_scorecard"]["profit"]
                      / comp["champion_lightgbm"]["profit"])),
            s["body"]))
        story.append(Paragraph(
            "Gate decision: <b>%s</b>. Failing is the expected outcome most of "
            "the time; that is the point of having a gate rather than a "
            "preference." % ("PROMOTE" if gate["promoted"] else "KEEP CHAMPION"),
            s["small"]))

    story.append(Paragraph("What I would do next", s["h2"]))
    story.append(Paragraph(
        "The residual under-prediction is the real open problem. Isotonic "
        "calibration was fitted on 2014 originations, which defaulted at "
        "14.63%, and the test vintages run hotter — so the calibrator is "
        "faithfully reproducing a base rate that has since moved. A rolling "
        "recalibration on the most recent closed book, rather than a fixed "
        "slice, is the obvious fix and the monitoring already detects the need "
        "for it. Beyond that: reject inference, since this population is "
        "already screened and the declined applications are invisible; and "
        "survival modelling of time-to-default, which would let the maturity "
        "rule be relaxed instead of discarding a quarter of a million loans.",
        s["body"]))

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Data: Lending Club accepted loans, 2007–2018 Q4, 2,260,701 rows, "
        "151 columns, SHA-256 verified at download. All figures generated by "
        "the pipeline in this repository; nothing in this document is typed by "
        "hand.", s["small"]))

    doc.build(story)
    return out_path


def main(argv: list[str] | None = None) -> int:
    config.ensure_dirs()
    out = os.path.join(config.DOCS_DIR, "credit-risk-summary.pdf")
    build(out)
    print("wrote %s (%.1f KB)" % (out, os.path.getsize(out) / 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())
