"""Figures for the README and the PDF.

Five charts, each one making a single point that is hard to make in prose:

  1. the leakage 2x2 -- what the two mistakes are worth in AUC
  2. single-feature AUC of the leaky columns -- one feature is the answer
  3. calibration, before and after -- ranking well is not the same as being right
  4. profit against cutoff -- where the AUC-optimal point sits versus the
     profit-optimal one
  5. PSI by vintage against AUC by vintage -- inputs and performance moving
     apart

Dark palette to match GitHub's default theme, because a white chart dropped
into a dark README looks like a screenshot someone forgot to retake.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src import config

BG = "#0d1117"
PANEL = "#161b22"
GRID = "#30363d"
TEXT = "#e6edf3"
MUTED = "#8b949e"
BLUE = "#58a6ff"
GREEN = "#3fb950"
RED = "#f85149"
AMBER = "#d29922"
PURPLE = "#bc8cff"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.dpi": 130,
    "savefig.facecolor": BG,
    "savefig.bbox": "tight",
})


def _save(fig, name: str) -> str:
    os.makedirs(config.ASSET_DIR, exist_ok=True)
    path = os.path.join(config.ASSET_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print("  %-34s %6.1f KB" % (name, os.path.getsize(path) / 1024))
    return path


def _load(name: str):
    path = os.path.join(config.REPORT_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# 1. the leakage 2x2
# ---------------------------------------------------------------------------

def chart_leakage_grid(leak: dict) -> str:
    cells = leak["cells"]
    dec = leak["decomposition"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    order = ["A_random_leaky", "B_oot_leaky", "C_random_honest", "D_oot_honest"]
    labels = ["A\nrandom split\n+ leaky cols",
              "B\nout-of-time\n+ leaky cols",
              "C\nrandom split\nhonest cols",
              "D\nout-of-time\nhonest cols"]
    aucs = [cells[k]["auc"] for k in order]
    colours = [RED, RED, BLUE, GREEN]

    bars = ax1.bar(labels, aucs, color=colours, width=0.62)
    for bar, value in zip(bars, aucs):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.012,
                 "%.4f" % value, ha="center", fontsize=11, fontweight="bold")

    ax1.axhline(0.5, color=MUTED, lw=1, ls=":")
    # Left edge, not the right: at the right it lands on top of bar D.
    ax1.text(-0.42, 0.515, "coin flip", color=MUTED, fontsize=8, ha="left")
    ax1.set_ylim(0.4, 1.09)
    ax1.set_ylabel("AUC on the test set")
    ax1.set_title("Same data, same model, four ways of evaluating it")
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_axisbelow(True)

    # Annotate the only cell that answers the underwriting question. The arrow
    # aims at the side of the bar rather than its top, which is where the value
    # label already sits.
    ax1.annotate("the only honest\nnumber here",
                 xy=(3.32, aucs[3] * 0.94), xytext=(2.55, 0.90),
                 color=GREEN, fontsize=9, ha="center",
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4,
                                 connectionstyle="arc3,rad=-0.25"))

    parts = ["leakage", "random split", "interaction"]
    values = [dec["attributable_to_leakage"],
              dec["attributable_to_random_split"],
              dec["interaction"]]
    cols = [RED, AMBER, PURPLE]
    bars = ax2.barh(parts, values, color=cols, height=0.5)
    for bar, value in zip(bars, values):
        offset = 0.006 if value >= 0 else -0.006
        ax2.text(value + offset, bar.get_y() + bar.get_height() / 2,
                 "%+.4f" % value, va="center",
                 ha="left" if value >= 0 else "right",
                 fontsize=10, fontweight="bold")

    ax2.axvline(0, color=MUTED, lw=1)
    pad = max(abs(min(values)), abs(max(values))) * 0.35
    ax2.set_xlim(min(values) - pad - 0.02, max(values) + pad + 0.02)
    ax2.set_xlabel("AUC attributable to each mistake")
    ax2.set_title("Decomposing the %+.4f overstatement" % dec["total_overstatement"])
    ax2.grid(axis="x", alpha=0.25)
    ax2.set_axisbelow(True)
    ax2.invert_yaxis()

    fig.suptitle("Post-origination columns are worth a third of an AUC. "
                 "The random split is worth almost nothing.",
                 fontsize=13, fontweight="bold", y=1.04)
    fig.text(0.5, 0.965,
             "Cell A scores a perfect 1.0000 and could not be deployed: "
             "in production those columns are empty.",
             ha="center", color=MUTED, fontsize=9)
    return _save(fig, "leakage_grid.png")


# ---------------------------------------------------------------------------
# 2. single-feature AUC
# ---------------------------------------------------------------------------

def chart_single_feature_auc(leak: dict) -> str:
    rows = list(leak["single_feature_aucs"])
    rows.sort(key=lambda r: r["auc_alone"])

    names = [r["column"] for r in rows]
    aucs = [r["auc_alone"] for r in rows]
    colours = [RED if a > 0.8 else (AMBER if a > 0.6 else MUTED) for a in aucs]

    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    bars = ax.barh(names, aucs, color=colours, height=0.62)
    for bar, value in zip(bars, aucs):
        ax.text(value + 0.006, bar.get_y() + bar.get_height() / 2,
                "%.4f" % value, va="center", fontsize=9, fontweight="bold")

    ax.axvline(0.5, color=MUTED, lw=1, ls=":")
    ax.text(0.505, -0.55, "no information", color=MUTED, fontsize=8)
    ax.set_xlim(0.45, 1.02)
    ax.set_xlabel("AUC using that one column alone")
    ax.set_title("A single post-origination column, on its own, beats the "
                 "entire honest model")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    honest = leak["cells"]["D_oot_honest"]["auc"]
    ax.axvline(honest, color=GREEN, lw=1.6, ls="--")
    # Sits in the empty space to the right of the two short bars at the bottom.
    # Placing it at the top of the axes puts it straight over the longest bar,
    # where it cannot be read.
    ax.text(honest + 0.009, 0.75,
            "the whole honest model\nscores %.4f" % honest, color=GREEN,
            fontsize=8.5, ha="left", va="center")

    fig.text(0.5, -0.04,
             "last_fico_range_* is the borrower's score at the most recent "
             "credit pull. On a defaulted loan that pull happened after the "
             "default.",
             ha="center", color=MUTED, fontsize=9)
    return _save(fig, "single_feature_auc.png")


# ---------------------------------------------------------------------------
# 3. calibration
# ---------------------------------------------------------------------------

def chart_calibration(metrics: dict) -> str:
    curves = metrics["calibration_curve"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    for ax, key, title, colour in (
        (ax1, "lightgbm_raw", "LightGBM, raw output", AMBER),
        (ax2, "lightgbm_calibrated", "LightGBM, isotonic calibration", GREEN),
    ):
        rows = curves[key]
        pred = [r["predicted"] for r in rows]
        obs = [r["observed"] for r in rows]
        top = max(max(pred), max(obs)) * 1.08

        ax.plot([0, top], [0, top], color=MUTED, ls=":", lw=1.2,
                label="perfect calibration")
        ax.plot(pred, obs, "o-", color=colour, lw=1.8, ms=5, label="observed")
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_xlabel("predicted default probability")
        ax.set_ylabel("observed default rate")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(facecolor=PANEL, edgecolor=GRID, fontsize=8.5, loc="upper left")

        res = metrics["models"][key]
        ax.text(0.97, 0.05,
                "Brier %.5f\nmean pred %.4f\nobserved  %.4f"
                % (res["brier"], res["mean_predicted"], res["observed_rate"]),
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8.5, color=MUTED, family="monospace")

    raw = metrics["models"]["lightgbm_raw"]
    cal = metrics["models"]["lightgbm_calibrated"]
    observed = cal["observed_rate"]
    gap_before = observed - raw["mean_predicted"]
    gap_after = observed - cal["mean_predicted"]
    closed = (gap_before - gap_after) / gap_before if gap_before else 0.0

    fig.suptitle("Isotonic calibration closed about %.0f%% of the pricing gap, "
                 "not all of it" % (100 * closed),
                 fontsize=13, fontweight="bold", y=1.02)
    fig.text(0.5, 0.945,
             "Mean predicted moved %.4f to %.4f against an observed %.4f. "
             "Both panels rank identically (AUC %.4f and %.4f) -- calibration "
             "changes the price, never the order."
             % (raw["mean_predicted"], cal["mean_predicted"], observed,
                raw["auc"], cal["auc"]),
             ha="center", color=MUTED, fontsize=9)
    fig.text(0.5, -0.04,
             "The residual under-prediction is not a bug in the calibrator: it "
             "was fitted on 2014 originations, which defaulted at 14.63%, and "
             "the test vintages run hotter than that.",
             ha="center", color=MUTED, fontsize=9)
    return _save(fig, "calibration.png")


# ---------------------------------------------------------------------------
# 4. profit against cutoff
# ---------------------------------------------------------------------------

def chart_profit_curve(metrics: dict, sweep: pd.DataFrame) -> str:
    """Profit against approval rate, plus a direct comparison of the policies.

    The first version of this put five annotations on one axes and they all
    landed on top of each other in the top-right corner, because every policy
    worth naming approves between 56% and 100% of applications. Two panels and a
    legend: the curve for shape, bars for the actual comparison.
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.5, 5),
        # Generous wspace: the right panel's policy names are long, and at the
        # default spacing they overhang into the left panel's plot area.
        gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.42})

    x = sweep["approval_rate"] * 100
    y = sweep["total_profit"] / 1e6

    ax1.plot(x, y, color=BLUE, lw=2.2, zorder=2)
    ax1.axhline(0, color=MUTED, lw=1, ls=":")
    ax1.grid(alpha=0.25)
    ax1.set_axisbelow(True)
    ax1.set_xlabel("share of applications approved (%)")
    ax1.set_ylabel("realised profit on the test book ($m)")
    ax1.set_title("Profit against approval rate")

    styles = {
        "approve all": (RED, "s", "approve everything (Lending Club)"),
        "0.50": (AMBER, "^", "cut at p > 0.50"),
        "Youden": (PURPLE, "v", "Youden J (statistically optimal)"),
        "expected value": (GREEN, "o", "per-loan expected value > 0"),
        "hindsight": (BLUE, "D", "best fixed cutoff (hindsight)"),
    }

    for pol in metrics["policies"]:
        for key, (colour, marker, label) in styles.items():
            if key in pol["policy"]:
                ax1.plot(pol["approval_rate"] * 100, pol["total_profit"] / 1e6,
                         marker, color=colour, ms=10, zorder=5,
                         markeredgecolor=BG, markeredgewidth=1.2, label=label)
                break

    ax1.legend(facecolor=PANEL, edgecolor=GRID, fontsize=8, loc="lower right",
               framealpha=0.95)

    # -- right panel: the comparison that actually matters ------------------
    pols = sorted(metrics["policies"], key=lambda p: p["profit_per_application"])
    names, values, colours, rates = [], [], [], []
    for pol in pols:
        colour = MUTED
        for key, (c, _m, _l) in styles.items():
            if key in pol["policy"]:
                colour = c
                break
        # Trim the parenthetical threshold; it is in the table, not needed here.
        label = pol["policy"].split(" (")[0].replace("model, ", "")
        names.append(label)
        values.append(pol["profit_per_application"])
        colours.append(colour)
        rates.append(100 * pol["approval_rate"])

    bars = ax2.barh(names, values, color=colours, height=0.6)
    lo, hi = min(values), max(values)
    span = hi - lo
    for bar, value, rate in zip(bars, values, rates):
        ax2.text(value + span * 0.03, bar.get_y() + bar.get_height() / 2,
                 "$%.2f    %.0f%% approved" % (value, rate),
                 va="center", fontsize=9, color=TEXT)

    ax2.set_xlim(lo - span * 0.12, hi + span * 0.62)
    ax2.set_xlabel("profit per application ($)")
    ax2.set_title("Profit per application, by policy")
    ax2.grid(axis="x", alpha=0.25)
    ax2.set_axisbelow(True)

    # Reference line at the do-nothing policy, so the bars read as a gain or
    # a loss against what Lending Club actually did.
    approve_all = next(p["profit_per_application"] for p in metrics["policies"]
                       if "approve all" in p["policy"])
    ax2.axvline(approve_all, color=RED, lw=1.3, ls="--", zorder=1)

    youden = next(p["profit_per_application"] for p in metrics["policies"]
                  if "Youden" in p["policy"])
    best_ev = next(p["profit_per_application"] for p in metrics["policies"]
                   if "expected value" in p["policy"])

    fig.suptitle("The statistically optimal cutoff destroys $%.0f of value per "
                 "application" % (approve_all - youden),
                 fontsize=13, fontweight="bold", y=1.03)
    fig.text(0.5, 0.955,
             "Youden J maximises TPR - FPR and takes no view on what a mistake "
             "costs. The expected-value rule adds $%.2f per application over "
             "approving everyone." % (best_ev - approve_all),
             ha="center", color=MUTED, fontsize=9)
    fig.text(0.5, -0.04,
             "Profit uses LGD measured from actual recoveries on charged-off "
             "loans, and scheduled interest as the upside.",
             ha="center", color=MUTED, fontsize=9)
    return _save(fig, "profit_curve.png")


# ---------------------------------------------------------------------------
# 5. drift
# ---------------------------------------------------------------------------

def chart_drift(drift_payload: dict) -> str:
    """Three stacked panels sharing an x-axis.

    The first attempt put AUC and default rate on twin y-axes in one panel. They
    crossed, which made it look as though discrimination and the outcome rate
    were converging on something. They are unrelated quantities on unrelated
    scales and a twin axis invites exactly that misreading, so each comparison
    now gets an axis where the units mean something.
    """
    psi_rows = pd.DataFrame(drift_payload["score_psi_by_vintage"])
    perf_rows = pd.DataFrame(drift_payload["performance_by_vintage"])
    periods = psi_rows["period"].astype(str).tolist()

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(11, 8.4), sharex=True,
        gridspec_kw={"hspace": 0.22, "height_ratios": [1, 0.85, 1]})

    # -- 1. PSI -------------------------------------------------------------
    colours = [GREEN if v == "stable" else (AMBER if v == "investigate" else RED)
               for v in psi_rows["verdict"]]
    bars = ax1.bar(periods, psi_rows["psi"], color=colours, width=0.55)
    for bar, value in zip(bars, psi_rows["psi"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.004,
                 "%.4f" % value, ha="center", fontsize=8.5, color=TEXT)

    ax1.axhline(0.10, color=AMBER, lw=1.2, ls="--")
    ax1.axhline(0.25, color=RED, lw=1.2, ls="--")
    ax1.text(-0.42, 0.105, "0.10  investigate", color=AMBER, fontsize=8)
    ax1.text(-0.42, 0.255, "0.25  shifted", color=RED, fontsize=8)
    ax1.set_ylim(0, 0.29)
    ax1.set_ylabel("score PSI")
    ax1.set_title("1. Inputs: population stability against the training baseline")
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_axisbelow(True)

    # -- 2. discrimination --------------------------------------------------
    ax2.plot(periods, perf_rows["auc"], "o-", color=BLUE, lw=2.2, ms=7)
    for xpos, value in zip(range(len(periods)), perf_rows["auc"]):
        ax2.text(xpos, value + 0.0012, "%.4f" % value, ha="center",
                 fontsize=8.5, color=TEXT)
    ax2.set_ylabel("AUC")
    lo, hi = perf_rows["auc"].min(), perf_rows["auc"].max()
    pad = max((hi - lo) * 0.45, 0.004)
    ax2.set_ylim(lo - pad, hi + pad)
    ax2.set_title("2. Discrimination: still fine, and drifting upwards if anything")
    ax2.grid(alpha=0.25)
    ax2.set_axisbelow(True)

    # -- 3. calibration -----------------------------------------------------
    observed = 100 * perf_rows["default_rate"]
    predicted = 100 * perf_rows["mean_predicted"]

    ax3.fill_between(periods, predicted, observed, color=RED, alpha=0.16,
                     zorder=1, label="under-prediction")
    ax3.plot(periods, observed, "s-", color=RED, lw=2, ms=6,
             label="observed default rate", zorder=3)
    ax3.plot(periods, predicted, "^--", color=AMBER, lw=2, ms=6,
             label="mean predicted", zorder=3)

    for xpos, (obs, pred) in enumerate(zip(observed, predicted)):
        ax3.text(xpos, (obs + pred) / 2, "%+.1f pp" % (pred - obs),
                 ha="center", va="center", fontsize=8, color=TEXT,
                 bbox=dict(facecolor=PANEL, edgecolor="none", pad=1.5, alpha=0.85))

    ax3.set_ylabel("default rate (%)")
    ax3.set_xlabel("origination vintage")
    ax3.set_title("3. Calibration: the model under-prices every quarter, "
                  "and the gap is not shrinking")
    ax3.grid(alpha=0.25)
    ax3.set_axisbelow(True)
    # Headroom above the highest line so the legend has somewhere to live that
    # is not on top of the data.
    lo3 = min(predicted.min(), observed.min())
    hi3 = max(predicted.max(), observed.max())
    ax3.set_ylim(lo3 - (hi3 - lo3) * 0.12, hi3 + (hi3 - lo3) * 0.38)
    ax3.legend(facecolor=PANEL, edgecolor=GRID, fontsize=8.5,
               loc="upper left", ncol=3, framealpha=0.95)

    fig.suptitle("PSI watches the inputs, AUC watches the ranking, "
                 "calibration watches the price. All three, or none.",
                 fontsize=12.5, fontweight="bold", y=0.975)
    fig.text(0.5, -0.02,
             "Isotonic calibration was fitted on 2014 originations, which "
             "defaulted at 14.63%. The test vintages run hotter, so the model "
             "under-predicts even after calibrating.",
             ha="center", color=MUTED, fontsize=9)
    return _save(fig, "drift_monitor.png")


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    config.ensure_dirs()
    print("writing charts to %s" % config.ASSET_DIR)

    leak = _load("leakage_experiment.json")
    metrics = _load("metrics.json")
    drift_payload = _load("drift.json")

    made = []
    if leak:
        made.append(chart_leakage_grid(leak))
        made.append(chart_single_feature_auc(leak))
    else:
        print("  no leakage_experiment.json, skipping 2 charts")

    if metrics:
        made.append(chart_calibration(metrics))
        sweep_path = os.path.join(config.REPORT_DIR, "profit_sweep.csv")
        if os.path.exists(sweep_path):
            made.append(chart_profit_curve(metrics, pd.read_csv(sweep_path)))
    else:
        print("  no metrics.json, skipping calibration and profit charts")

    if drift_payload:
        made.append(chart_drift(drift_payload))
    else:
        print("  no drift.json, skipping drift chart")

    print("%d charts written" % len(made))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
