"""Run the monitoring checks and decide whether a challenger may be promoted.

Three outputs:

  score PSI by vintage        is the population still the one we trained on?
  characteristic analysis     if not, which variables moved?
  performance by vintage      is discrimination holding up regardless?

Then a promotion gate. This is the part that makes the difference between "I
trained a model" and "I would let this replace the one in production". The gate
is deliberately strict and deliberately arbitrary in places -- a real one is
signed off by a committee, not derived -- so every threshold says why it is
where it is.

A challenger has to clear all of:

  1. AUC no worse than champion minus a small tolerance. Not "better": a
     simpler or cheaper model that matches is worth promoting.
  2. Calibration no worse. A model that ranks the same but misprices is not an
     improvement, it is a repricing bug.
  3. Profit at least as high on the out-of-time book.
  4. No input variable in a shifted state, or if there is, it must be flagged
     rather than silently accepted.

Failing the gate is the expected outcome most of the time. That is the point of
having one.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

from src import build_warehouse, config, cost, drift, features

# Tolerances for the promotion gate. Stated here, in one place, so a reviewer
# can disagree with a number rather than hunt for it.
AUC_TOLERANCE = 0.002        # within a fifth of a Gini point counts as equal
BRIER_TOLERANCE = 0.0005     # calibration must not degrade materially
PROFIT_TOLERANCE = 0.0       # profit is not allowed to go backwards at all


def load_model(path: str) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def align(reference_levels: dict, X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for col, levels in reference_levels.items():
        if col in out.columns:
            out[col] = pd.Categorical(out[col], categories=levels)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=os.path.join(config.MODEL_DIR, "model.pkl"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    if not os.path.exists(args.model):
        raise SystemExit("no model at %s -- run `python -m src.train` first"
                         % args.model)

    config.ensure_dirs()
    bundle = load_model(args.model)
    champion = bundle["lightgbm_calibrated"]
    card = bundle["scorecard"]
    levels = bundle["category_levels"]
    lgd = bundle["lgd"]
    use_lc = bundle["use_lc_grade"]

    con = build_warehouse.connect(read_only=True)
    try:
        df = features.load_population(con, args.limit)
    finally:
        con.close()

    train_df, test_df = features.split_out_of_time(df)
    X_train, y_train, names = features.build_matrix(train_df, use_lc)
    X_test, y_test, _ = features.build_matrix(test_df, use_lc)
    X_train = align(levels, X_train)
    X_test = align(levels, X_test)

    print("baseline (training vintages): %d loans" % len(X_train))
    print("later    (test vintages):     %d loans" % len(X_test))

    p_train = champion.predict_proba(X_train)[:, 1]
    p_test = champion.predict_proba(X_test)[:, 1]

    # -- score PSI by vintage ------------------------------------------------
    later = test_df.copy()
    later["score"] = p_test
    later["vintage"] = pd.to_datetime(later["issue_date"]).dt.to_period("Q").astype(str)

    psi_by_vintage = drift.score_psi_by_period(p_train, later, "score", "vintage")
    print()
    print("score PSI by vintage quarter, against the training distribution")
    print("  %-10s %8s %8s  %s" % ("vintage", "n", "PSI", "verdict"))
    for row in psi_by_vintage.itertuples():
        print("  %-10s %8d %8.4f  %s" % (row.period, row.n, row.psi, row.verdict))

    # -- characteristic analysis --------------------------------------------
    charac = drift.characteristic_analysis(X_train, X_test, names)
    print()
    print("most-shifted input variables (top 12 by PSI)")
    print("  %-34s %10s  %s" % ("variable", "PSI", "verdict"))
    for row in charac.head(12).itertuples():
        print("  %-34s %10.4f  %s" % (row.variable, row.psi, row.verdict))

    n_shifted = int((charac["verdict"] == "shifted").sum())
    n_investigate = int((charac["verdict"] == "investigate").sum())
    print()
    print("  %d of %d variables shifted, %d worth investigating"
          % (n_shifted, len(charac), n_investigate))

    # -- performance by vintage ---------------------------------------------
    perf = drift.performance_by_period(y_test.to_numpy(), p_test,
                                       later["vintage"].to_numpy())
    print()
    print("performance by vintage quarter")
    print("  %-10s %8s %10s %10s %8s"
          % ("vintage", "n", "default%", "predicted%", "AUC"))
    for row in perf.itertuples():
        print("  %-10s %8d %9.2f%% %9.2f%% %8.4f"
              % (row.period, row.n, 100 * row.default_rate,
                 100 * row.mean_predicted, row.auc))

    # -- promotion gate: scorecard as the challenger -------------------------
    # The scorecard is the interesting challenger precisely because it is the
    # one a credit team would rather deploy. If it clears the gate, the
    # gradient booster has to justify its existence.
    from sklearn.metrics import brier_score_loss, roc_auc_score

    p_card = card.predict_default_proba(X_test)
    econ = cost.loan_economics(test_df, lgd)
    y_arr = y_test.to_numpy()

    def profit_of(p: np.ndarray) -> float:
        approved = p < econ["breakeven_p"].to_numpy()
        return cost.realised_profit(approved, y_arr, econ)

    comparison = {
        "champion_lightgbm": {
            "auc": float(roc_auc_score(y_arr, p_test)),
            "brier": float(brier_score_loss(y_arr, p_test)),
            "profit": profit_of(p_test),
        },
        "challenger_scorecard": {
            "auc": float(roc_auc_score(y_arr, p_card)),
            "brier": float(brier_score_loss(y_arr, p_card)),
            "profit": profit_of(p_card),
        },
    }

    champ = comparison["champion_lightgbm"]
    chall = comparison["challenger_scorecard"]

    checks = [
        ("discrimination", chall["auc"] >= champ["auc"] - AUC_TOLERANCE,
         "AUC %.4f vs %.4f (tolerance %.4f)"
         % (chall["auc"], champ["auc"], AUC_TOLERANCE)),
        ("calibration", chall["brier"] <= champ["brier"] + BRIER_TOLERANCE,
         "Brier %.5f vs %.5f (tolerance %.5f)"
         % (chall["brier"], champ["brier"], BRIER_TOLERANCE)),
        ("profit", chall["profit"] >= champ["profit"] - PROFIT_TOLERANCE,
         "profit %.0f vs %.0f" % (chall["profit"], champ["profit"])),
        ("population stability",
         not (psi_by_vintage["verdict"] == "shifted").any(),
         "%d vintage(s) flagged as shifted"
         % int((psi_by_vintage["verdict"] == "shifted").sum())),
    ]

    print()
    print("promotion gate: WOE scorecard as challenger to the LightGBM champion")
    for name, passed, detail in checks:
        print("  [%s] %-22s %s" % ("PASS" if passed else "FAIL", name, detail))

    promoted = all(passed for _, passed, _ in checks)
    print()
    print("  decision: %s" % ("PROMOTE" if promoted else "KEEP CHAMPION"))
    if not promoted:
        failed = [n for n, passed, _ in checks if not passed]
        print("  blocked by: %s" % ", ".join(failed))

    payload = {
        "score_psi_by_vintage": psi_by_vintage.to_dict("records"),
        "characteristic_analysis": charac.to_dict("records"),
        "performance_by_vintage": perf.to_dict("records"),
        "shifted_count": n_shifted,
        "investigate_count": n_investigate,
        "comparison": comparison,
        "gate": {
            "checks": [{"name": n, "passed": bool(p), "detail": d}
                       for n, p, d in checks],
            "promoted": bool(promoted),
            "tolerances": {
                "auc": AUC_TOLERANCE,
                "brier": BRIER_TOLERANCE,
                "profit": PROFIT_TOLERANCE,
            },
        },
    }
    out = os.path.join(config.REPORT_DIR, "drift.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print()
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
