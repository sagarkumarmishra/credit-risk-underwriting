"""Train the candidates, evaluate them honestly, and turn the winner into a policy.

Run order inside here:

  1. split the population by origination date -- train, then a calibration slice
     carved off the END of the training window, then the out-of-time test set
  2. fit three models: a constant baseline, a WOE scorecard, and LightGBM
  3. calibrate LightGBM, because a decision rule needs a probability, not a rank
  4. score every model on discrimination AND calibration, since a model can be
     excellent at one and useless at the other
  5. convert the best into an approve/decline policy and count the money

Point 1 matters and is easy to get wrong. The calibration slice is taken from
the last 12 months of the training window, not sampled at random from it. If you
calibrate on a random subset you are calibrating on the same period you trained
on, and the calibration looks far better than it will be in production.

Metrics are the ones credit risk actually reports. AUC and Gini for ranking, KS
because every scorecard review asks for it, Brier and log loss for calibration
quality. Accuracy is not reported: at a 20% default rate, predicting "everyone
repays" scores 80% and is worthless.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)

from src import build_warehouse, config, cost, features, scorecard

LGB_PARAMS = dict(
    objective="binary",
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=300,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=5.0,
    n_jobs=-1,
    random_state=config.SEED,
    verbose=-1,
)

CALIBRATION_MONTHS = 12


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    """Largest gap between the cumulative good and bad distributions.

    Reported as a percentage by convention in credit risk, where 30 to 45 is
    typical for an application scorecard.
    """
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def evaluate(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    auc = float(roc_auc_score(y, p))
    return {
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y, p),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
        "mean_predicted": float(p.mean()),
        "observed_rate": float(y.mean()),
        # How far the average prediction is from the truth. A model can rank
        # perfectly and still price every loan wrong, and this is the number
        # that catches it.
        "calibration_error_overall": float(p.mean() - y.mean()),
        "n": int(len(y)),
    }


def calibration_bins(y: np.ndarray, p: np.ndarray, bins: int = 20) -> list[dict]:
    """Observed vs predicted default rate by predicted-probability decile."""
    frame = pd.DataFrame({"y": y, "p": p})
    frame["bucket"] = pd.qcut(frame["p"], q=bins, duplicates="drop")
    rows = []
    for _bucket, chunk in frame.groupby("bucket", observed=True):
        rows.append({
            "predicted": float(chunk["p"].mean()),
            "observed": float(chunk["y"].mean()),
            "n": int(len(chunk)),
        })
    return sorted(rows, key=lambda r: r["predicted"])


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------

def three_way_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_all, test = features.split_out_of_time(df)
    issued = pd.to_datetime(train_all["issue_date"])
    cutoff = issued.max() - pd.DateOffset(months=CALIBRATION_MONTHS)
    fit = train_all[issued <= cutoff].reset_index(drop=True)
    calib = train_all[issued > cutoff].reset_index(drop=True)
    return fit, calib, test


def align_categories(reference: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """Force categorical levels to match the training matrix.

    LightGBM stores category *codes*, not labels. If the test matrix builds its
    own category order, code 3 means a different thing in each and predictions
    are quietly scrambled.
    """
    out = other.copy()
    for col in reference.columns:
        if str(reference[col].dtype) == "category":
            out[col] = pd.Categorical(out[col],
                                      categories=reference[col].cat.categories)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="sample this many loans, for faster iteration")
    ap.add_argument("--no-lc-grade", action="store_true",
                    help="drop LC's own grade, sub_grade and int_rate")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args(argv)

    config.ensure_dirs()
    use_lc = not args.no_lc_grade

    con = build_warehouse.connect(read_only=True)
    try:
        df = features.load_population(con, args.limit)
        lgd_stats = cost.estimate_lgd(con)
    finally:
        con.close()

    print("population: %d loans, %.2f%% charged off"
          % (len(df), 100 * df[config.TARGET].mean()))
    print("LGD from %d charged-off loans: mean %.3f, median %.3f (IQR %.3f-%.3f)"
          % (lgd_stats["n_charged_off"], lgd_stats["lgd_mean"],
             lgd_stats["lgd_median"], lgd_stats["lgd_q25"], lgd_stats["lgd_q75"]))

    fit_df, calib_df, test_df = three_way_split(df)
    print()
    print("splits, by origination date")
    for name, part in (("fit", fit_df), ("calibrate", calib_df), ("test", test_df)):
        issued = pd.to_datetime(part["issue_date"])
        print("  %-10s %8d loans   %s to %s   %5.2f%% default"
              % (name, len(part), issued.min().date(), issued.max().date(),
                 100 * part[config.TARGET].mean()))

    X_fit, y_fit, names = features.build_matrix(fit_df, use_lc)
    X_cal, y_cal, _ = features.build_matrix(calib_df, use_lc)
    X_test, y_test, _ = features.build_matrix(test_df, use_lc)
    X_cal = align_categories(X_fit, X_cal)
    X_test = align_categories(X_fit, X_test)

    print()
    print("features: %d%s" % (len(names), "" if use_lc else " (LC grade removed)"))

    results: dict[str, dict] = {}
    artefacts: dict[str, object] = {}

    # -- 1. baseline --------------------------------------------------------
    # Predict the training default rate for everyone. Ranks nothing, so AUC is
    # 0.5 by construction, but it is perfectly calibrated on average. It exists
    # so that every later Brier score has something to be compared against.
    base_rate = float(y_fit.mean())
    p_base = np.full(len(y_test), base_rate)
    results["baseline_constant"] = evaluate(y_test, p_base)
    print()
    print("baseline (constant %.4f):  AUC %.4f  Brier %.5f"
          % (base_rate, results["baseline_constant"]["auc"],
             results["baseline_constant"]["brier"]))

    # -- 2. WOE scorecard ---------------------------------------------------
    print()
    print("fitting WOE scorecard ...")
    started = time.time()
    card = scorecard.Scorecard(max_bins=10, min_iv=0.02)
    card.fit(X_fit, y_fit)
    p_card = card.predict_default_proba(X_test)
    results["scorecard_woe"] = evaluate(y_test, p_card)
    results["scorecard_woe"]["fit_seconds"] = round(time.time() - started, 1)
    results["scorecard_woe"]["n_variables"] = len(card.selected_)
    artefacts["scorecard"] = card
    print("  %d of %d variables cleared IV >= 0.02 in %.0fs"
          % (len(card.selected_), len(names), time.time() - started))
    print("  AUC %.4f  KS %.4f  Brier %.5f"
          % (results["scorecard_woe"]["auc"], results["scorecard_woe"]["ks"],
             results["scorecard_woe"]["brier"]))

    # -- 3. LightGBM, raw then calibrated -----------------------------------
    print()
    print("fitting LightGBM ...")
    started = time.time()
    gbm = lgb.LGBMClassifier(**LGB_PARAMS)
    # Early stopping watches the calibration slice, which is the 12 months
    # immediately after the fit window. That is the closest thing available to
    # "next year's applications", so the stopping point is chosen against the
    # kind of data the model will actually meet.
    gbm.fit(X_fit, y_fit, eval_X=X_cal, eval_y=y_cal,
            eval_metric="auc",
            callbacks=[lgb.early_stopping(50, verbose=False)])
    p_gbm = gbm.predict_proba(X_test)[:, 1]
    results["lightgbm_raw"] = evaluate(y_test, p_gbm)
    results["lightgbm_raw"]["fit_seconds"] = round(time.time() - started, 1)
    results["lightgbm_raw"]["best_iteration"] = int(gbm.best_iteration_ or
                                                    LGB_PARAMS["n_estimators"])
    print("  stopped at iteration %s, %.0fs"
          % (results["lightgbm_raw"]["best_iteration"], time.time() - started))
    print("  AUC %.4f  KS %.4f  Brier %.5f"
          % (results["lightgbm_raw"]["auc"], results["lightgbm_raw"]["ks"],
             results["lightgbm_raw"]["brier"]))

    # Isotonic on the held-out calibration slice. Isotonic rather than sigmoid
    # because there is plenty of data and no reason to assume the miscalibration
    # has a logistic shape.
    print()
    print("calibrating (isotonic, on the held-out %d-month slice) ..."
          % CALIBRATION_MONTHS)
    # FrozenEstimator rather than cv="prefit": sklearn removed the string form.
    # Freezing is what we want anyway -- refitting the booster inside the
    # calibrator would undo the out-of-time discipline of the split.
    calibrated = CalibratedClassifierCV(FrozenEstimator(gbm), method="isotonic")
    calibrated.fit(X_cal, y_cal)
    p_cal = calibrated.predict_proba(X_test)[:, 1]
    results["lightgbm_calibrated"] = evaluate(y_test, p_cal)
    artefacts["lightgbm"] = gbm
    artefacts["lightgbm_calibrated"] = calibrated
    print("  AUC %.4f  KS %.4f  Brier %.5f  mean predicted %.4f vs observed %.4f"
          % (results["lightgbm_calibrated"]["auc"],
             results["lightgbm_calibrated"]["ks"],
             results["lightgbm_calibrated"]["brier"],
             results["lightgbm_calibrated"]["mean_predicted"],
             results["lightgbm_calibrated"]["observed_rate"]))

    # -- 4. the policy ------------------------------------------------------
    lgd = lgd_stats["lgd_mean"]
    econ = cost.loan_economics(test_df, lgd)
    y_arr = y_test.to_numpy()

    policies = []
    # Lending Club approved every loan in this file, so approve-everything is
    # not a straw man, it is the policy that actually ran.
    policies.append(cost.policy_summary("approve all (what LC did)",
                                        np.ones(len(y_arr), dtype=bool),
                                        y_arr, econ))

    # The cutoff nearly every tutorial uses.
    policies.append(cost.policy_summary("model, cut at p > 0.50",
                                        p_cal <= 0.50, y_arr, econ))

    # Youden's J: the point that maximises TPR - FPR. Purely statistical, takes
    # no view on what a mistake costs.
    fpr, tpr, thr = roc_curve(y_arr, p_cal)
    j_thr = float(thr[int(np.argmax(tpr - fpr))])
    policies.append(cost.policy_summary("model, cut at Youden J (%.4f)" % j_thr,
                                        p_cal <= j_thr, y_arr, econ))

    # Per-loan break-even. No single threshold: each application is compared
    # against its own economics.
    policies.append(cost.policy_summary("model, per-loan expected value > 0",
                                        p_cal < econ["breakeven_p"].to_numpy(),
                                        y_arr, econ))

    # Best single cutoff in hindsight. An upper bound on what one fixed number
    # could have achieved, not something deployable.
    best_thr, _ = cost.best_global_threshold(p_cal, y_arr, econ)
    policies.append(cost.policy_summary(
        "model, best fixed cutoff in hindsight (%.4f)" % best_thr,
        p_cal <= best_thr, y_arr, econ))

    print()
    print("policy comparison on the out-of-time test set (%d applications)"
          % len(y_arr))
    print("  %-46s %8s %10s %14s %12s"
          % ("policy", "approve%", "book bad%", "total profit", "per appl."))
    for pol in policies:
        print("  %-46s %7.1f%% %9.2f%% %14s %12.2f"
              % (pol["policy"], 100 * pol["approval_rate"],
                 100 * pol["book_default_rate"],
                 "{:,.0f}".format(pol["total_profit"]),
                 pol["profit_per_application"]))

    sweep = cost.sweep_threshold(p_cal, y_arr, econ)

    # -- 5. persist ---------------------------------------------------------
    # Defaults for every feature, taken from the fitting window only. The API
    # needs these: asking a caller to supply all 105 bureau attributes to score
    # one application is not a usable interface, so unsupplied fields fall back
    # to the training median or modal value. Computing them from the fit split
    # rather than the whole population keeps the out-of-time discipline intact.
    defaults: dict[str, object] = {}
    for col in X_fit.columns:
        series = X_fit[col]
        if str(series.dtype) == "category":
            modes = series.mode(dropna=True)
            defaults[col] = (str(modes.iloc[0]) if len(modes) else None)
        else:
            numeric = pd.to_numeric(series, errors="coerce")
            # Several co-applicant columns are 100% null in the early vintages.
            # Calling median() on those emits a "Mean of empty slice" warning
            # from numpy for every one of them, which buries the real output,
            # so check first rather than filtering warnings globally.
            if numeric.notna().any():
                value = numeric.median()
                defaults[col] = (None if pd.isna(value) else float(value))
            else:
                defaults[col] = None

    model_path = os.path.join(config.MODEL_DIR, "model.pkl")
    with open(model_path, "wb") as fh:
        pickle.dump({
            "lightgbm_calibrated": calibrated,
            "scorecard": card,
            "feature_names": names,
            "categorical": [c for c in X_fit.columns
                            if str(X_fit[c].dtype) == "category"],
            "category_levels": {c: list(X_fit[c].cat.categories)
                                for c in X_fit.columns
                                if str(X_fit[c].dtype) == "category"},
            "feature_defaults": defaults,
            "lgd": lgd,
            "use_lc_grade": use_lc,
            "train_default_rate": base_rate,
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, fh)

    card.points_table().to_csv(
        os.path.join(config.REPORT_DIR, "scorecard_points.csv"), index=False)
    card.iv_summary().to_csv(
        os.path.join(config.REPORT_DIR, "information_value.csv"), index=False)
    sweep.to_csv(os.path.join(config.REPORT_DIR, "profit_sweep.csv"), index=False)

    np.save(os.path.join(config.REPORT_DIR, "test_probabilities.npy"), p_cal)
    np.save(os.path.join(config.REPORT_DIR, "test_outcomes.npy"), y_arr)

    metrics = {
        "population": {
            "n_total": int(len(df)),
            "n_fit": int(len(fit_df)),
            "n_calibrate": int(len(calib_df)),
            "n_test": int(len(test_df)),
            "default_rate_overall": float(df[config.TARGET].mean()),
            "train_vintage_end": config.TRAIN_VINTAGE_END,
            "test_vintage_start": config.TEST_VINTAGE_START,
        },
        "lgd": lgd_stats,
        "features": {"n": len(names), "use_lc_grade": use_lc, "names": names},
        "models": results,
        "calibration_curve": {
            "lightgbm_raw": calibration_bins(y_arr, p_gbm),
            "lightgbm_calibrated": calibration_bins(y_arr, p_cal),
            "scorecard_woe": calibration_bins(y_arr, p_card),
        },
        "policies": policies,
        "thresholds": {
            "youden_j": j_thr,
            "best_fixed_in_hindsight": best_thr,
            "breakeven_p_median": float(econ["breakeven_p"].median()),
            "breakeven_p_min": float(econ["breakeven_p"].min()),
            "breakeven_p_max": float(econ["breakeven_p"].max()),
            "breakeven_p_p05": float(econ["breakeven_p"].quantile(0.05)),
            "breakeven_p_p95": float(econ["breakeven_p"].quantile(0.95)),
            # A small number of rows have scheduled payments that do not exceed
            # principal, which makes interest_if_paid zero and the break-even
            # zero with it. Those loans can never clear an expected-value test.
            # Reporting the count is better than letting a 0.0% minimum sit in
            # the results looking like a bug.
            "n_zero_interest": int((econ["interest_if_paid"] <= 0).sum()),
            "share_zero_interest": float((econ["interest_if_paid"] <= 0).mean()),
        },
        "scorecard_top_iv": card.iv_summary().head(15).to_dict("records"),
    }
    metrics_path = os.path.join(config.REPORT_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=float)

    if not args.no_mlflow:
        _log_to_mlflow(results, policies, metrics, names)

    print()
    print("wrote %s" % model_path)
    print("wrote %s" % metrics_path)
    return 0


def _log_to_mlflow(results, policies, metrics, names) -> None:
    """Record the run. Kept in a function so a broken tracking store cannot
    take down a training run that otherwise succeeded."""
    try:
        import mlflow
    except ImportError:
        print("mlflow not installed, skipping tracking")
        return

    try:
        # SQLite rather than a bare ./mlruns directory. MLflow 3 put the plain
        # filesystem backend into maintenance mode and refuses to write to it,
        # so the file store silently stops recording runs. A local sqlite file
        # is still zero-setup and still committed nowhere.
        store = os.path.join(config.ROOT, "mlflow.db")
        mlflow.set_tracking_uri("sqlite:///" + store.replace("\\", "/"))
        mlflow.set_experiment("credit-risk-underwriting")
        with mlflow.start_run():
            mlflow.log_params({
                "n_features": len(names),
                "use_lc_grade": metrics["features"]["use_lc_grade"],
                "train_vintage_end": config.TRAIN_VINTAGE_END,
                "lgd_mean": round(metrics["lgd"]["lgd_mean"], 4),
                **{"lgb_" + k: v for k, v in LGB_PARAMS.items()
                   if isinstance(v, (int, float, str))},
            })
            for model_name, res in results.items():
                for key in ("auc", "gini", "ks", "brier", "log_loss",
                            "average_precision"):
                    if key in res:
                        mlflow.log_metric("%s__%s" % (model_name, key), res[key])
            for pol in policies:
                tag = (pol["policy"].split(",")[0].replace(" ", "_")
                       .replace("(", "").replace(")", ""))
                mlflow.log_metric("profit__" + tag, pol["profit_per_application"])
            mlflow.log_artifact(os.path.join(config.REPORT_DIR, "metrics.json"))
        print("logged run to mlflow.db")
    except Exception as exc:
        print("mlflow logging failed (%s), continuing" % str(exc)[:60])


if __name__ == "__main__":
    sys.exit(main())
