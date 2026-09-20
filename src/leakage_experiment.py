"""Measure what the two classic mistakes are actually worth, in AUC.

The claim this project is built on is that most public loan-default models are
measuring something other than credit risk. There are two separate mistakes and
they are usually made together, which makes it impossible to tell which one is
doing the damage. So this runs a 2x2:

                     honest columns      + post-origination columns
    random split           C                        A
    out-of-time split      D                        B

A is the careless configuration almost every tutorial uses. D is the only one
that answers the underwriting question. The difference A - D is the headline,
and splitting it into "how much was leakage" versus "how much was the split"
is the part that actually teaches you something.

Then, separately, the AUC of each leaky column on its own. If a single feature
scores 0.9 by itself, it is not a feature, it is the answer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import build_warehouse, columns, config, features

# Same hyperparameters in all four cells. If they differed, the comparison
# would be measuring my tuning rather than the mistakes.
PARAMS = dict(
    objective="binary",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=200,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=config.SEED,
    verbose=-1,
)

# The leaky columns used for cells A and B. Restricted to ones that need no
# date parsing, so the experiment tests leakage rather than my parsing.
LEAKY_USED = list(columns.WORST_OFFENDERS)
LEAKY_CATEGORICAL = ("debt_settlement_flag",)


def load(con, limit: int | None) -> pd.DataFrame:
    """Population plus the leaky columns, which we need here on purpose."""
    base = list(columns.MODELLABLE) + [
        config.TARGET, "issue_date", "vintage_year", "vintage_month",
        "term_months", "credit_history_months",
    ] + LEAKY_USED

    seen, cols = set(), []
    for c in base:
        if c not in seen:
            seen.add(c)
            cols.append(c)

    sql = "SELECT %s FROM loans" % ", ".join('"%s"' % c for c in cols)
    if limit:
        sql += " USING SAMPLE %d ROWS" % limit
    df = con.execute(sql).df()
    print("loaded %d rows, %d columns" % (len(df), df.shape[1]))
    return df


def matrix(df: pd.DataFrame, with_leakage: bool) -> tuple[pd.DataFrame, pd.Series]:
    eng = features.engineer(df)
    names = features.feature_names(eng, use_lc_judgement=True)

    if with_leakage:
        names = names + [c for c in LEAKY_USED if c in eng.columns]

    X = eng[names].copy()
    for col in list(features.CATEGORICAL) + list(LEAKY_CATEGORICAL):
        if col in X.columns:
            X[col] = X[col].astype("category")

    obj = [c for c in X.columns if X[c].dtype == object]
    if obj:
        raise TypeError("unencoded text columns: %s" % obj)

    return X, eng[config.TARGET].astype("int8")


def fit_and_score(train: pd.DataFrame, test: pd.DataFrame, with_leakage: bool) -> dict:
    Xtr, ytr = matrix(train, with_leakage)
    Xte, yte = matrix(test, with_leakage)

    # Categories must agree between the two matrices or LightGBM maps codes to
    # the wrong levels at predict time. Easy to miss, silently wrong.
    for col in Xtr.columns:
        if str(Xtr[col].dtype) == "category":
            levels = Xtr[col].cat.categories
            Xte[col] = pd.Categorical(Xte[col], categories=levels)

    started = time.time()
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]

    return {
        "auc": float(roc_auc_score(yte, p)),
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "n_features": int(Xtr.shape[1]),
        "default_rate_train": float(ytr.mean()),
        "default_rate_test": float(yte.mean()),
        "fit_seconds": round(time.time() - started, 1),
    }


def single_feature_aucs(df: pd.DataFrame) -> list[dict]:
    """AUC of each leaky column used alone.

    Direction is irrelevant here: a feature that predicts the outcome perfectly
    backwards is just as much a leak, so take max(auc, 1 - auc).
    """
    y = df[config.TARGET].astype("int8")
    rows = []
    for col in LEAKY_USED:
        if col not in df.columns:
            continue
        s = df[col]
        if s.dtype == object:
            s = s.astype("category").cat.codes.astype(float)
        s = pd.to_numeric(s, errors="coerce")
        mask = s.notna()
        if mask.sum() < 1000 or y[mask].nunique() < 2:
            continue
        auc = roc_auc_score(y[mask], s[mask])
        rows.append({
            "column": col,
            "auc_alone": float(max(auc, 1 - auc)),
            "non_null_share": float(mask.mean()),
            "reason": columns.CONTRACT[col][1],
        })
    rows.sort(key=lambda r: -r["auc_alone"])
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="sample this many loans (faster, for iteration)")
    args = ap.parse_args(argv)

    config.ensure_dirs()
    con = build_warehouse.connect(read_only=True)
    try:
        df = load(con, args.limit)
    finally:
        con.close()

    rnd_train, rnd_test = features.split_at_random(df)
    oot_train, oot_test = features.split_out_of_time(df)

    print()
    print("split sizes")
    print("  random       train %8d   test %8d" % (len(rnd_train), len(rnd_test)))
    print("  out-of-time  train %8d   test %8d" % (len(oot_train), len(oot_test)))

    cells = {
        "A_random_leaky": ("random split, post-origination columns included",
                           rnd_train, rnd_test, True),
        "B_oot_leaky": ("out-of-time split, post-origination columns included",
                        oot_train, oot_test, True),
        "C_random_honest": ("random split, origination columns only",
                            rnd_train, rnd_test, False),
        "D_oot_honest": ("out-of-time split, origination columns only",
                         oot_train, oot_test, False),
    }

    results = {}
    print()
    for key, (label, tr, te, leak) in cells.items():
        print("fitting %s ..." % key)
        res = fit_and_score(tr, te, leak)
        res["label"] = label
        res["leakage"] = leak
        res["split"] = "random" if key.split("_")[1] == "random" else "out_of_time"
        results[key] = res
        print("  AUC %.4f   (%d features, %.0fs)"
              % (res["auc"], res["n_features"], res["fit_seconds"]))

    a = results["A_random_leaky"]["auc"]
    b = results["B_oot_leaky"]["auc"]
    c = results["C_random_honest"]["auc"]
    d = results["D_oot_honest"]["auc"]

    decomposition = {
        "careless_auc": a,
        "honest_auc": d,
        "total_overstatement": a - d,
        # Holding the split fixed at out-of-time, what does leakage buy?
        "attributable_to_leakage": b - d,
        # Holding columns fixed at honest, what does the random split buy?
        "attributable_to_random_split": c - d,
        "interaction": (a - d) - (b - d) - (c - d),
    }

    print()
    print("where the overstatement comes from")
    print("  careless (A)                       %.4f" % a)
    print("  honest   (D)                       %.4f" % d)
    print("  total overstatement                %+.4f"
          % decomposition["total_overstatement"])
    print("  ... of which leakage               %+.4f"
          % decomposition["attributable_to_leakage"])
    print("  ... of which the random split      %+.4f"
          % decomposition["attributable_to_random_split"])
    print("  ... interaction                    %+.4f"
          % decomposition["interaction"])

    singles = single_feature_aucs(df)
    print()
    print("AUC of each post-origination column on its own")
    print("  %-28s %8s  %8s" % ("column", "AUC", "non-null"))
    for r in singles:
        print("  %-28s %8.4f  %7.1f%%"
              % (r["column"], r["auc_alone"], 100 * r["non_null_share"]))

    payload = {
        "cells": results,
        "decomposition": decomposition,
        "single_feature_aucs": singles,
        "params": PARAMS,
        "leaky_columns_used": LEAKY_USED,
        "n_loans": int(len(df)),
    }
    out = os.path.join(config.REPORT_DIR, "leakage_experiment.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print()
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
