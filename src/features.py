"""Turn the loan population into a model-ready matrix.

Three jobs, in order:

  1. pull only the columns the point-in-time contract permits
  2. parse the handful of fields Lending Club stores as awkward text
  3. add the few derived features that encode something a raw column does not

There is deliberately not much of step 3. Dozens of hand-crafted ratios is how
you overfit a credit model and how you make it impossible to explain to a
credit officer. Each derived feature below has a one-line reason for existing,
and if I could not write that line I left it out.

The leakage guard runs at the end of build_matrix(), not as a comment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import columns, config

# ---------------------------------------------------------------------------
# Columns that need parsing rather than modelling
# ---------------------------------------------------------------------------

# Lending Club's own grade, sub-grade and priced interest rate. These are
# legitimate at origination, but they are also the output of LC's internal
# scoring model. Train on them and part of what you learn is "agree with
# Lending Club", which flatters the metrics and teaches you nothing about the
# borrower. train.py can drop them with --no-lc-grade so the cost is measured
# rather than assumed.
LC_JUDGEMENT = ("grade", "sub_grade", "int_rate")

# Free-text or coded fields that need converting before use.
CATEGORICAL = (
    "grade",
    "sub_grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "initial_list_status",
    "application_type",
    "disbursement_method",
    "verification_status_joint",
)

# Everything ORIGINATION that is not categorical and not one of the raw date
# strings we replaced with a derived numeric.
RAW_DATE_STRINGS = ("earliest_cr_line", "sec_app_earliest_cr_line")

EMP_LENGTH_MAP = {
    "< 1 year": 0.5,
    "1 year": 1.0,
    "2 years": 2.0,
    "3 years": 3.0,
    "4 years": 4.0,
    "5 years": 5.0,
    "6 years": 6.0,
    "7 years": 7.0,
    "8 years": 8.0,
    "9 years": 9.0,
    "10+ years": 10.0,
}


def load_population(con, limit: int | None = None) -> pd.DataFrame:
    """Read the modelling population out of DuckDB.

    Selects the contracted ORIGINATION columns plus the few META/derived ones
    the split and the parsing need. The leaky columns stay in the warehouse and
    are simply not asked for -- except by leakage_experiment.py, which asks on
    purpose.
    """
    wanted = list(columns.MODELLABLE) + [
        config.TARGET,
        "issue_date",
        "vintage_year",
        "vintage_month",
        "term_months",
        "credit_history_months",
        "emp_length",
    ]
    # emp_length is already in MODELLABLE; de-duplicate while keeping order.
    seen, cols = set(), []
    for c in wanted:
        if c not in seen:
            seen.add(c)
            cols.append(c)

    select = ", ".join('"%s"' % c for c in cols)
    sql = "SELECT %s FROM loans" % select
    if limit:
        sql += " USING SAMPLE %d ROWS" % limit
    return con.execute(sql).df()


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide, treating non-positive denominators as unknown rather than zero.

    A stated income of 0 is a data-entry artefact, not a real income. Mapping
    it to inf would let a single bad row dominate any tree split, and mapping
    it to 0 would assert the borrower's loan is trivially affordable. NaN says
    what is actually true: we do not know.
    """
    denom = denominator.where(denominator > 0)
    return numerator / denom


def _to_plain_float(out: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas nullable extension dtypes to plain float64.

    DuckDB hands back Int64/boolean extension arrays wherever a column contains
    nulls. Those propagate pd.NA through every subsequent operation, and pd.NA
    cannot be cast to a fixed-width int -- so a perfectly ordinary
    `(x < 36).astype("int8")` blows up several steps later with a message that
    names neither the column nor the reason.

    Rather than sprinkle fillna() around and quietly invent values, everything
    numeric becomes float64 up front, where missing is np.nan. LightGBM treats
    nan as its own branch at every split, which is what we want: "unknown" is
    genuinely informative in credit data and should not be imputed away.
    """
    for col in out.columns:
        dtype = str(out[col].dtype)
        if dtype.startswith(("Int", "UInt", "Float", "boolean")):
            out[col] = out[col].astype("float64")
    return out


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Parse awkward fields and add derived features."""
    out = _to_plain_float(df.copy())

    # -- parsing -----------------------------------------------------------
    # emp_length is an ordered category stored as prose. Ordinal, not one-hot:
    # the ordering is real and a tree can use it in one split.
    out["emp_length_years"] = out["emp_length"].map(EMP_LENGTH_MAP)

    # 'term' was already parsed to term_months in the warehouse.

    # -- derived -----------------------------------------------------------
    # FICO arrives as a 5-point band. The midpoint is the usable number; the
    # width is constant so it carries nothing.
    out["fico"] = (out["fico_range_low"] + out["fico_range_high"]) / 2.0

    # How big is the loan relative to what the borrower earns. dti already
    # covers existing debt but not this new obligation.
    out["loan_to_income"] = _safe_ratio(out["loan_amnt"], out["annual_inc"])

    # Annual cost of this loan as a share of income. Closer to what an
    # underwriter actually asks: can they make the payment.
    out["installment_to_income"] = _safe_ratio(out["installment"] * 12.0,
                                               out["annual_inc"])

    # Revolving balance against total revolving limit. revol_util is LC's own
    # version of this and is sometimes null, so a recomputed one fills gaps.
    out["revol_util_calc"] = 100.0 * _safe_ratio(out["revol_bal"],
                                                 out["total_rev_hi_lim"])

    # Thin-file flag. A borrower with two years of history is a different
    # proposition from one with twenty, and the model should be able to say so
    # without having to discover the threshold. Where history length is unknown
    # the flag is nan rather than 0 -- claiming "not thin" would be a guess.
    history = out["credit_history_months"]
    out["is_thin_file"] = np.where(history.isna(), np.nan,
                                   (history < 36).astype(float))

    # Any derogatory mark at all. The individual counts are sparse and
    # zero-inflated; a single "has anything gone wrong before" flag is often
    # what survives into a scorecard. A null in these particular counters does
    # mean zero -- the bureau reports no such records rather than declining to
    # say -- so filling with 0 here is a statement about the data, not a shrug.
    derog = ["delinq_2yrs", "pub_rec", "pub_rec_bankruptcies", "tax_liens",
             "collections_12_mths_ex_med"]
    present = [c for c in derog if c in out.columns]
    totals = out[present].fillna(0.0).sum(axis=1).to_numpy(dtype=float)
    out["has_derogatory"] = (totals > 0).astype(float)

    return out


DERIVED = (
    "emp_length_years",
    "fico",
    "loan_to_income",
    "installment_to_income",
    "revol_util_calc",
    "is_thin_file",
    "has_derogatory",
)

# Replaced by a derived numeric, so the original is no longer a feature.
SUPERSEDED = ("emp_length", "fico_range_low", "fico_range_high", "term")


def feature_names(df: pd.DataFrame, use_lc_judgement: bool = True) -> list[str]:
    """The exact column list a model is allowed to train on."""
    allowed = set(columns.MODELLABLE) | set(DERIVED) | {"term_months",
                                                        "credit_history_months"}
    drop = set(SUPERSEDED) | set(RAW_DATE_STRINGS)
    if not use_lc_judgement:
        drop |= set(LC_JUDGEMENT)

    names = [c for c in df.columns if c in allowed and c not in drop]

    # The guard rail. If an OUTCOME column ever reaches this point, stop.
    columns.assert_no_leakage(names)
    return names


def build_matrix(df: pd.DataFrame, use_lc_judgement: bool = True
                 ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return (X, y, feature_names) ready for LightGBM.

    Categoricals are left as pandas `category` dtype rather than one-hot
    encoded. LightGBM splits on them natively, and one-hot encoding
    `sub_grade` into 35 sparse columns makes the trees worse, not better.
    """
    eng = engineer(df)
    names = feature_names(eng, use_lc_judgement)

    X = eng[names].copy()
    for col in CATEGORICAL:
        if col in X.columns:
            X[col] = X[col].astype("category")

    # Anything still object-typed would silently break LightGBM.
    leftover = [c for c in X.columns if X[c].dtype == object]
    if leftover:
        raise TypeError(
            "these columns are still text and have no encoding: %s" % leftover
        )

    y = eng[config.TARGET].astype("int8")
    return X, y, names


def split_out_of_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on origination date, never at random.

    A random split lets the model see 2018 loans while predicting 2013 ones.
    Real deployment is always forwards in time, so the evaluation has to be
    too. This is the single change that moves reported AUC the most on this
    dataset after removing leakage.
    """
    issued = pd.to_datetime(df["issue_date"])
    train = df[issued <= pd.Timestamp(config.TRAIN_VINTAGE_END)]
    test = df[issued >= pd.Timestamp(config.TEST_VINTAGE_START)]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def split_at_random(df: pd.DataFrame, test_size: float = 0.25
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The split I am arguing against, kept so the gap can be measured."""
    rng = np.random.default_rng(config.SEED)
    mask = rng.random(len(df)) < test_size
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)
