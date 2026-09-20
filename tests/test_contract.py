"""Tests for the point-in-time contract.

These are the tests that matter most in this project. A model that quietly
trains on a post-origination column produces a beautiful number and a useless
model, and nothing else in the pipeline will complain. So the contract gets
tested directly, and the guard that enforces it gets tested for actually
failing when it should.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import columns


def test_every_column_has_a_category_and_a_reason():
    for name, (category, reason) in columns.CONTRACT.items():
        assert category in (columns.ORIGINATION, columns.OUTCOME, columns.LABEL,
                            columns.META, columns.DROP, columns.SENSITIVE), name
        assert reason and len(reason) > 10, (
            "%s has no real justification written for it" % name)


def test_contract_is_not_accidentally_empty():
    summary = columns.summary()
    assert summary[columns.ORIGINATION] > 80
    assert summary[columns.OUTCOME] > 20
    assert summary[columns.LABEL] == 1


def test_label_is_never_modellable():
    assert "loan_status" not in columns.MODELLABLE
    assert "issue_d" not in columns.MODELLABLE


def test_known_leaky_columns_are_classified_as_outcome():
    """The specific columns that ruin published loan-default models."""
    must_be_leaky = [
        "recoveries", "collection_recovery_fee", "total_pymnt",
        "total_rec_prncp", "total_rec_int", "out_prncp", "last_pymnt_d",
        "last_pymnt_amnt", "next_pymnt_d",
        # The subtle pair. These are the borrower's FICO at the most recent
        # credit pull, which for a defaulted loan happened after the default.
        "last_fico_range_high", "last_fico_range_low",
        "debt_settlement_flag", "settlement_amount", "hardship_flag",
    ]
    for name in must_be_leaky:
        assert name in columns.CONTRACT, "%s missing from the contract" % name
        assert columns.CONTRACT[name][0] == columns.OUTCOME, (
            "%s must be classified OUTCOME, it is %s"
            % (name, columns.CONTRACT[name][0]))
        assert name in columns.LEAKY


def test_bureau_columns_that_look_leaky_are_not():
    """Guard against over-correcting.

    chargeoff_within_12_mths sounds like an outcome and is not: it counts
    charge-offs on the borrower's OTHER accounts before this loan existed.
    Excluding it would throw away real signal for no reason.
    """
    for name in ("chargeoff_within_12_mths", "collections_12_mths_ex_med",
                 "delinq_2yrs", "num_accts_ever_120_pd",
                 "fico_range_low", "fico_range_high"):
        assert columns.CONTRACT[name][0] == columns.ORIGINATION, name
        assert name in columns.MODELLABLE


def test_geography_is_excluded_on_fair_lending_grounds():
    for name in ("zip_code", "addr_state"):
        assert columns.CONTRACT[name][0] == columns.SENSITIVE
        assert name not in columns.MODELLABLE


def test_assert_no_leakage_accepts_a_clean_feature_list():
    columns.assert_no_leakage(["loan_amnt", "dti", "fico_range_low", "grade"])


@pytest.mark.parametrize("bad", ["recoveries", "last_fico_range_high",
                                 "total_pymnt", "out_prncp"])
def test_assert_no_leakage_rejects_a_dirty_feature_list(bad):
    """The guard has to actually fail. A check that never fires is decoration."""
    with pytest.raises(AssertionError) as exc:
        columns.assert_no_leakage(["loan_amnt", "dti", bad])
    assert bad in str(exc.value)


def test_check_coverage_rejects_an_unreviewed_column():
    source = list(columns.CONTRACT) + ["some_new_column_lending_club_added"]
    with pytest.raises(AssertionError) as exc:
        columns.check_coverage(source)
    assert "some_new_column_lending_club_added" in str(exc.value)


def test_check_coverage_rejects_a_disappeared_column():
    source = [c for c in columns.CONTRACT if c != "loan_amnt"]
    with pytest.raises(AssertionError) as exc:
        columns.check_coverage(source)
    assert "loan_amnt" in str(exc.value)


def test_check_coverage_passes_on_the_exact_contract():
    columns.check_coverage(list(columns.CONTRACT))


def test_feature_names_refuses_to_pass_through_a_leaky_column():
    """End-to-end: the feature builder must trip the guard, not just define it."""
    from src import features

    frame = pd.DataFrame({
        "loan_amnt": [10000.0],
        "annual_inc": [50000.0],
        "installment": [320.0],
        "fico_range_low": [700.0],
        "fico_range_high": [704.0],
        "revol_bal": [5000.0],
        "total_rev_hi_lim": [20000.0],
        "credit_history_months": [120.0],
        "emp_length": ["5 years"],
        "delinq_2yrs": [0.0],
        "pub_rec": [0.0],
        "pub_rec_bankruptcies": [0.0],
        "tax_liens": [0.0],
        "collections_12_mths_ex_med": [0.0],
        # The intruder.
        "recoveries": [0.0],
    })
    engineered = features.engineer(frame)
    names = features.feature_names(engineered)
    assert "recoveries" not in names

    with pytest.raises(AssertionError):
        columns.assert_no_leakage(list(engineered.columns))
