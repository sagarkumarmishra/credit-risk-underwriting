"""Tests for the pieces that are easy to get quietly wrong.

Weight of evidence, PSI and the expected-value arithmetic are all short enough
that a sign error or a swapped numerator survives code review and then produces
plausible-looking nonsense for months. Each one is checked against a case where
the right answer is known by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import cost, drift, scorecard

# ---------------------------------------------------------------------------
# weight of evidence
# ---------------------------------------------------------------------------

def test_woe_is_positive_for_a_safer_than_average_bin():
    """WOE is oriented towards good. A low-risk bin must score positive.

    Getting this backwards is the classic scorecard bug: everything still fits,
    the AUC is unchanged, and the points table reads exactly inverted.
    """
    binned = pd.Series(["safe"] * 100 + ["risky"] * 100)
    y = pd.Series([0] * 95 + [1] * 5 + [0] * 50 + [1] * 50)

    table = scorecard.woe_table(binned, y)
    safe = table[table["bin"] == "safe"].iloc[0]
    risky = table[table["bin"] == "risky"].iloc[0]

    assert safe["woe"] > 0
    assert risky["woe"] < 0
    assert safe["bad_rate"] < risky["bad_rate"]


def test_information_value_is_near_zero_when_a_variable_says_nothing():
    """An uninformative variable has the same default rate in every bin.

    Worth being careful constructing this. The obvious one-liner --
    `bins = ["a", "b"] * 500` against `y = [0, 1] * 500` -- looks uninformative
    and is in fact a perfect separator, because bin "a" lands on every good and
    bin "b" on every bad. That gives an IV of about 13.8, which is the right
    answer to the wrong question.
    """
    binned = pd.Series(["a"] * 500 + ["b"] * 500)
    y = pd.Series(([0] * 400 + [1] * 100) + ([0] * 400 + [1] * 100))

    table = scorecard.woe_table(binned, y)
    # Same 20% bad rate on both sides, so neither bin is evidence of anything.
    assert table["bad_rate"].nunique() == 1
    assert table["iv"].sum() < 0.01


def test_information_value_is_large_when_a_variable_separates_well():
    binned = pd.Series(["a"] * 500 + ["b"] * 500)
    y = pd.Series([0] * 480 + [1] * 20 + [0] * 100 + [1] * 400)
    table = scorecard.woe_table(binned, y)
    assert table["iv"].sum() > 0.3


def test_woe_survives_a_bin_with_no_defaults_at_all():
    """Without smoothing this is a division by zero and then a NaN model."""
    binned = pd.Series(["clean"] * 50 + ["mixed"] * 50)
    y = pd.Series([0] * 50 + [0] * 25 + [1] * 25)
    table = scorecard.woe_table(binned, y)
    assert np.isfinite(table["woe"]).all()
    assert np.isfinite(table["iv"]).all()


def test_missing_values_get_their_own_bin_rather_than_being_imputed():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan, np.nan])
    edges = scorecard.bin_numeric(x, max_bins=3)
    binned = scorecard.apply_bins(x, edges)
    assert (binned == scorecard.MISSING_LABEL).sum() == 2


def test_bins_have_open_ends_so_unseen_extremes_still_land_somewhere():
    train = pd.Series(np.arange(100, dtype=float))
    edges = scorecard.bin_numeric(train, max_bins=5)
    assert edges[0] == -np.inf
    assert edges[-1] == np.inf

    unseen = pd.Series([-5000.0, 99999.0])
    binned = scorecard.apply_bins(unseen, edges)
    assert (binned != scorecard.MISSING_LABEL).all()


def test_points_scaling_doubles_the_odds_every_pdo_points():
    card = scorecard.Scorecard(base_score=600, base_odds=10.0, pdo=20)
    assert card.factor == pytest.approx(20 / np.log(2))
    # A score of base_score must correspond to base_odds by construction.
    assert card.offset + card.factor * np.log(10.0) == pytest.approx(600)


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------

def test_psi_is_zero_when_nothing_moved():
    rng = np.random.default_rng(0)
    sample = rng.normal(size=20000)
    value, _ = drift.psi(sample, sample.copy())
    assert value == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_as_the_distribution_shifts():
    rng = np.random.default_rng(0)
    baseline = rng.normal(size=20000)
    small, _ = drift.psi(baseline, rng.normal(loc=0.1, size=20000))
    large, _ = drift.psi(baseline, rng.normal(loc=1.0, size=20000))
    assert small < large
    assert large > drift.PSI_SHIFTED


def test_psi_uses_baseline_edges_not_recomputed_ones():
    """The most common way PSI gets silently broken.

    If edges are re-derived on the new sample, every distribution looks stable
    because the bins move with the data. A large location shift must not report
    as stable.
    """
    rng = np.random.default_rng(1)
    baseline = rng.normal(size=20000)
    shifted = rng.normal(loc=3.0, size=20000)
    value, _ = drift.psi(baseline, shifted)
    assert drift.verdict(value) == "shifted"


def test_psi_reports_insufficient_data_rather_than_a_misleading_zero():
    baseline = np.array([np.nan] * 500)
    later = np.array([1.0, 2.0, 3.0])
    value, _ = drift.psi(baseline, later)
    assert not np.isfinite(value)
    assert drift.verdict(value) == "insufficient data"


def test_verdict_thresholds():
    assert drift.verdict(0.05) == "stable"
    assert drift.verdict(0.15) == "investigate"
    assert drift.verdict(0.40) == "shifted"


# ---------------------------------------------------------------------------
# the economics
# ---------------------------------------------------------------------------

def _loans():
    return pd.DataFrame({
        # cheap 36-month loan: little interest, so little risk tolerance
        # expensive 60-month loan: lots of interest, much more tolerance
        "funded_amnt": [10000.0, 10000.0],
        "installment": [310.0, 300.0],
        "term_months": [36, 60],
    })


def test_breakeven_probability_rises_with_the_interest_earned():
    econ = cost.loan_economics(_loans(), lgd=0.5)
    cheap, expensive = econ["breakeven_p"].iloc[0], econ["breakeven_p"].iloc[1]
    assert expensive > cheap, (
        "a loan earning more interest must tolerate more risk")


def test_breakeven_probability_falls_as_loss_given_default_rises():
    lenient = cost.loan_economics(_loans(), lgd=0.3)["breakeven_p"].iloc[0]
    harsh = cost.loan_economics(_loans(), lgd=0.9)["breakeven_p"].iloc[0]
    assert harsh < lenient


def test_expected_value_flips_sign_exactly_at_breakeven():
    econ = cost.loan_economics(_loans(), lgd=0.5)
    p_star = econ["breakeven_p"].to_numpy()

    at = cost.expected_value(p_star, econ)
    assert np.allclose(at, 0.0, atol=1e-6)

    below = cost.expected_value(p_star - 0.02, econ)
    above = cost.expected_value(p_star + 0.02, econ)
    assert (below > 0).all()
    assert (above < 0).all()


def test_realised_profit_counts_only_approved_loans():
    econ = cost.loan_economics(_loans(), lgd=0.5)
    y = np.array([0, 1])

    none_approved = cost.realised_profit(np.array([False, False]), y, econ)
    assert none_approved == 0.0

    good_only = cost.realised_profit(np.array([True, False]), y, econ)
    assert good_only == pytest.approx(econ["interest_if_paid"].iloc[0])

    bad_only = cost.realised_profit(np.array([False, True]), y, econ)
    assert bad_only == pytest.approx(-econ["loss_if_default"].iloc[1])


def test_zero_income_becomes_unknown_not_infinite():
    """A stated income of 0 is a data-entry artefact, not a real income."""
    from src import features

    frame = pd.DataFrame({
        "loan_amnt": [10000.0, 10000.0],
        "annual_inc": [50000.0, 0.0],
        "installment": [320.0, 320.0],
        "fico_range_low": [700.0, 700.0],
        "fico_range_high": [704.0, 704.0],
        "revol_bal": [5000.0, 5000.0],
        "total_rev_hi_lim": [20000.0, 0.0],
        "credit_history_months": [120.0, 120.0],
        "emp_length": ["5 years", "5 years"],
        "delinq_2yrs": [0.0, 0.0],
        "pub_rec": [0.0, 0.0],
        "pub_rec_bankruptcies": [0.0, 0.0],
        "tax_liens": [0.0, 0.0],
        "collections_12_mths_ex_med": [0.0, 0.0],
    })
    out = features.engineer(frame)

    assert np.isfinite(out["loan_to_income"].iloc[0])
    assert np.isnan(out["loan_to_income"].iloc[1])
    assert np.isnan(out["revol_util_calc"].iloc[1])


def test_unknown_credit_history_does_not_become_not_thin():
    from src import features

    frame = pd.DataFrame({
        "loan_amnt": [10000.0],
        "annual_inc": [50000.0],
        "installment": [320.0],
        "fico_range_low": [700.0],
        "fico_range_high": [704.0],
        "revol_bal": [5000.0],
        "total_rev_hi_lim": [20000.0],
        "credit_history_months": [np.nan],
        "emp_length": ["5 years"],
        "delinq_2yrs": [0.0],
        "pub_rec": [0.0],
        "pub_rec_bankruptcies": [0.0],
        "tax_liens": [0.0],
        "collections_12_mths_ex_med": [0.0],
    })
    out = features.engineer(frame)
    assert np.isnan(out["is_thin_file"].iloc[0])
