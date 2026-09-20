"""API contract tests.

Skipped rather than failed when no trained model is present, so the suite still
runs on a clean checkout and in CI, where the 1.56 GB download is not available.
The contract itself is what is being tested: a caller should be able to send a
handful of fields and get a decision, not a stack trace.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from src import config, serve

MODEL_PRESENT = os.path.exists(os.path.join(config.MODEL_DIR, "model.pkl"))
needs_model = pytest.mark.skipif(
    not MODEL_PRESENT,
    reason="no trained model on disk; run `make all` to exercise these")


@pytest.fixture(scope="module")
def client():
    return TestClient(serve.app)


MINIMAL = {"loan_amnt": 15000, "annual_inc": 65000}

TYPICAL = {
    "loan_amnt": 15000,
    "term_months": 36,
    "int_rate": 12.5,
    "annual_inc": 65000,
    "dti": 18.2,
    "fico_range_low": 690,
    "fico_range_high": 694,
    "emp_length": "5 years",
    "home_ownership": "MORTGAGE",
    "purpose": "debt_consolidation",
    "grade": "C",
    "revol_util": 45.0,
    "open_acc": 9,
    "credit_history_months": 168,
}


def test_amount_must_be_positive(client):
    bad = dict(TYPICAL, loan_amnt=-1)
    assert client.post("/score", json=bad).status_code == 422


def test_income_must_be_positive(client):
    bad = dict(TYPICAL, annual_inc=0)
    assert client.post("/score", json=bad).status_code == 422


def test_fico_outside_the_real_range_is_rejected(client):
    bad = dict(TYPICAL, fico_range_low=120)
    assert client.post("/score", json=bad).status_code == 422


def test_missing_required_field_is_rejected(client):
    assert client.post("/score", json={"loan_amnt": 15000}).status_code == 422


@needs_model
def test_health_reports_a_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["n_features"] > 50
    assert 0 < body["lgd_assumption"] < 1


@needs_model
def test_a_minimal_request_still_produces_a_decision(client):
    """Two fields in, a full decision out, with the defaulting declared."""
    response = client.post("/score", json=MINIMAL)
    assert response.status_code == 200
    body = response.json()

    assert 0.0 <= body["probability_of_default"] <= 1.0
    assert body["recommendation"] in ("approve", "decline")
    assert body["fields_defaulted"] > body["fields_supplied"], (
        "a two-field request must admit that most inputs were defaulted")


@needs_model
def test_a_typical_request_supplies_more_and_defaults_less(client):
    minimal = client.post("/score", json=MINIMAL).json()
    typical = client.post("/score", json=TYPICAL).json()
    assert typical["fields_supplied"] > minimal["fields_supplied"]
    assert typical["fields_defaulted"] < minimal["fields_defaulted"]


@needs_model
def test_recommendation_agrees_with_the_expected_value(client):
    body = client.post("/score", json=TYPICAL).json()
    if body["recommendation"] == "approve":
        assert body["expected_value"] > 0
        assert body["probability_of_default"] < body["breakeven_probability"]
    else:
        assert body["expected_value"] <= 0
        assert body["probability_of_default"] >= body["breakeven_probability"]


@needs_model
def test_a_worse_credit_file_scores_worse(client):
    """Monotonicity where it is not negotiable.

    A model is allowed to be surprising about interactions. It is not allowed to
    think a 640 FICO with recent delinquencies is safer than a 790 with none.
    """
    strong = client.post("/score", json=dict(
        TYPICAL, fico_range_low=790, fico_range_high=794, delinq_2yrs=0,
        inq_last_6mths=0, dti=8.0, grade="A")).json()
    weak = client.post("/score", json=dict(
        TYPICAL, fico_range_low=640, fico_range_high=644, delinq_2yrs=3,
        inq_last_6mths=5, dti=34.0, grade="F")).json()

    assert weak["probability_of_default"] > strong["probability_of_default"]
    assert weak["scorecard_points"] < strong["scorecard_points"]


@needs_model
def test_a_longer_dearer_loan_tolerates_more_risk(client):
    """The break-even probability is a property of the loan, not a constant."""
    cheap = client.post("/score", json=dict(
        TYPICAL, term_months=36, int_rate=7.0)).json()
    dear = client.post("/score", json=dict(
        TYPICAL, term_months=60, int_rate=26.0)).json()
    assert dear["breakeven_probability"] > cheap["breakeven_probability"]


@needs_model
def test_adverse_action_reasons_are_returned_for_a_weak_applicant(client):
    body = client.post("/score", json=dict(
        TYPICAL, fico_range_low=640, fico_range_high=644, delinq_2yrs=3,
        dti=34.0, grade="F")).json()
    reasons = body["adverse_action_reasons"]
    assert reasons, "a declined-looking applicant must come with stated reasons"
    for reason in reasons:
        assert reason["points_lost"] > 0
        assert reason["variable"]
    # Ordered worst-first, so a letter can quote the top few.
    losses = [r["points_lost"] for r in reasons]
    assert losses == sorted(losses, reverse=True)


@needs_model
def test_no_post_origination_field_is_accepted_as_an_input(client):
    """The API must not become a back door around the contract.

    `extra` is a free-form dict, so it is the one place a leaky column could
    sneak in. Sending one must not change the score.
    """
    from src import columns

    baseline = client.post("/score", json=TYPICAL).json()
    smuggled = client.post("/score", json=dict(
        TYPICAL, extra={"recoveries": 5000.0, "last_fico_range_high": 500.0})
    ).json()

    assert (smuggled["probability_of_default"]
            == baseline["probability_of_default"]), (
        "a post-origination value changed the prediction; the contract leaked")

    for name in ("recoveries", "last_fico_range_high"):
        assert name in columns.LEAKY
