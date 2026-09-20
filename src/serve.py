"""Scoring API.

Two endpoints, which is all an underwriting service needs:

    GET  /health    is the model loaded, and which one
    POST /score     one application in, a decision out

The decision is the point. Returning a bare probability pushes the hard part
back onto the caller, who then invents a threshold. This returns the
probability, the loan's own break-even probability, the expected value in
dollars, a recommendation, and -- when the scorecard is used -- the specific
reasons the applicant lost points, because a US lender has to be able to state
them.

A caller is not asked for all 105 features. Anything omitted falls back to the
training-set median or modal value, which is recorded in the model bundle. That
is a real modelling decision with a real cost, so the response says how many
fields were defaulted: a score built from four supplied fields and a hundred
defaults deserves less trust than one built from a full application, and the
caller should be able to see the difference.
"""

from __future__ import annotations

import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import config, cost

MODEL_PATH = os.environ.get("MODEL_PATH",
                            os.path.join(config.MODEL_DIR, "model.pkl"))

app = FastAPI(
    title="Credit risk underwriting",
    description=("Probability of charge-off for a consumer instalment loan, "
                 "with an expected-value decision. Trained only on data "
                 "available at origination."),
    version="1.0",
)

_bundle: dict[str, Any] | None = None


def load_bundle(path: str = MODEL_PATH) -> dict[str, Any]:
    global _bundle
    if _bundle is None:
        if not os.path.exists(path):
            raise FileNotFoundError(
                "no model at %s -- run `python -m src.train` first" % path)
        with open(path, "rb") as fh:
            _bundle = pickle.load(fh)
    return _bundle


# ---------------------------------------------------------------------------
# request / response
# ---------------------------------------------------------------------------

class Application(BaseModel):
    """A loan application.

    The named fields are the ones that carry most of the signal and that a
    front end would realistically collect. `extra` takes any other contracted
    origination feature by name, so the full matrix is reachable without
    spelling out 105 optional fields here.
    """

    loan_amnt: float = Field(..., gt=0, description="amount requested, dollars")
    term_months: int = Field(36, description="36 or 60")
    int_rate: float | None = Field(None, description="offered APR, percent")
    annual_inc: float = Field(..., gt=0, description="stated annual income")
    dti: float | None = Field(None, description="debt-to-income at application")
    fico_range_low: float | None = Field(None, ge=300, le=850)
    fico_range_high: float | None = Field(None, ge=300, le=850)
    emp_length: str | None = Field(None, description="e.g. '10+ years', '< 1 year'")
    home_ownership: str | None = Field(None, description="RENT / OWN / MORTGAGE")
    purpose: str | None = Field(None, description="e.g. debt_consolidation")
    grade: str | None = Field(None, description="Lending Club grade A-G")
    sub_grade: str | None = None
    verification_status: str | None = None
    revol_util: float | None = None
    revol_bal: float | None = None
    open_acc: float | None = None
    total_acc: float | None = None
    delinq_2yrs: float | None = None
    pub_rec: float | None = None
    inq_last_6mths: float | None = None
    credit_history_months: float | None = Field(
        None, description="months since earliest credit line")

    extra: dict[str, float | str | None] = Field(
        default_factory=dict,
        description="any other origination feature, by column name")

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class Reason(BaseModel):
    variable: str
    value: str
    points_lost: float


class Decision(BaseModel):
    probability_of_default: float
    breakeven_probability: float
    expected_value: float
    recommendation: str
    scorecard_points: float
    fields_supplied: int
    fields_defaulted: int
    adverse_action_reasons: list[Reason]
    model_trained_at: str


# ---------------------------------------------------------------------------
# feature assembly
# ---------------------------------------------------------------------------

def _supplied_values(app_in: Application) -> dict[str, Any]:
    """Flatten the request into a plain column -> value mapping."""
    raw = app_in.model_dump(exclude_none=True)
    extra = raw.pop("extra", {}) or {}

    values: dict[str, Any] = {}
    values.update({k: v for k, v in raw.items() if v is not None})
    values.update({k: v for k, v in extra.items() if v is not None})

    # The model was trained on derived columns, not on the raw ones a caller
    # would naturally send, so recreate them here using exactly the same
    # arithmetic as src/features.py.
    lo = values.get("fico_range_low")
    hi = values.get("fico_range_high")
    if lo is not None and hi is not None:
        values["fico"] = (float(lo) + float(hi)) / 2.0
    elif lo is not None:
        values["fico"] = float(lo) + 2.0

    if "emp_length" in values:
        from src.features import EMP_LENGTH_MAP
        values["emp_length_years"] = EMP_LENGTH_MAP.get(str(values["emp_length"]))

    amount = values.get("loan_amnt")
    income = values.get("annual_inc")
    if amount and income and float(income) > 0:
        values["loan_to_income"] = float(amount) / float(income)

    if "revol_bal" in values and values.get("total_rev_hi_lim"):
        limit = float(values["total_rev_hi_lim"])
        if limit > 0:
            values["revol_util_calc"] = 100.0 * float(values["revol_bal"]) / limit

    history = values.get("credit_history_months")
    if history is not None:
        values["is_thin_file"] = 1.0 if float(history) < 36 else 0.0

    derog = ["delinq_2yrs", "pub_rec", "pub_rec_bankruptcies", "tax_liens",
             "collections_12_mths_ex_med"]
    if any(d in values for d in derog):
        total = sum(float(values.get(d) or 0) for d in derog)
        values["has_derogatory"] = 1.0 if total > 0 else 0.0

    return values


def build_row(app_in: Application, bundle: dict) -> tuple[pd.DataFrame, int, int]:
    names = bundle["feature_names"]
    defaults = bundle["feature_defaults"]
    levels = bundle["category_levels"]

    supplied = _supplied_values(app_in)
    used = [n for n in names if n in supplied]

    row = {}
    for name in names:
        row[name] = supplied.get(name, defaults.get(name))

    frame = pd.DataFrame([row], columns=names)
    for col, cats in levels.items():
        if col in frame.columns:
            frame[col] = pd.Categorical(frame[col].astype(object), categories=cats)
    for col in frame.columns:
        if col not in levels:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    return frame, len(used), len(names) - len(used)


def adverse_action_reasons(card, frame: pd.DataFrame, limit: int = 4) -> list[Reason]:
    """The variables costing this applicant the most points.

    This is the mechanism behind a decline letter. Only the scorecard can do it
    honestly -- the reasons come straight out of the fitted points table, not
    from a post-hoc attribution of a black box.
    """
    points = card.points_table()
    reasons: list[Reason] = []

    for variable in card.selected_:
        if variable not in frame.columns:
            continue
        value = frame[variable].iloc[0]
        binned = card._bin_column(pd.Series([value], name=variable),
                                  variable, fitting=False)
        label = str(binned.iloc[0])
        match = points[(points["variable"] == variable) & (points["bin"] == label)]
        if match.empty:
            continue
        earned = float(match["points"].iloc[0])
        best = float(points[points["variable"] == variable]["points"].max())
        reasons.append(Reason(variable=variable, value=label,
                              points_lost=round(best - earned, 2)))

    reasons.sort(key=lambda r: -r.points_lost)
    return [r for r in reasons if r.points_lost > 0][:limit]


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    try:
        bundle = load_bundle()
    except FileNotFoundError as exc:
        # 503 rather than 500: the service is fine, the artefact is absent. A
        # container started without its model mounted should say so.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "ok",
        "model_trained_at": bundle["trained_at"],
        "n_features": len(bundle["feature_names"]),
        "uses_lc_grade": bundle["use_lc_grade"],
        "lgd_assumption": round(bundle["lgd"], 4),
        "training_default_rate": round(bundle.get("train_default_rate", 0.0), 4),
    }


@app.post("/score", response_model=Decision)
def score(application: Application) -> Decision:
    try:
        bundle = load_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    frame, n_supplied, n_defaulted = build_row(application, bundle)

    model = bundle["lightgbm_calibrated"]
    card = bundle["scorecard"]
    lgd = bundle["lgd"]

    probability = float(model.predict_proba(frame)[:, 1][0])
    points = float(card.score(frame)[0])

    econ = cost.loan_economics(
        pd.DataFrame([{
            "funded_amnt": float(application.loan_amnt),
            "installment": _installment(application),
            "term_months": int(application.term_months),
        }]),
        lgd,
    )
    breakeven = float(econ["breakeven_p"].iloc[0])
    ev = float(cost.expected_value(np.array([probability]), econ)[0])

    return Decision(
        probability_of_default=round(probability, 6),
        breakeven_probability=round(breakeven, 6),
        expected_value=round(ev, 2),
        recommendation="approve" if probability < breakeven else "decline",
        scorecard_points=round(points, 1),
        fields_supplied=n_supplied,
        fields_defaulted=n_defaulted,
        adverse_action_reasons=adverse_action_reasons(card, frame),
        model_trained_at=bundle["trained_at"],
    )


def _installment(app_in: Application) -> float:
    """Monthly payment. Supplied if the caller has it, otherwise amortised.

    Standard annuity formula. Falls back to straight-line if no rate is given,
    which understates the payment slightly and therefore the upside -- the
    conservative direction.
    """
    supplied = (app_in.extra or {}).get("installment")
    if supplied:
        return float(supplied)

    principal = float(app_in.loan_amnt)
    months = int(app_in.term_months)
    if app_in.int_rate is None:
        return principal / months

    monthly = float(app_in.int_rate) / 100.0 / 12.0
    if monthly <= 0:
        return principal / months
    factor = (1 + monthly) ** months
    return principal * monthly * factor / (factor - 1)
