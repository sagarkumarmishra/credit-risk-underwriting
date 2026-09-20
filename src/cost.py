"""The money. Turning a probability into an approve/decline decision.

A model that ranks borrowers well is not yet a lending policy. The policy
question is where to cut, and the answer does not come from the ROC curve. It
comes from arithmetic on what a good loan earns and what a bad one costs.

For a single loan, approving it has expected value

    EV = (1 - p) * interest_earned  -  p * LGD * exposure

where p is the probability of charge-off. Set EV > 0 and rearrange, and the
break-even probability is

    p* = interest_earned / (interest_earned + LGD * exposure)

That is worth sitting with, because it says the correct cutoff is not one
number. A 60-month loan at 26% earns far more interest than a 36-month loan at
7%, so it can tolerate far more risk. Every "we chose a threshold of 0.5"
notebook is implicitly claiming otherwise.

LGD is estimated from history rather than assumed. This is the one place where
the post-origination columns are the right tool: a cost model is built from
resolved loans, looking backwards, which is exactly what those columns are for.
Using them as model *features* predicts the future from the future; using them
to measure what past defaults actually cost is just accounting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_lgd(con) -> dict[str, float]:
    """Loss given default, measured on loans that actually charged off.

    recoveries can exceed remaining principal on odd rows (fees, adjustments),
    so the per-loan figure is clipped to [0, 1] before averaging. The median is
    reported alongside the mean because the distribution is lumpy -- a lot of
    loans recover nothing at all.
    """
    row = con.execute(
        """
        WITH co AS (
            SELECT
                funded_amnt,
                total_rec_prncp,
                recoveries,
                collection_recovery_fee,
                greatest(0.0, least(1.0,
                    1.0 - (total_rec_prncp + recoveries - collection_recovery_fee)
                          / nullif(funded_amnt, 0)
                )) AS lgd
            FROM loans
            WHERE is_charged_off = 1 AND funded_amnt > 0
        )
        SELECT
            count(*),
            avg(lgd),
            median(lgd),
            quantile_cont(lgd, 0.25),
            quantile_cont(lgd, 0.75),
            avg(recoveries / nullif(funded_amnt, 0))
        FROM co
        """
    ).fetchone()

    n, mean, med, q25, q75, rec_rate = row
    return {
        "n_charged_off": int(n),
        "lgd_mean": float(mean),
        "lgd_median": float(med),
        "lgd_q25": float(q25),
        "lgd_q75": float(q75),
        "recovery_rate_mean": float(rec_rate),
    }


def loan_economics(df: pd.DataFrame, lgd: float) -> pd.DataFrame:
    """Per-loan interest earned if repaid, and loss if charged off.

    Interest is the scheduled amount: installment * term - funded. That is what
    an underwriter knows at decision time, which is the whole point. Using the
    interest actually received would be looking at the outcome again.
    """
    out = pd.DataFrame(index=df.index)
    exposure = df["funded_amnt"].astype(float)
    scheduled_total = df["installment"].astype(float) * df["term_months"].astype(float)

    out["exposure"] = exposure
    out["interest_if_paid"] = (scheduled_total - exposure).clip(lower=0.0)
    out["loss_if_default"] = lgd * exposure

    # The break-even default probability for this specific loan.
    denom = out["interest_if_paid"] + out["loss_if_default"]
    out["breakeven_p"] = np.where(denom > 0,
                                  out["interest_if_paid"] / denom,
                                  0.0)
    return out


def expected_value(p: np.ndarray, econ: pd.DataFrame) -> np.ndarray:
    """Expected profit of approving each loan at predicted probability p."""
    return (1.0 - p) * econ["interest_if_paid"].to_numpy() \
        - p * econ["loss_if_default"].to_numpy()


def realised_profit(approved: np.ndarray, y: np.ndarray, econ: pd.DataFrame) -> float:
    """Actual money made by a policy, using the outcomes we know happened.

    Good approved loan earns its interest; bad approved loan loses LGD *
    exposure; declined loans earn and lose nothing.
    """
    interest = econ["interest_if_paid"].to_numpy()
    loss = econ["loss_if_default"].to_numpy()
    per_loan = np.where(y == 1, -loss, interest)
    return float(per_loan[approved].sum())


def policy_summary(name: str, approved: np.ndarray, y: np.ndarray,
                   econ: pd.DataFrame) -> dict:
    n = len(y)
    n_app = int(approved.sum())
    profit = realised_profit(approved, y, econ)
    approved_bad = int(y[approved].sum()) if n_app else 0
    return {
        "policy": name,
        "approval_rate": n_app / n,
        "n_approved": n_app,
        "book_default_rate": (approved_bad / n_app) if n_app else 0.0,
        "total_profit": profit,
        # Per *application*, not per approval. A policy that approves three
        # loans and makes a fortune on each is not better than one that
        # approves thousands, and dividing by approvals would say it was.
        "profit_per_application": profit / n,
        "capital_deployed": float(econ["exposure"].to_numpy()[approved].sum()),
    }


def sweep_threshold(p: np.ndarray, y: np.ndarray, econ: pd.DataFrame,
                    steps: int = 200) -> pd.DataFrame:
    """Profit as a function of a single global cutoff.

    Produces the curve behind the 'AUC-optimal is not profit-optimal' chart.
    """
    grid = np.linspace(0.0, 1.0, steps + 1)
    rows = []
    for t in grid:
        approved = p <= t
        n_app = int(approved.sum())
        rows.append({
            "threshold": float(t),
            "approval_rate": n_app / len(y),
            "total_profit": realised_profit(approved, y, econ),
            "book_default_rate": (float(y[approved].mean()) if n_app else 0.0),
        })
    return pd.DataFrame(rows)


def best_global_threshold(p: np.ndarray, y: np.ndarray, econ: pd.DataFrame,
                          steps: int = 200) -> tuple[float, float]:
    """The cutoff that made the most money in hindsight.

    Hindsight is the operative word: this is tuned on the outcomes it is being
    scored against, so it is an upper bound on what a single fixed cutoff could
    have achieved, not a deployable number. Reported as such everywhere.
    """
    sweep = sweep_threshold(p, y, econ, steps)
    row = sweep.loc[sweep["total_profit"].idxmax()]
    return float(row["threshold"]), float(row["total_profit"])
