"""Model monitoring: has the population moved away from what we trained on?

Banks do not retrain credit models continuously. A scorecard gets built, signed
off, and then runs for a year or three while the world changes underneath it. So
the monitoring question is specific: is the thing still being asked the question
it was built to answer?

Two measures, both standard practice rather than invented here.

Population Stability Index, on the model score. Bin the training score
distribution, then ask how much probability mass has moved:

    PSI = sum_i ( actual_i - expected_i ) * ln( actual_i / expected_i )

The thresholds everyone uses: below 0.10 stable, 0.10 to 0.25 worth
investigating, above 0.25 the model is being asked about a different
population. Those cut-offs are convention, not theory, and I say so in the
output rather than presenting them as laws.

Characteristic analysis, the same calculation per input variable. PSI on the
score tells you something moved; per-variable PSI tells you what.

Lending Club is a good place to do this honestly because the shift is real and
has a name. The book grew from a few thousand loans a year to hundreds of
thousands, LC repeatedly rewrote its own credit policy, and the mix of loan
purposes changed completely. Nothing needs to be injected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Conventional reading of a PSI value. Widely used, not derived from anything.
PSI_STABLE = 0.10
PSI_SHIFTED = 0.25

# Below this many non-null observations on either side, PSI is not meaningful.
# Several co-applicant columns are close to 100% null in the early vintages, and
# a PSI of 0.0 on an empty column would read as "stable" when the truth is
# "nothing to compare".
MIN_OBSERVATIONS = 100

SMOOTH = 1e-6


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10,
        edges: np.ndarray | None = None) -> tuple[float, pd.DataFrame]:
    """PSI between a baseline sample and a later one.

    Bin edges come from the baseline, always. Re-deriving them on the new
    sample would make the two distributions agree by construction and report
    near-zero drift no matter what happened -- which is the single most common
    way to get PSI wrong.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]

    if len(expected) < MIN_OBSERVATIONS or len(actual) < MIN_OBSERVATIONS:
        return float("nan"), pd.DataFrame()

    if edges is None:
        qs = np.linspace(0, 1, bins + 1)
        edges = np.unique(np.quantile(expected, qs))
        if len(edges) < 3:
            return 0.0, pd.DataFrame()
        edges[0] = -np.inf
        edges[-1] = np.inf

    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)

    e_share = e_counts / max(e_counts.sum(), 1)
    a_share = a_counts / max(a_counts.sum(), 1)

    e_adj = np.clip(e_share, SMOOTH, None)
    a_adj = np.clip(a_share, SMOOTH, None)
    contrib = (a_adj - e_adj) * np.log(a_adj / e_adj)

    detail = pd.DataFrame({
        "bin_low": edges[:-1],
        "bin_high": edges[1:],
        "expected_share": e_share,
        "actual_share": a_share,
        "psi_contribution": contrib,
    })
    return float(contrib.sum()), detail


def categorical_psi(expected: pd.Series, actual: pd.Series
                    ) -> tuple[float, pd.DataFrame]:
    """PSI for a categorical variable, over levels instead of bins."""
    e = expected.astype(object).fillna("__missing__").value_counts(normalize=True)
    a = actual.astype(object).fillna("__missing__").value_counts(normalize=True)
    levels = sorted(set(e.index) | set(a.index), key=str)

    e_share = np.array([e.get(k, 0.0) for k in levels])
    a_share = np.array([a.get(k, 0.0) for k in levels])
    e_adj = np.clip(e_share, SMOOTH, None)
    a_adj = np.clip(a_share, SMOOTH, None)
    contrib = (a_adj - e_adj) * np.log(a_adj / e_adj)

    detail = pd.DataFrame({
        "level": levels,
        "expected_share": e_share,
        "actual_share": a_share,
        "psi_contribution": contrib,
    }).sort_values("psi_contribution", ascending=False)
    return float(contrib.sum()), detail.reset_index(drop=True)


def verdict(value: float) -> str:
    if not np.isfinite(value):
        return "insufficient data"
    if value < PSI_STABLE:
        return "stable"
    if value < PSI_SHIFTED:
        return "investigate"
    return "shifted"


def score_psi_by_period(baseline_scores: np.ndarray, frame: pd.DataFrame,
                        score_col: str, period_col: str, bins: int = 10
                        ) -> pd.DataFrame:
    """Score PSI for each later period against one fixed baseline."""
    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(baseline_scores[np.isfinite(baseline_scores)], qs))
    edges[0] = -np.inf
    edges[-1] = np.inf

    rows = []
    for period, chunk in frame.groupby(period_col, observed=True):
        value, _ = psi(baseline_scores, chunk[score_col].to_numpy(), edges=edges)
        rows.append({
            "period": period,
            "n": int(len(chunk)),
            "psi": value,
            "verdict": verdict(value),
        })
    return pd.DataFrame(rows).sort_values("period").reset_index(drop=True)


def characteristic_analysis(train: pd.DataFrame, later: pd.DataFrame,
                            feature_names: list[str], bins: int = 10
                            ) -> pd.DataFrame:
    """Per-variable PSI, so a score shift can be attributed to a cause."""
    rows = []
    for name in feature_names:
        if name not in train.columns or name not in later.columns:
            continue
        col = train[name]
        try:
            if str(col.dtype) in ("object", "category", "bool"):
                value, _ = categorical_psi(train[name], later[name])
                kind = "categorical"
            else:
                value, _ = psi(train[name].to_numpy(dtype=float),
                               later[name].to_numpy(dtype=float), bins=bins)
                kind = "numeric"
        except (TypeError, ValueError):
            continue
        rows.append({
            "variable": name,
            "kind": kind,
            "psi": value,
            "verdict": verdict(value),
        })
    # NaN sorts last, so columns with nothing to compare do not head the table.
    return (pd.DataFrame(rows)
            .sort_values("psi", ascending=False, na_position="last")
            .reset_index(drop=True))


def performance_by_period(y: np.ndarray, p: np.ndarray, period: np.ndarray
                          ) -> pd.DataFrame:
    """AUC and observed default rate per period.

    PSI only sees the inputs. A model can look perfectly stable on PSI and still
    be losing discrimination, so the two have to be read together.
    """
    from sklearn.metrics import roc_auc_score

    frame = pd.DataFrame({"y": y, "p": p, "period": period})
    rows = []
    for per, chunk in frame.groupby("period", observed=True):
        auc = (float(roc_auc_score(chunk["y"], chunk["p"]))
               if chunk["y"].nunique() > 1 else float("nan"))
        rows.append({
            "period": per,
            "n": int(len(chunk)),
            "default_rate": float(chunk["y"].mean()),
            "mean_predicted": float(chunk["p"].mean()),
            "auc": auc,
        })
    return pd.DataFrame(rows).sort_values("period").reset_index(drop=True)
