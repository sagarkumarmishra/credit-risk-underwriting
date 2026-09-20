"""A weight-of-evidence scorecard: the model a credit team can actually deploy.

Gradient boosting will win on AUC. It is still not automatically the right
answer, because a lender in the US has to send an adverse action notice
explaining *why* an application was declined, in specific reasons, to the
applicant. "The 400-tree ensemble assigned you a high score" is not a reason.

The industry answer, and it has been for forty years, is a points-based
scorecard: bin every variable, replace each bin with its weight of evidence,
fit a logistic regression on those, then rescale the coefficients into points.
The result is a table a human can read, audit, and argue with, and a decline
reason falls out of it directly -- whichever bins cost the applicant the most
points.

This module builds one properly rather than gesturing at one:

    WOE_i = ln( (goods in bin i / all goods) / (bads in bin i / all bads) )
    IV    = sum_i ( goods_i/G - bads_i/B ) * WOE_i

Positive WOE means the bin is safer than average. IV is the variable's total
predictive strength, and the conventional reading is: below 0.02 useless,
0.02-0.1 weak, 0.1-0.3 medium, above 0.3 strong.

Points use the standard scaling: a fixed number of points doubles the odds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MISSING_LABEL = "__missing__"

# Laplace-style smoothing. Without it a bin containing no defaults at all gives
# WOE = +inf, which then propagates into the regression as a NaN and takes the
# whole model down. 0.5 is the conventional choice.
SMOOTH = 0.5


def bin_numeric(x: pd.Series, max_bins: int = 10, min_share: float = 0.05
                ) -> list[float]:
    """Quantile edges for a numeric variable, coarse enough that bins are stable.

    Quantiles rather than equal width, because credit variables are heavily
    skewed -- equal-width bins on revol_bal would put 99% of borrowers in the
    first bin. Duplicate edges are dropped, which is why the returned list can
    be shorter than max_bins asks for.
    """
    vals = pd.to_numeric(x, errors="coerce").dropna()
    if vals.nunique() <= 1:
        return []

    n_bins = max(2, min(max_bins, int(1.0 / min_share)))
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(vals, qs))

    # Open both ends so unseen extremes at predict time still land somewhere.
    if len(edges) < 3:
        return []
    edges[0] = -np.inf
    edges[-1] = np.inf
    return [float(e) for e in edges]


def apply_bins(x: pd.Series, edges: list[float]) -> pd.Series:
    """Label each value with its bin, with nulls in their own explicit bin.

    Missing is not imputed. In credit data a null is usually informative -- no
    mortgage accounts means no mortgage, not an unknown number of them -- and
    giving it its own bin lets the WOE say how risky "unknown" is.
    """
    if not edges:
        return pd.Series([MISSING_LABEL] * len(x), index=x.index, dtype=object)
    num = pd.to_numeric(x, errors="coerce")
    cut = pd.cut(num, bins=edges, include_lowest=True)
    out = cut.astype(object)
    out[num.isna()] = MISSING_LABEL
    return out.astype(str)


def bin_categorical(x: pd.Series, min_count: int = 500) -> pd.Series:
    """Collapse rare levels, so a 12-loan category cannot get its own WOE."""
    s = x.astype(object).where(x.notna(), MISSING_LABEL).astype(str)
    counts = s.value_counts()
    rare = set(counts[counts < min_count].index)
    return s.where(~s.isin(rare), "__rare__")


def woe_table(binned: pd.Series, y: pd.Series) -> pd.DataFrame:
    """WOE and IV contribution per bin."""
    frame = pd.DataFrame({"bin": binned.to_numpy(), "y": y.to_numpy()})
    grp = frame.groupby("bin", observed=True)["y"].agg(["count", "sum"])
    grp.columns = ["n", "bad"]
    grp["good"] = grp["n"] - grp["bad"]

    total_bad = grp["bad"].sum() + SMOOTH * len(grp)
    total_good = grp["good"].sum() + SMOOTH * len(grp)

    grp["bad_rate"] = grp["bad"] / grp["n"]
    grp["dist_bad"] = (grp["bad"] + SMOOTH) / total_bad
    grp["dist_good"] = (grp["good"] + SMOOTH) / total_good
    grp["woe"] = np.log(grp["dist_good"] / grp["dist_bad"])
    grp["iv"] = (grp["dist_good"] - grp["dist_bad"]) * grp["woe"]
    return grp.reset_index()


class Scorecard:
    """Fit, transform and score. Deliberately small and inspectable.

    Not a sklearn Pipeline: the binning has to be learned on train and frozen,
    and the fitted bin table is the deliverable a credit officer reviews. Hiding
    that inside a pipeline object makes the one artefact people actually want
    harder to get at.
    """

    def __init__(self, max_bins: int = 10, min_iv: float = 0.02,
                 base_score: int = 600, base_odds: float = 10.0, pdo: int = 20):
        self.max_bins = max_bins
        self.min_iv = min_iv
        self.base_score = base_score
        self.base_odds = base_odds
        self.pdo = pdo

        self.edges_: dict[str, list[float]] = {}
        self.categorical_: set[str] = set()
        self.woe_maps_: dict[str, dict[str, float]] = {}
        self.iv_: dict[str, float] = {}
        self.selected_: list[str] = []
        self.model_ = None

    # -- fitting -----------------------------------------------------------

    def _bin_column(self, x: pd.Series, name: str, fitting: bool) -> pd.Series:
        is_cat = str(x.dtype) in ("object", "category", "bool")
        if fitting and is_cat:
            self.categorical_.add(name)
        if name in self.categorical_:
            return bin_categorical(x)
        if fitting:
            self.edges_[name] = bin_numeric(x, self.max_bins)
        return apply_bins(x, self.edges_.get(name, []))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Scorecard:
        from sklearn.linear_model import LogisticRegression

        y = y.astype(int).reset_index(drop=True)
        X = X.reset_index(drop=True)

        woe_frames = {}
        for name in X.columns:
            binned = self._bin_column(X[name], name, fitting=True)
            table = woe_table(binned, y)
            iv = float(table["iv"].sum())
            self.iv_[name] = iv
            self.woe_maps_[name] = dict(zip(table["bin"], table["woe"]))
            woe_frames[name] = binned.map(self.woe_maps_[name]).astype(float)

        # Keep only variables with enough standalone strength. This is the
        # normal first pass in scorecard work; it also keeps the final table
        # short enough for a person to read.
        self.selected_ = sorted(
            [n for n, iv in self.iv_.items() if iv >= self.min_iv],
            key=lambda n: -self.iv_[n],
        )
        if not self.selected_:
            raise ValueError("no variable cleared min_iv=%.3f" % self.min_iv)

        W = pd.DataFrame({n: woe_frames[n] for n in self.selected_})
        W = W.fillna(0.0)

        # Modest L2 so that correlated bureau variables -- and there are many --
        # do not produce wild coefficients that flip sign between vintages. L2
        # is sklearn's default, so it is left unnamed: passing penalty="l2"
        # explicitly is deprecated as of sklearn 1.8.
        self.model_ = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        self.model_.fit(W, y)
        return self

    # -- use ---------------------------------------------------------------

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = {}
        for name in self.selected_:
            binned = self._bin_column(X[name], name, fitting=False)
            cols[name] = binned.map(self.woe_maps_[name]).astype(float)
        return pd.DataFrame(cols, index=X.index).fillna(0.0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(self.transform(X))

    def predict_default_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X)[:, 1]

    # -- the artefact a human reads ----------------------------------------

    @property
    def factor(self) -> float:
        return self.pdo / np.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * np.log(self.base_odds)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Points, where higher is safer and every `pdo` points doubles the odds."""
        p = np.clip(self.predict_default_proba(X), 1e-6, 1 - 1e-6)
        odds_good_to_bad = (1 - p) / p
        return self.offset + self.factor * np.log(odds_good_to_bad)

    def points_table(self) -> pd.DataFrame:
        """Per-bin points. This is the scorecard, in the sense a lender means it."""
        coefs = dict(zip(self.selected_, self.model_.coef_[0]))
        rows = []
        for name in self.selected_:
            beta = coefs[name]
            for bin_label, woe in self.woe_maps_[name].items():
                # Negative sign because the regression predicts default while
                # WOE is oriented towards good. Without it the table reads
                # backwards, which is a genuinely easy mistake to ship.
                rows.append({
                    "variable": name,
                    "bin": bin_label,
                    "woe": woe,
                    "coefficient": beta,
                    "points": -self.factor * beta * woe,
                    "iv_variable": self.iv_[name],
                })
        out = pd.DataFrame(rows)
        return out.sort_values(["iv_variable", "variable", "bin"],
                              ascending=[False, True, True]).reset_index(drop=True)

    def iv_summary(self) -> pd.DataFrame:
        rows = [{"variable": n, "iv": iv, "selected": n in self.selected_}
                for n, iv in self.iv_.items()]
        out = pd.DataFrame(rows).sort_values("iv", ascending=False)
        out["strength"] = pd.cut(
            out["iv"],
            bins=[-np.inf, 0.02, 0.1, 0.3, np.inf],
            labels=["useless", "weak", "medium", "strong"],
        )
        return out.reset_index(drop=True)
