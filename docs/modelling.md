# Modelling

Everything from the raw extract to a calibrated probability, and why each step is
what it is.

---

## 1. The population

Two filters, both load-bearing.

```sql
WHERE loan_status IN ('Fully Paid', 'Charged Off',
                      'Does not meet the credit policy. Status:Fully Paid',
                      'Does not meet the credit policy. Status:Charged Off')
  AND (issue_date + INTERVAL (term_months) MONTH) <= DATE '2019-03-01'
```

The first is obvious: a label only exists once the loan has finished. The two
"does not meet the credit policy" variants are loans Lending Club issued under
older rules and later disowned — still genuine resolved outcomes, so they stay.

`Default` is excluded despite the name. In Lending Club's vocabulary it is a
transitional state on the way to charge-off or recovery, not an endpoint. A lot
of published notebooks treat it as bad and get a slightly wrong label.

The second filter is the one worth arguing about, and it is covered in
[decision-log.md §2](decision-log.md). Short version: resolved-status filtering
alone keeps recent vintages **only if they defaulted early**, because a 60-month
loan issued in 2018 has not had time to be `Fully Paid`. It costs 569,034 loans
and removes a survivorship bias that points the wrong way.

Final population: **779,025 loans, 15.19% charge-off rate.**

Default rate by vintage, after filtering:

```
2007       603  26.20%
2008      2393  20.73%
2009      5281  13.69%
2010     12537  14.01%
2011     21721  15.18%
2012     53367  16.20%
2013    134804  15.60%
2014    175509  14.63%
2015    283026  14.89%
2016     89784  16.10%
```

The 2007–08 vintages are visibly worse — that is the financial crisis, and it is
real rather than noise. They are also tiny (3,000 loans between them), so they
contribute almost nothing to the fit.

---

## 2. Features

102 of the 151 columns are usable at origination. After replacing four with
derived versions and dropping two raw date strings, **105 features** reach the
model.

Seven derived features, each with a reason:

| Feature | Why it exists |
|---|---|
| `fico` | FICO arrives as a 5-point band; the midpoint is the usable number and the width is constant |
| `loan_to_income` | `dti` covers existing debt but not this new obligation |
| `installment_to_income` | closer to what an underwriter asks: can they make the payment |
| `revol_util_calc` | recomputed utilisation, fills gaps where LC's own `revol_util` is null |
| `emp_length_years` | an ordered category stored as prose; ordinal so a tree can split it once |
| `is_thin_file` | two years of history is a different proposition from twenty |
| `has_derogatory` | the individual counters are sparse and zero-inflated; one flag often survives into a scorecard |

That is deliberately not many. Dozens of hand-crafted ratios is how you overfit a
credit model and how you make it impossible to explain to a credit officer. If I
could not write the "why" line, the feature is not there.

**Missing values are not imputed.** In credit data a null is usually
informative — no mortgage accounts means no mortgage, not an unknown number of
them. LightGBM sends `nan` down its own branch at every split; the scorecard gives
it its own WOE bin. Both let the model say how risky "unknown" is.

Two places where naive arithmetic would lie:

```python
# A stated income of 0 is a data-entry artefact, not a real income.
# inf lets one bad row dominate a split; 0 asserts the loan is trivially
# affordable. NaN says what is true: we do not know.
denom = denominator.where(denominator > 0)

# Where credit history length is unknown, is_thin_file is nan rather than 0.
# Claiming "not thin" would be a guess.
np.where(history.isna(), np.nan, (history < 36).astype(float))
```

**Categoricals stay categorical.** LightGBM splits on them natively.
One-hot encoding `sub_grade` into 35 sparse columns makes the trees worse, not
better. The category *levels* are frozen from the training matrix and reapplied
to test and serving matrices — LightGBM stores category codes, not labels, so a
test matrix that builds its own ordering silently scrambles predictions. There is
a function for it with the reason written above it.

---

## 3. The split

```
fit         230,706 loans   2007-06 to 2013-12   15.65% default
calibrate   175,509 loans   2014-01 to 2014-12   14.63% default
test        372,810 loans   2015-01 to 2016-03   15.18% default
```

Boundaries are **dates, not percentages**. Credit models are always deployed on
vintages that did not exist at training time, so a random split measures the
wrong thing.

The calibration slice is the **last 12 months of the training window**, not a
random subset. Calibrating on a random subset means calibrating on the period you
trained on, and the result looks much better than it will be in production. This
choice has a visible consequence in §6, which I have left visible rather than
tuned away.

Early stopping watches the calibration slice too, so the stopping point is chosen
against the closest available analogue to next year's applications. It stopped at
iteration 340 of a permitted 600.

---

## 4. Three models

### Constant baseline

Predicts the training default rate for everyone. AUC 0.5000 by construction, but
perfectly calibrated on average, with Brier 0.12877. It exists so every later
Brier score has something to be compared against — a model that cannot beat
0.12877 is not adding information, only noise.

### WOE scorecard

The industry standard for forty years, and it is here because of a legal
requirement rather than nostalgia: a US lender must send an adverse action notice
stating specific reasons for a decline.

```
WOE_i = ln( (goods in bin i / all goods) / (bads in bin i / all bads) )
IV    = Σ ( goods_i/G − bads_i/B ) · WOE_i
```

Positive WOE means the bin is safer than average. IV reads: below 0.02 useless,
0.02–0.1 weak, 0.1–0.3 medium, above 0.3 strong.

Implementation choices that matter:

- **Quantile bins, not equal-width.** Credit variables are heavily skewed; equal
  width on `revol_bal` puts 99% of borrowers in the first bin.
- **Open-ended outer bins** (`-inf`, `+inf`) so unseen extremes at serving time
  still land somewhere.
- **Missing is its own bin**, never imputed.
- **Rare categorical levels collapsed** at 500 observations, so a 12-loan
  category cannot earn its own WOE.
- **Laplace smoothing at 0.5.** Without it a bin containing no defaults gives
  `WOE = +inf`, which propagates a NaN into the regression and takes the model
  down.
- **L2 regularisation.** Bureau variables are heavily correlated; without it
  coefficients flip sign between vintages.

27 of 105 variables cleared IV ≥ 0.02. Points are scaled so 20 points doubles the
odds: `factor = PDO/ln(2)`, `offset = base − factor·ln(base_odds)`.

The points formula needs a **negative** sign, because the regression predicts
default while WOE is oriented towards good:

```python
"points": -self.factor * beta * woe,
```

Omit it and every sign inverts, the AUC is unchanged, and the scorecard reads
exactly backwards. It is a genuinely easy bug to ship, so there is a test that
asserts a low-risk bin scores positive WOE.

### LightGBM

600 trees permitted, learning rate 0.03, 63 leaves, `min_child_samples=300`,
subsample and colsample at 0.8/0.7, `reg_lambda=5`. Conservative on purpose:
with 230k rows and 105 correlated features the risk is memorising vintage-specific
noise, not underfitting.

---

## 5. Results

| Model | AUC | Gini | KS | Brier | Log loss |
|---|---|---|---|---|---|
| Constant baseline | 0.5000 | 0.0000 | — | 0.12877 | — |
| WOE scorecard (27 vars) | 0.6892 | 0.3783 | 0.2752 | 0.12146 | — |
| LightGBM | 0.7009 | 0.4019 | 0.2919 | 0.12129 | — |
| LightGBM + isotonic | 0.7007 | 0.4014 | 0.2918 | **0.12069** | — |

**Accuracy is not in that table on purpose.** At a 15% default rate, predicting
that everybody repays scores 85%.

KS of 0.29 and Gini of 0.40 are in the normal range for an application scorecard
on pre-screened consumer loans. If this reported 0.95 AUC I would be looking for
the leak, not celebrating.

The scorecard reaches 98.3% of LightGBM's Gini with 27 readable variables instead
of 340 trees. That is the interesting comparison, and §7 puts a price on the gap.

---

## 6. Calibration, and its residual failure

Isotonic rather than Platt scaling — plenty of data, and no reason to assume the
miscalibration has a logistic shape. `FrozenEstimator` wraps the booster so the
calibrator cannot refit it and undo the out-of-time discipline.

Brier improves 0.12129 → 0.12069 with AUC unchanged. **Calibration changes the
price, never the order.**

It only closes about a third of the gap, though:

| | mean predicted | observed |
|---|---|---|
| raw | 0.1208 | 0.1518 |
| calibrated | 0.1310 | 0.1518 |

The calibrator was fitted on 2014 originations, which defaulted at 14.63%. The
test vintages run hotter. So it is faithfully reproducing a base rate that has
since moved — which is a real limitation of a fixed calibration slice, and
exactly what the monitoring in [mlops.md](mlops.md) detects.

I could have hidden this by calibrating on a random slice of the test period.
That would produce a prettier chart and a dishonest one.

---

## 7. The price of interpretability

Champion/challenger, on the out-of-time book:

| | LightGBM | Scorecard | Gap |
|---|---|---|---|
| AUC | 0.7007 | 0.6892 | −0.0115 |
| Brier | 0.12069 | 0.12146 | +0.00077 |
| Profit | $335.0m | $332.1m | −$2.9m |

So being able to explain a decline costs **0.012 AUC and 0.86% of profit**.

That is the whole point of running the comparison. "Interpretable models are
worth it" and "just use gradient boosting" are both opinions; 0.86% of a $335m
book is a number a credit committee can accept or reject.

---

## 8. Known limitations

- **Selection bias.** Every loan was already approved by Lending Club. Reject
  inference is the standard remedy and is not implemented — the declined
  applications simply are not in the data.
- **569,034 loans discarded** by the maturity rule. Survival modelling of
  time-to-default would recover them.
- **One pooled LGD.** In practice LGD varies by term, grade and vintage.
- **Scheduled interest as the upside.** No funding cost, no prepayment, no
  discounting. Each omission is real; none changes the ranking of the policies.
- **No segmentation.** Separate scorecards by term or thin-file status is normal
  practice and would probably close part of the gap to LightGBM.
- **Geography excluded**, at an unmeasured cost, for the reasons in
  [decision-log.md §4](decision-log.md).
