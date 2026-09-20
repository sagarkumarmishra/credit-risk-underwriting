# Decision log

Every judgement call in the project, why it went the way it did, and what it
cost. Including the ones I got wrong, because a log that only records successes
is marketing.

---

## 1. Dataset: four rounds of rejection before Lending Club

I needed finance-domain, industry-realistic, large, with a time dimension for
out-of-time validation, a genuine leakage surface, and reliably downloadable
without credentials. That last requirement killed more candidates than any
other: if `make data` fails for whoever reviews the repo, the project is dead on
arrival.

| Candidate | Outcome |
|---|---|
| UCI German Credit (1,000 rows) | Rejected — too small, and **no post-origination columns at all**, so the central thesis is impossible to demonstrate |
| `CreditScoring.csv` (182 KB) | Rejected — same problem, plus no time dimension |
| HMDA mortgage data (CFPB) | Rejected — 4 KB ranged requests worked, 8 MB requests returned HTTP 403. Rate-limited or size-capped. Unreliable for a reviewer |
| Freddie Mac / Fannie Mae | Rejected — registration required |
| **Lending Club accepted loans** | **Chosen** — 1.56 GB, 2.26M loans, 151 columns, 38 of them post-origination, `issue_d` spanning 2007–2018 |

Lending Club took their own download page offline, so the source is a mirror.
`src/config.py` records the URL and the exact byte count, and `get_data.py`
**refuses to proceed** if the remote size ever differs — every number in the docs
was measured against the file as it is today, and silently training on a
different file would make all of them lies.

---

## 2. The population: two rules, and the second one is the interesting one

**Rule one: resolved outcomes only.** A loan's label only exists once it has
finished. `Current`, `Late (16-30)`, `Late (31-120)` and `In Grace Period` all
describe loans still in flight.

`Default` looks like it belongs with `Charged Off`, and plenty of public
notebooks put it there. It does not belong: in Lending Club's vocabulary
`Default` is a transitional state on the way to either charge-off or recovery,
not an endpoint.

**Rule two: the loan must have had its full term to mature.** This is the one I
would want to be asked about in an interview.

Filtering on resolved status alone *looks* sufficient and is not. A 60-month loan
issued in mid-2018 physically cannot appear as `Fully Paid` in a 2018 Q4
snapshot — there has not been time. So among recent vintages, the only loans with
a resolved status are the ones that **resolved early, which means they defaulted**.
Keeping them without a maturity rule loads the sample with defaults in exactly
the periods you would most want to validate on.

The rule is `issue_date + term <= snapshot`. What it costs:

| Vintage | Kept | Dropped | Kept % |
|---|---|---|---|
| 2013 | 134,804 | 0 | 100% |
| 2014 | 175,509 | 47,593 | 78.7% |
| 2015 | 283,026 | 92,519 | 75.4% |
| 2016 | 89,784 | 203,311 | 30.6% |
| 2017 | 0 | 169,300 | **0%** |
| 2018 | 0 | 56,311 | **0%** |

569,034 loans discarded. It hurts and it is correct. The alternative —
right-censored survival modelling of time-to-default — would recover them and is
listed as future work rather than pretended.

---

## 3. The leaky columns are banned as features and used for the cost model

This looks like a contradiction and is not. `recoveries`, `total_rec_prncp` and
`collection_recovery_fee` are forbidden as model inputs, and they are exactly the
right columns for estimating loss given default:

```python
lgd = 1 - (total_rec_prncp + recoveries - collection_recovery_fee) / funded_amnt
```

Using them as **features** predicts the future from the future. Using them to
measure **what past defaults actually cost** is accounting. The difference is the
direction of time, and it is worth being explicit about because "never touch
those columns" is the wrong lesson.

Result: LGD mean 0.539, median 0.562, IQR 0.364–0.723, from 118,371 charged-off
loans. An assumed 100% LGD — which is common in write-ups — would have
overstated losses by roughly a factor of two and made every policy look worse
than it is.

---

## 4. Geography excluded on legal grounds, not statistical ones

`zip_code` and `addr_state` are genuinely known at application and carry real
predictive signal. They are out anyway.

US fair-lending law (ECOA, Regulation B) treats geography as a potential proxy
for protected characteristics, and three-digit ZIP in particular maps closely
onto racial composition. A model that declines applicants for living in the
wrong postcode is a redlining problem regardless of its AUC.

They are classified `SENSITIVE` rather than `DROP` so the reason is recorded in
the code rather than lost. The cost of excluding them is real and unmeasured —
quantifying it would mean building the model I have just argued should not exist.

---

## 5. Lending Club's own grade is kept, with an escape hatch

`grade`, `sub_grade` and `int_rate` are legitimately available at origination.
They are also the *output of Lending Club's internal scoring model*. Training on
them means part of what the model learns is "agree with Lending Club", which
flatters the metrics and teaches you nothing about the borrower.

I kept them, because a real underwriting model would have access to the pricing
it is being asked to approve. `train.py --no-lc-grade` drops all three so the
cost is measurable rather than assumed. Worth noting that `int_rate` is also one
of the three most-shifted variables in monitoring (PSI 0.2977) — Lending Club
repriced over the period, and any model leaning on it inherits that instability.

---

## 6. Split: out-of-time, three ways, and the calibration slice comes from the end

Train on 2007–2013, calibrate on 2014, test on 2015–2016 Q1. All boundaries are
dates, not percentages.

The detail that matters: **the calibration slice is the last 12 months of the
training window, not a random subset of it**. Calibrating on a random subset
means calibrating on the same period you trained on, and the calibration then
looks far better than it will be in production. Taking the most recent 12 months
is the closest available analogue to "next year's applications".

This turned out to have a visible consequence, discussed in §10.

---

## 7. Two models, for two different reasons

**LightGBM** because it wins: 0.7009 against the scorecard's 0.6892.

**A WOE scorecard** because a US lender has to send an adverse action notice
stating specific reasons for a decline. "The 340-tree ensemble assigned you a
high score" is not a reason a regulator accepts. The scorecard produces a points
table a human can read and audit, and a decline reason falls out of it directly.

Built the way the industry actually builds them: quantile bins with open ends,
missing as its own bin rather than imputed, rare categorical levels collapsed,
Laplace smoothing so a bin with zero defaults does not produce infinite WOE, IV
selection at 0.02, and points scaled so 20 points doubles the odds.

27 of 105 variables cleared IV ≥ 0.02. The rest are noise or redundant, and a
shorter table is easier for a credit officer to sign off.

One implementation detail that is a genuinely easy bug to ship: the points
formula needs a **negative** sign, because the regression predicts default while
WOE is oriented towards good. Without it every sign in the table is inverted,
the AUC is unchanged, and the scorecard reads exactly backwards. There is a test
for it.

---

## 8. Calibration: isotonic, and it only half-works

Isotonic rather than Platt scaling — there is plenty of data and no reason to
assume the miscalibration has a logistic shape.

It improves Brier from 0.12129 to 0.12069 and leaves AUC at 0.7007. That is the
point: **calibration changes the price, never the order.**

But it only closes about a third of the pricing gap. Mean predicted moves from
0.1208 to 0.1310 against an observed 0.1518. The reason is in §6: the calibrator
was fitted on 2014 originations, which defaulted at 14.63%, and the test vintages
run hotter. The calibrator is faithfully reproducing a base rate that has since
moved.

I could have hidden this by calibrating on a random slice of the test period.
That would have produced a prettier chart and a dishonest one.

---

## 9. The decision rule is per-loan, not a global threshold

`p* = interest / (interest + LGD · exposure)`.

The break-even probability is a property of the loan. Middle 90% of this book:
14.9% to 36.2%, median 24.9%. A 60-month loan at 26% tolerates far more risk
than a 36-month loan at 7%.

Policies compared on the same out-of-time book:

| Policy | Approved | Profit / application |
|---|---|---|
| approve all (what LC did) | 100.0% | $888.80 |
| cut at p > 0.50 | 100.0% | $889.57 |
| cut at Youden J | 56.8% | $649.83 |
| per-loan expected value > 0 | 97.6% | $898.62 |
| best fixed cutoff, in hindsight | 94.0% | $904.60 |

Two findings I did not plan for:

**`p > 0.50` is not a conservative choice, it is not a choice at all.** Not one
of 372,810 applications is predicted above 0.5, so the industry-default threshold
approves the entire book and differs from doing nothing by $0.77 per application.

**Youden's J destroys $239 per application.** It maximises TPR − FPR, a criterion
with no opinion about what a mistake costs, and declines 43% of a book that is
profitable at 15% default.

The "best fixed cutoff in hindsight" row is labelled as such everywhere because
it is tuned on the outcomes it is scored against. It is an upper bound on what a
single fixed number could have achieved, not a deployable policy.

**The honest caveat:** the expected-value rule adds 1.1%, not 40%. Every loan
here was already approved by Lending Club, so the model is finding residual risk
among applicants who already passed a credit screen. The easy declines happened
upstream and are invisible in this data. Reject inference is the standard remedy
and is not implemented.

---

## 10. A hypothesis of mine that failed

I predicted two mistakes would both inflate AUC: leakage, and a random split
instead of out-of-time.

| Mistake | Effect on AUC |
|---|---|
| post-origination columns | **+0.2916** |
| random instead of out-of-time split | **−0.0042** |

The first held far harder than expected — cell A is a *perfect* 1.0000
classifier, not the 0.99 I guessed. The second is the **opposite sign**: the
random split was very slightly harder.

The mechanism is my own maturity rule from §2. Removing the unmatured vintages
takes out 2017 and 2018 entirely, and everything left defaults at 14–16% in every
year from 2010 to 2016. There is almost no temporal drift left to exploit, so a
random split gains nothing — and the particular out-of-time test window happens
to be marginally easier.

On a dataset without a maturity filter, I would expect the textbook result. I am
keeping the failed prediction in the write-up with the mechanism, because it is
true and because one claim holding spectacularly while the other fails is more
credible than two tidy confirmations.

A related detail worth recording: `out_prncp` scores exactly **0.5000** alone. It
is leaky in principle but inert in a matured population, where outstanding
principal is always zero.

---

## 11. Monitoring thresholds are convention, and labelled as such

PSI bands of 0.10 and 0.25 are what the industry uses. They are not derived from
anything, and the code says so rather than presenting them as theory.

Gate tolerances are stated in one place with reasons:

```python
AUC_TOLERANCE = 0.002        # within a fifth of a Gini point counts as equal
BRIER_TOLERANCE = 0.0005     # calibration must not degrade materially
PROFIT_TOLERANCE = 0.0       # profit is not allowed to go backwards at all
```

AUC has a tolerance because a simpler or cheaper model that *matches* the
champion is worth promoting. Profit has none.

The scorecard failed all three quality checks, narrowly. Interpretability costs
0.012 AUC and 0.86% of profit on a $335m book. That is a number a committee can
decide on.

---

## 12. Things that broke, and what they taught me

**`ValueError: cannot convert NA to integer`.** DuckDB returns pandas *nullable*
`Int64`/`boolean` arrays for any column containing nulls, and `pd.NA` cannot be
cast to a fixed-width int. The failure surfaced several steps downstream of the
cause, naming neither the column nor the reason. Fixed by converting all
extension dtypes to `float64` at the top of `engineer()`, where missing is
`np.nan` and LightGBM treats it as its own branch. Deliberately **not** fixed
with `fillna()`, which would have invented values.

**A PSI of 0.0 that meant "no data".** Several `sec_app_*` columns are ~100% null
in early vintages. After dropping non-finite values the baseline array was empty
and `np.quantile` raised `IndexError`. The tempting fix is to return 0.0, which
would report those columns as perfectly *stable* when the truth is "nothing to
compare". Returns `NaN` and a verdict of `"insufficient data"` instead.

**Three API breaks from current library versions.** `CalibratedClassifierCV(cv="prefit")`
removed in favour of `FrozenEstimator`; `LogisticRegression(penalty="l2")`
deprecated; LightGBM's `eval_set` renamed to `eval_X`/`eval_y`. All three would
have been invisible if I had pinned to versions from a year-old tutorial.

**MLflow silently not recording.** MLflow 3 put the plain filesystem backend into
maintenance mode and refuses to write to `./mlruns`. Switched to a local sqlite
file. The tracking call is wrapped in `try/except` so a broken store cannot kill
an otherwise-successful training run.

**A test that was wrong, not the code.** I wrote an "uninformative variable"
test as `bins = ["a","b"]*500` against `y = [0,1]*500` and expected IV near zero.
It returned 13.79 — correctly, because that construction makes bin "a" land on
every good and bin "b" on every bad. A *perfect* separator. The test now builds
two bins with the same 20% bad rate and asserts that first.

**Charts I had to rewrite after looking at them.** The profit chart originally
put five annotations on one axes; every policy approves between 56% and 100%, so
they all collided in one corner and 85% of the plot showed the trivial region.
Rewritten as two panels with a legend. The drift chart originally used twin
y-axes for AUC and default rate — they crossed, making two unrelated quantities
on unrelated scales look like they were converging on something. Rewritten as
three stacked panels sharing an x-axis.

---

## 13. What I would do next

1. **Rolling recalibration.** §8's residual under-prediction is the clearest open
   problem, and the monitoring already detects it.
2. **Reject inference.** The population is pre-screened; the declined
   applications are invisible. This is the single biggest limitation.
3. **Survival modelling of time-to-default.** Would let the maturity rule be
   relaxed and recover 569,034 loans.
4. **Measure what excluding geography costs**, carefully and without deploying
   the result.
5. **Segmented scorecards** by term or by thin-file status, which is standard
   practice and would probably close part of the gap to LightGBM.
