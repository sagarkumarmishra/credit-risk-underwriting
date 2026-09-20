# Serving, monitoring and promotion

The part that decides whether a model is an artefact or a system.

---

## 1. Scoring API

```bash
make api    # uvicorn src.serve:app --port 8000
```

Two endpoints. `GET /health` and `POST /score`.

`/score` returns a **decision**, not a bare probability:

```json
{
  "probability_of_default": 0.1341,
  "breakeven_probability": 0.2912,
  "expected_value": 1783.24,
  "recommendation": "approve",
  "scorecard_points": 612.4,
  "fields_supplied": 9,
  "fields_defaulted": 96,
  "adverse_action_reasons": [
    {"variable": "sub_grade", "value": "C3", "points_lost": 18.4},
    {"variable": "int_rate",  "value": "(11.99, 13.67]", "points_lost": 12.1}
  ],
  "model_trained_at": "2026-09-20T00:31:05"
}
```

Returning only `probability_of_default` pushes the hard part back to the caller,
who then invents a threshold. The break-even is computed from the loan's own
economics, so the recommendation is the model's job rather than the client's.

**`fields_defaulted` is not noise.** A caller is not asked to supply 105 bureau
attributes to score one application — anything omitted falls back to the training
median or modal value. That is a real modelling compromise with a real cost, so
the response states its size. Nine supplied fields and ninety-six defaults
deserves less trust than a full application, and the caller should be able to see
which they got.

**Adverse action reasons come from the scorecard**, not from a post-hoc
attribution of the booster. They are read directly out of the fitted points table:
for each selected variable, how many points this applicant lost against the best
band. That is the mechanism a decline letter actually needs, and it is auditable
because the table is a table.

### The API is not a back door

`Application.extra` is a free-form dict, which makes it the one place a
post-origination column could sneak into a feature matrix. There is a test:

```python
def test_no_post_origination_field_is_accepted_as_an_input(client):
    baseline = client.post("/score", json=TYPICAL).json()
    smuggled = client.post("/score", json=dict(
        TYPICAL, extra={"recoveries": 5000.0, "last_fico_range_high": 500.0})).json()

    assert (smuggled["probability_of_default"]
            == baseline["probability_of_default"])
```

If that ever fails, the contract has leaked through the serving layer.

---

## 2. Container

```bash
docker build -t credit-risk .
docker run -p 8000:8000 -v "$(pwd)/data/models:/app/data/models:ro" credit-risk
```

Two decisions worth stating.

**The model is not baked into the image.** It is built from 1.56 GB of source
data that has no business inside a container, and a model that ships inside its
own image cannot be retrained without a rebuild. It is mounted read-only at run
time.

**`/health` returns 503 when no model is mounted, rather than the process
crashing.** A container that dies because its artefact is missing is much harder
to diagnose in a cluster than one that starts and explains itself. CI asserts
exactly this: start the image with no model and require a response.

Only the serving dependencies are installed — streamlit, mlflow, matplotlib and
reportlab are development tools and would roughly triple the image. `libgomp1` is
installed explicitly because it is LightGBM's OpenMP runtime, and without it the
import fails at startup with a missing shared object.

> **Honest note:** Docker is not installed on the machine this was built on, so
> the image is built and smoke-tested by GitHub Actions rather than locally. The
> CI job is the evidence, not my word.

---

## 3. What CI actually checks

```yaml
- ruff check src tests app
- python -m src.columns          # the contract covers every source column
- pytest tests -v                # 45 tests
- docker build                   # plus a /health smoke test
```

The 1.56 GB file is **not** downloaded in CI, and tests that need a trained model
skip themselves. That is deliberate. CI checks the things that break silently:
the point-in-time contract, the WOE and PSI arithmetic, the expected-value
algebra, the API shape. Whether a 340-tree booster reaches 0.70 AUC is not
something a six-minute job on a shared runner should be asserting.

45 tests. The ones that matter most:

| Test | What it protects |
|---|---|
| `test_assert_no_leakage_rejects_a_dirty_feature_list` | the guard **actually fails** — a check that never fires is decoration |
| `test_check_coverage_rejects_an_unreviewed_column` | a new upstream column stops the pipeline |
| `test_bureau_columns_that_look_leaky_are_not` | guards against over-correcting |
| `test_woe_is_positive_for_a_safer_than_average_bin` | the scorecard sign error that inverts the whole table |
| `test_psi_uses_baseline_edges_not_recomputed_ones` | the most common way PSI is silently broken |
| `test_expected_value_flips_sign_exactly_at_breakeven` | the decision algebra |
| `test_a_worse_credit_file_scores_worse` | monotonicity where it is not negotiable |
| `test_no_post_origination_field_is_accepted_as_an_input` | the serving back door |

---

## 4. Monitoring

```bash
make monitor
```

Three views of the same five quarters. They disagree, which is the argument for
having all three.

### Score PSI — the inputs

Bin edges always come from the baseline. Re-deriving them on the new sample makes
the distributions agree by construction and reports near-zero drift no matter what
happened; it is the single most common way to get PSI wrong, and there is a test
for it.

```
2015Q1   0.0609  stable
2015Q2   0.0739  stable
2015Q3   0.0942  stable
2015Q4   0.1236  investigate
2016Q1   0.1031  investigate
```

The 0.10 and 0.25 bands are **industry convention, not theory**, and the code
says so rather than presenting them as laws.

Where a column is almost entirely null — several `sec_app_*` fields in the early
vintages — PSI returns `NaN` with a verdict of `"insufficient data"`. Returning
0.0 would report an empty column as perfectly *stable*.

### Characteristic analysis — which variable moved

```
initial_list_status       0.4904  shifted
mths_since_last_record    0.3225  shifted
int_rate                  0.2977  shifted
sub_grade                 0.0964  stable
```

The largest shift is `initial_list_status`, which records whether Lending Club
listed a loan whole or fractionally. That is **a change in their own platform
operations, not in borrower quality.** Retraining in response would be solving
the wrong problem — and that distinction is the entire reason per-variable PSI
exists rather than a single score-level number.

`int_rate` at 0.2977 is the one that should worry you: Lending Club repriced over
the period, and the model leans on that field.

### Performance by vintage — the answer

```
vintage      n     default%  predicted%   AUC
2015Q1   56568       14.83%      13.58%  0.6910
2015Q2   64222       15.38%      13.45%  0.6932
2015Q3   73567       14.58%      13.09%  0.7019
2015Q4   88669       14.82%      12.69%  0.7020
2016Q1   89784       16.10%      12.98%  0.7102
```

Read those three blocks together and the picture is unambiguous, and unavailable
from any one of them:

- PSI says the population is drifting → *retrain*
- AUC says discrimination is fine and improving → *do nothing*
- Calibration says under-pricing every quarter, gap widening 1.3 → 3.1 pp →
  **recalibrate**

The third is correct. Retraining on a model that still ranks well would be
expensive and would not fix the problem; recalibrating on the most recent closed
book would.

---

## 5. The promotion gate

Tolerances live in one place with reasons attached:

```python
AUC_TOLERANCE = 0.002        # within a fifth of a Gini point counts as equal
BRIER_TOLERANCE = 0.0005     # calibration must not degrade materially
PROFIT_TOLERANCE = 0.0       # profit is not allowed to go backwards at all
```

AUC gets a tolerance because a simpler, cheaper or more explainable model that
*matches* the champion is worth promoting. Profit gets none.

Running the scorecard as challenger:

```
[FAIL] discrimination        AUC 0.6892 vs 0.7007 (tolerance 0.0020)
[FAIL] calibration           Brier 0.12146 vs 0.12069 (tolerance 0.00050)
[FAIL] profit                332,131,241 vs 335,013,537
[PASS] population stability  0 vintage(s) flagged as shifted

decision: KEEP CHAMPION
blocked by: discrimination, calibration, profit
```

It fails, narrowly, on all three quality checks. **Failing is the expected
outcome most of the time — that is the point of having a gate rather than a
preference.** And the failure is informative rather than disappointing: the cost
of interpretability is now 0.012 AUC and 0.86% of profit on a $335m book, which
is a decision a business can make.

---

## 6. Experiment tracking

```bash
make mlflow    # sqlite-backed, local
```

Parameters, metrics for all four models, profit per policy, and `metrics.json` as
an artefact.

Backed by a local sqlite file rather than a bare `./mlruns` directory, because
MLflow 3 put the plain filesystem store into maintenance mode and **silently stops
recording**. The first training run logged nothing and reported it as a one-line
warning, which is how I found out.

The whole tracking call is wrapped in `try/except`: a broken tracking store must
not kill a training run that otherwise succeeded.

---

## 7. What is missing, honestly

- **No model registry or staged rollout.** The gate decides, a human deploys.
- **No shadow scoring.** A real deployment would run challenger alongside
  champion on live traffic before promoting.
- **No automated retraining schedule.** Monitoring detects the need; acting on it
  is manual.
- **No drift alerting.** PSI is computed and reported, not pushed anywhere.
- **Single pooled LGD in the cost model**, so the economics are directionally
  right rather than precise.
