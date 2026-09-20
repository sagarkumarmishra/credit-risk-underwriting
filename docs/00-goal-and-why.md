# What I set out to prove

Short version: that the data engineering I have been doing for years is the
reason I can be trusted with a credit model, not a detour on the way to one.

## The problem with the obvious version of this project

"Predict loan defaults with machine learning" is one of the most common portfolio
projects there is. Most of them are worthless, and they are worthless in a way
that is easy to describe.

Lending Club publishes 151 columns per loan. Thirty-eight of them are recorded
*after* the money goes out: how much has been repaid, when the last payment
landed, whether the loan went into a hardship plan, how much was recovered after
charge-off, the borrower's credit score at the most recent bureau pull. A model
trained on those reports an AUC somewhere between 0.95 and 0.99 and could not
score a single real application, because at the moment an underwriter decides,
every one of those fields is empty.

That is not a modelling error. It is a **lineage** error: the feature was
recorded downstream of the event it claims to predict. Which is the kind of
mistake a data engineer is trained to see and a pure modeller often is not,
because from inside a dataframe `last_fico_range_high` and `fico_range_high` look
like the same sort of thing.

So the project is built around that: **the DE background is the reason the trap
gets avoided, and the ML is the bulk of the work.**

## Three claims, and what happened to each

**1. Post-origination columns are worth a third of an AUC.**

Held, harder than I expected. The careless configuration scores a *perfect*
1.0000 — not the 0.99 I guessed — against 0.7083 for the honest one. Leakage
accounts for +0.2916 of the +0.2917 total gap.

**2. A random train/test split inflates the score too.**

**Failed.** It contributed −0.0042; the random split was marginally *harder*.
The mechanism is the maturity rule this project introduces: once loans are
required to have had their full term, 2017 and 2018 vanish and every remaining
year defaults at 14–16%, so there is almost no temporal drift left to exploit.
Kept in the write-up with the mechanism, because one claim holding while the
other fails is more credible than two tidy confirmations.

**3. AUC is not the deliverable; money is.**

Held, and more sharply than planned. Youden's J — the statistically optimal
cutoff — destroys **$239 of value per application**. Meanwhile the industry
default of `p > 0.50` turns out to approve the entire book, because not one
application in 372,810 scores above 0.5. It is not a conservative threshold, it
is not a threshold.

## What the project deliberately does not do

- It does not chase AUC. 0.7007 is roughly what an honest application scorecard
  on pre-screened consumer loans should look like. A submission claiming 0.95
  would be evidence of a bug.
- It does not report accuracy. At a 15% default rate, "everybody repays" scores
  85%.
- It does not use geography, even though `zip_code` predicts well, because ECOA
  treats it as a proxy for race.
- It does not pretend the profit lift is large. It is 1.1%, because every loan in
  this dataset was already approved by Lending Club's own underwriting and the
  easy declines are not in the data.

## Where to look

| If you want | Read |
|---|---|
| the findings in four pages | [credit-risk-summary.pdf](credit-risk-summary.pdf) |
| the modelling in detail | [modelling.md](modelling.md) |
| serving, monitoring, promotion | [mlops.md](mlops.md) |
| every judgement call, including the wrong ones | [decision-log.md](decision-log.md) |
| the single most important file | [`src/columns.py`](../src/columns.py) |
