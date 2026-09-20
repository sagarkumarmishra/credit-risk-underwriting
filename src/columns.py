"""The point-in-time contract: which of the 151 source columns may a model see?

This is the most important file in the project, so it is worth being blunt
about why it exists.

An underwriting model answers a question asked at one specific instant: the
moment before the money goes out. At that instant, nobody knows how much of
the loan was eventually repaid, whether the borrower entered a hardship plan,
or what their credit score looked like two years later. Every one of those
facts is in this dataset, sitting in a column with an innocent name.

Train on them and you get a model with an AUC around 0.99 that cannot be
deployed, because in production those columns are empty. This is not a subtle
statistical issue, it is a lineage issue: the feature was recorded downstream
of the event it claims to predict. Coming from data engineering, that framing
is second nature -- it is the same discipline as refusing to join a fact table
to a dimension snapshot taken after the fact.

So rather than hand-pick a feature list and hope, every single source column is
classified here with a written reason, and COVERAGE is asserted. If Lending
Club's mirror ever gains a column, the pipeline stops instead of quietly
letting an unreviewed field into the model.

Categories
    ORIGINATION  known at the moment of underwriting; a model may use it
    OUTCOME      recorded during or after servicing; using it is leakage
    LABEL        the thing being predicted
    META         needed for splitting or auditing, never a feature
    DROP         constant, empty, free text, or an identifier
    SENSITIVE    known at origination, but excluded on fair-lending grounds
"""

from __future__ import annotations

ORIGINATION = "ORIGINATION"
OUTCOME = "OUTCOME"
LABEL = "LABEL"
META = "META"
DROP = "DROP"
SENSITIVE = "SENSITIVE"

# column -> (category, why)
CONTRACT: dict[str, tuple[str, str]] = {
    # -- identifiers and dead weight ------------------------------------------
    "id": (DROP, "surrogate key, carries no signal"),
    "member_id": (DROP, "fully null in this extract, LC scrubbed it"),
    "url": (DROP, "link to the LC listing page"),
    "desc": (DROP, "borrower free text, almost entirely null after 2013"),
    "emp_title": (DROP, "free text, ~500k distinct values, no clean encoding"),
    "title": (DROP, "borrower's own label for the loan, duplicates `purpose`"),
    "policy_code": (DROP, "constant 1 across the whole file"),

    # -- the label and the clock ----------------------------------------------
    "loan_status": (LABEL, "the outcome being predicted"),
    "issue_d": (META, "origination month; defines the out-of-time split"),

    # -- terms of the loan, set at origination --------------------------------
    "loan_amnt": (ORIGINATION, "amount the borrower asked for"),
    "funded_amnt": (ORIGINATION, "amount committed at issuance"),
    "funded_amnt_inv": (ORIGINATION, "investor-funded portion at issuance"),
    "term": (ORIGINATION, "36 or 60 months, fixed up front"),
    "int_rate": (ORIGINATION, "priced at origination"),
    "installment": (ORIGINATION, "monthly payment, arithmetic from amount/rate/term"),
    "grade": (ORIGINATION, "LC's own risk grade at listing"),
    "sub_grade": (ORIGINATION, "finer version of grade"),
    "initial_list_status": (ORIGINATION, "whole vs fractional listing"),
    "application_type": (ORIGINATION, "individual or joint"),
    "disbursement_method": (ORIGINATION, "cash or direct pay, chosen at signing"),
    "purpose": (ORIGINATION, "borrower's stated use of funds"),

    # -- borrower profile as stated or verified at application ----------------
    "emp_length": (ORIGINATION, "years employed, self-reported at application"),
    "home_ownership": (ORIGINATION, "rent / own / mortgage at application"),
    "annual_inc": (ORIGINATION, "stated income at application"),
    "verification_status": (ORIGINATION, "whether LC verified that income"),
    "dti": (ORIGINATION, "debt-to-income computed at application"),
    "annual_inc_joint": (ORIGINATION, "joint income, only for joint applications"),
    "dti_joint": (ORIGINATION, "joint DTI, only for joint applications"),
    "verification_status_joint": (ORIGINATION, "verification of joint income"),

    # -- credit bureau pull taken at application ------------------------------
    # Everything below is a bureau attribute as of the application pull. They
    # look alarmingly like outcome data ("chargeoff_within_12_mths") but refer
    # to the borrower's OTHER accounts before this loan existed.
    "fico_range_low": (ORIGINATION, "FICO band floor at application"),
    "fico_range_high": (ORIGINATION, "FICO band ceiling at application"),
    "earliest_cr_line": (ORIGINATION, "first ever credit line; gives history length"),
    "delinq_2yrs": (ORIGINATION, "delinquencies in 24 months before applying"),
    "inq_last_6mths": (ORIGINATION, "credit enquiries before applying"),
    "mths_since_last_delinq": (ORIGINATION, "recency of past delinquency"),
    "mths_since_last_record": (ORIGINATION, "recency of public record"),
    "open_acc": (ORIGINATION, "open credit lines at application"),
    "pub_rec": (ORIGINATION, "derogatory public records"),
    "revol_bal": (ORIGINATION, "revolving balance"),
    "revol_util": (ORIGINATION, "revolving utilisation"),
    "total_acc": (ORIGINATION, "total credit lines ever"),
    "collections_12_mths_ex_med": (ORIGINATION, "collections in prior 12 months"),
    "mths_since_last_major_derog": (ORIGINATION, "recency of major derogatory"),
    "acc_now_delinq": (ORIGINATION, "accounts currently delinquent at application"),
    "tot_coll_amt": (ORIGINATION, "total amount ever in collections"),
    "tot_cur_bal": (ORIGINATION, "total current balance, all accounts"),
    "open_acc_6m": (ORIGINATION, "accounts opened in last 6 months"),
    "open_act_il": (ORIGINATION, "active instalment accounts"),
    "open_il_12m": (ORIGINATION, "instalment accounts opened, 12 months"),
    "open_il_24m": (ORIGINATION, "instalment accounts opened, 24 months"),
    "mths_since_rcnt_il": (ORIGINATION, "recency of instalment account"),
    "total_bal_il": (ORIGINATION, "instalment balance"),
    "il_util": (ORIGINATION, "instalment utilisation"),
    "open_rv_12m": (ORIGINATION, "revolving accounts opened, 12 months"),
    "open_rv_24m": (ORIGINATION, "revolving accounts opened, 24 months"),
    "max_bal_bc": (ORIGINATION, "highest bankcard balance"),
    "all_util": (ORIGINATION, "utilisation across all accounts"),
    "total_rev_hi_lim": (ORIGINATION, "revolving high credit limit"),
    "inq_fi": (ORIGINATION, "finance enquiries"),
    "total_cu_tl": (ORIGINATION, "credit union trades"),
    "inq_last_12m": (ORIGINATION, "enquiries in last 12 months"),
    "acc_open_past_24mths": (ORIGINATION, "accounts opened in 24 months"),
    "avg_cur_bal": (ORIGINATION, "average balance across accounts"),
    "bc_open_to_buy": (ORIGINATION, "bankcard headroom"),
    "bc_util": (ORIGINATION, "bankcard utilisation"),
    "chargeoff_within_12_mths": (ORIGINATION,
                                 "charge-offs on OTHER accounts pre-application"),
    "delinq_amnt": (ORIGINATION, "amount currently delinquent elsewhere"),
    "mo_sin_old_il_acct": (ORIGINATION, "age of oldest instalment account"),
    "mo_sin_old_rev_tl_op": (ORIGINATION, "age of oldest revolving account"),
    "mo_sin_rcnt_rev_tl_op": (ORIGINATION, "recency of revolving account opening"),
    "mo_sin_rcnt_tl": (ORIGINATION, "recency of any account opening"),
    "mort_acc": (ORIGINATION, "mortgage accounts held"),
    "mths_since_recent_bc": (ORIGINATION, "recency of bankcard opening"),
    "mths_since_recent_bc_dlq": (ORIGINATION, "recency of bankcard delinquency"),
    "mths_since_recent_inq": (ORIGINATION, "recency of enquiry"),
    "mths_since_recent_revol_delinq": (ORIGINATION, "recency of revolving delinquency"),
    "num_accts_ever_120_pd": (ORIGINATION, "accounts ever 120 days past due"),
    "num_actv_bc_tl": (ORIGINATION, "active bankcard trades"),
    "num_actv_rev_tl": (ORIGINATION, "active revolving trades"),
    "num_bc_sats": (ORIGINATION, "satisfactory bankcard accounts"),
    "num_bc_tl": (ORIGINATION, "bankcard trades"),
    "num_il_tl": (ORIGINATION, "instalment trades"),
    "num_op_rev_tl": (ORIGINATION, "open revolving trades"),
    "num_rev_accts": (ORIGINATION, "revolving accounts"),
    "num_rev_tl_bal_gt_0": (ORIGINATION, "revolving trades carrying a balance"),
    "num_sats": (ORIGINATION, "satisfactory accounts"),
    "num_tl_120dpd_2m": (ORIGINATION, "trades 120 dpd in last 2 months"),
    "num_tl_30dpd": (ORIGINATION, "trades 30 dpd"),
    "num_tl_90g_dpd_24m": (ORIGINATION, "trades 90+ dpd in 24 months"),
    "num_tl_op_past_12m": (ORIGINATION, "trades opened in 12 months"),
    "pct_tl_nvr_dlq": (ORIGINATION, "share of trades never delinquent"),
    "percent_bc_gt_75": (ORIGINATION, "share of bankcards over 75% utilised"),
    "pub_rec_bankruptcies": (ORIGINATION, "bankruptcies on record"),
    "tax_liens": (ORIGINATION, "tax liens on record"),
    "tot_hi_cred_lim": (ORIGINATION, "total high credit limit"),
    "total_bal_ex_mort": (ORIGINATION, "balance excluding mortgage"),
    "total_bc_limit": (ORIGINATION, "total bankcard limit"),
    "total_il_high_credit_limit": (ORIGINATION, "instalment high credit limit"),
    "revol_bal_joint": (ORIGINATION, "joint revolving balance"),
    "sec_app_fico_range_low": (ORIGINATION, "co-applicant FICO floor at application"),
    "sec_app_fico_range_high": (ORIGINATION, "co-applicant FICO ceiling at application"),
    "sec_app_earliest_cr_line": (ORIGINATION, "co-applicant credit history start"),
    "sec_app_inq_last_6mths": (ORIGINATION, "co-applicant enquiries"),
    "sec_app_mort_acc": (ORIGINATION, "co-applicant mortgages"),
    "sec_app_open_acc": (ORIGINATION, "co-applicant open accounts"),
    "sec_app_revol_util": (ORIGINATION, "co-applicant revolving utilisation"),
    "sec_app_open_act_il": (ORIGINATION, "co-applicant active instalment accounts"),
    "sec_app_num_rev_accts": (ORIGINATION, "co-applicant revolving accounts"),
    "sec_app_chargeoff_within_12_mths": (ORIGINATION,
                                         "co-applicant charge-offs pre-application"),
    "sec_app_collections_12_mths_ex_med": (ORIGINATION,
                                           "co-applicant collections pre-application"),
    "sec_app_mths_since_last_major_derog": (ORIGINATION,
                                            "co-applicant derogatory recency"),

    # -- excluded on fair-lending grounds, not on statistical ones ------------
    # Both are genuinely known at application and both carry real predictive
    # signal. They are still out. US fair-lending law (ECOA, Regulation B)
    # treats geography as a potential proxy for protected characteristics, and
    # three-digit ZIP in particular maps closely onto racial composition. A
    # model that declines people for living in the wrong postcode is a
    # redlining problem regardless of its AUC. I measure what excluding them
    # costs -- see docs/modelling.md -- rather than pretending it is free.
    "zip_code": (SENSITIVE, "3-digit ZIP; recognised proxy for race under ECOA"),
    "addr_state": (SENSITIVE, "state; weaker proxy, excluded for the same reason"),

    # -- recorded after the money went out: leakage --------------------------
    "pymnt_plan": (OUTCOME, "payment plan is a servicing event"),
    "out_prncp": (OUTCOME, "principal still outstanding today"),
    "out_prncp_inv": (OUTCOME, "investor share of outstanding principal"),
    "total_pymnt": (OUTCOME, "total received to date"),
    "total_pymnt_inv": (OUTCOME, "total received to date, investor share"),
    "total_rec_prncp": (OUTCOME, "principal received to date"),
    "total_rec_int": (OUTCOME, "interest received to date"),
    "total_rec_late_fee": (OUTCOME, "late fees charged during servicing"),
    "recoveries": (OUTCOME, "post-charge-off recovery; non-zero only if it defaulted"),
    "collection_recovery_fee": (OUTCOME, "fee on those recoveries"),
    "last_pymnt_d": (OUTCOME, "date of most recent payment"),
    "last_pymnt_amnt": (OUTCOME, "amount of most recent payment"),
    "next_pymnt_d": (OUTCOME, "scheduled next payment; null once the loan ends"),
    "last_credit_pull_d": (OUTCOME, "date LC last pulled bureau data"),
    # The two below are the ones worth staring at. They are not payment fields
    # and they sit in the middle of a block of legitimate FICO columns. They
    # are the borrower's score at the MOST RECENT pull, which for a defaulted
    # loan happened after the default wrecked their score. Sorted by name they
    # land next to fico_range_high, which is exactly how they get used by
    # accident.
    "last_fico_range_high": (OUTCOME, "FICO at latest pull, i.e. after the outcome"),
    "last_fico_range_low": (OUTCOME, "FICO at latest pull, i.e. after the outcome"),
    "hardship_flag": (OUTCOME, "hardship programmes are a servicing intervention"),
    "hardship_type": (OUTCOME, "hardship detail, post-origination"),
    "hardship_reason": (OUTCOME, "hardship detail, post-origination"),
    "hardship_status": (OUTCOME, "hardship detail, post-origination"),
    "deferral_term": (OUTCOME, "hardship detail, post-origination"),
    "hardship_amount": (OUTCOME, "hardship detail, post-origination"),
    "hardship_start_date": (OUTCOME, "hardship detail, post-origination"),
    "hardship_end_date": (OUTCOME, "hardship detail, post-origination"),
    "payment_plan_start_date": (OUTCOME, "hardship detail, post-origination"),
    "hardship_length": (OUTCOME, "hardship detail, post-origination"),
    "hardship_dpd": (OUTCOME, "days past due during hardship"),
    "hardship_loan_status": (OUTCOME, "loan status during hardship; near-copy of label"),
    "orig_projected_additional_accrued_interest": (OUTCOME, "hardship projection"),
    "hardship_payoff_balance_amount": (OUTCOME, "hardship projection"),
    "hardship_last_payment_amount": (OUTCOME, "hardship projection"),
    "debt_settlement_flag": (OUTCOME, "settlement only happens after distress"),
    "debt_settlement_flag_date": (OUTCOME, "settlement detail"),
    "settlement_status": (OUTCOME, "settlement detail"),
    "settlement_date": (OUTCOME, "settlement detail"),
    "settlement_amount": (OUTCOME, "settlement detail"),
    "settlement_percentage": (OUTCOME, "settlement detail"),
    "settlement_term": (OUTCOME, "settlement detail"),
}

# Columns a model is allowed to see. SENSITIVE is deliberately absent.
MODELLABLE = tuple(c for c, (cat, _) in CONTRACT.items() if cat == ORIGINATION)
LEAKY = tuple(c for c, (cat, _) in CONTRACT.items() if cat == OUTCOME)
SENSITIVE_COLS = tuple(c for c, (cat, _) in CONTRACT.items() if cat == SENSITIVE)

# Shortlist used for the leakage demonstration. These are the columns a
# careless pipeline picks up because they are numeric, dense and named like
# credit attributes. Ranked roughly by how badly each one gives the game away.
WORST_OFFENDERS = (
    "recoveries",
    "collection_recovery_fee",
    "last_fico_range_high",
    "last_fico_range_low",
    "total_rec_prncp",
    "out_prncp",
    "total_pymnt",
    "last_pymnt_amnt",
    "debt_settlement_flag",
)


def check_coverage(source_columns) -> None:
    """Fail if the source and the contract disagree.

    This is the guard rail. An unreviewed column must never reach a model just
    because somebody upstream added it.
    """
    source = set(source_columns)
    known = set(CONTRACT)

    unreviewed = sorted(source - known)
    if unreviewed:
        raise AssertionError(
            "%d source column(s) are not in the point-in-time contract: %s\n"
            "Classify each one in src/columns.py before it can be used."
            % (len(unreviewed), ", ".join(unreviewed))
        )

    missing = sorted(known - source)
    if missing:
        raise AssertionError(
            "%d contracted column(s) are absent from the source: %s\n"
            "The extract has changed shape; the contract needs revisiting."
            % (len(missing), ", ".join(missing))
        )


def assert_no_leakage(feature_names) -> None:
    """Refuse to train if an OUTCOME column made it into the feature matrix.

    Called from the training code, and again from the test suite. A comment
    saying "do not use these" is a wish; this is a check.
    """
    banned = set(LEAKY)
    found = sorted(f for f in feature_names if f in banned)
    if found:
        raise AssertionError(
            "post-origination column(s) reached the feature matrix: %s\n"
            "These are unknown at underwriting time. See src/columns.py."
            % ", ".join(found)
        )


def summary() -> dict[str, int]:
    counts: dict[str, int] = {}
    for cat, _ in CONTRACT.values():
        counts[cat] = counts.get(cat, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    print("point-in-time contract")
    for cat, n in summary().items():
        print("  %-12s %3d" % (cat, n))
    print("  %-12s %3d" % ("TOTAL", len(CONTRACT)))
