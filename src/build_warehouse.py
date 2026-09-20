"""Load the raw CSV into DuckDB and carve out the modelling population.

Two tables come out of this:

    raw_loans   all 2.26M rows, all 151 columns, typed. Includes the leaky
                ones, on purpose -- the leakage experiment needs them.

    loans       the modelling population: resolved outcomes only, and only
                vintages old enough to have finished. Plus a handful of
                derived columns that are pure parsing rather than modelling
                choices, so they belong here rather than in features.py.

Why DuckDB rather than reading 1.6 GB into pandas: the population filter
throws away roughly half the rows, and there is no reason to pay for those in
memory. DuckDB streams the CSV, applies the predicate, and hands back only
what survives. Same instinct as pushing a filter into the warehouse instead of
pulling everything into the client.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import duckdb

from src import columns, config


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    config.ensure_dirs()
    return duckdb.connect(config.DUCKDB_PATH, read_only=read_only)


def _quoted_status_list(values) -> str:
    return ", ".join("'%s'" % v.replace("'", "''") for v in values)


def build_raw(con: duckdb.DuckDBPyConnection, csv_path: str) -> int:
    """Read the CSV into `raw_loans`.

    sample_size=-1 makes DuckDB scan the whole file before deciding types.
    That costs a pass over 1.6 GB and it is worth it: with a small sample it
    guesses INTEGER for columns whose first few hundred thousand rows happen to
    be whole numbers, then fails on a decimal 900k rows in.
    """
    print("reading %s" % csv_path)
    started = time.time()
    con.execute("DROP TABLE IF EXISTS raw_loans")
    con.execute(
        """
        CREATE TABLE raw_loans AS
        SELECT * FROM read_csv(
            ?,
            header = true,
            sample_size = -1,
            ignore_errors = false
        )
        """,
        [csv_path],
    )
    n = con.execute("SELECT count(*) FROM raw_loans").fetchone()[0]
    print("  raw_loans: %d rows in %.1fs" % (n, time.time() - started))
    return n


def verify_contract(con: duckdb.DuckDBPyConnection) -> None:
    """Every source column must be classified before anything downstream runs."""
    names = [r[0] for r in con.execute("DESCRIBE raw_loans").fetchall()]
    columns.check_coverage(names)
    print("  contract: %d columns, all classified" % len(names))


def build_population(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Apply the two population rules and report what each one removed.

    Reporting the attrition matters. "I filtered the data" is unverifiable;
    "the maturity rule removed 431,812 rows, all of them 2016-2018 vintages"
    is something a reviewer can argue with.
    """
    good = _quoted_status_list(config.RESOLVED_GOOD)
    bad = _quoted_status_list(config.RESOLVED_BAD)

    con.execute("DROP TABLE IF EXISTS loans")
    con.execute(
        f"""
        CREATE TABLE loans AS
        WITH parsed AS (
            SELECT
                *,
                -- issue_d arrives as 'Dec-2015'
                strptime(issue_d, '%b-%Y')::DATE                    AS issue_date,
                -- term arrives as ' 36 months'
                CAST(regexp_extract(term, '(\\d+)', 1) AS INTEGER)   AS term_months,
                strptime(earliest_cr_line, '%b-%Y')::DATE           AS first_credit_date
            FROM raw_loans
            WHERE loan_status IN ({good}, {bad})
        )
        SELECT
            *,
            CASE WHEN loan_status IN ({bad}) THEN 1 ELSE 0 END      AS {config.TARGET},
            date_trunc('month', issue_date)                          AS vintage_month,
            year(issue_date)                                         AS vintage_year,
            quarter(issue_date)                                      AS vintage_quarter,
            -- Months of credit history at application. A raw date is useless to
            -- a model; its distance from the application date is the feature.
            datediff('month', first_credit_date, issue_date)
                                                     AS credit_history_months,
            -- The scheduled final payment date, used by the maturity rule.
            (issue_date + INTERVAL (term_months) MONTH)::DATE
                                                     AS scheduled_end_date
        FROM parsed
        WHERE (issue_date + INTERVAL (term_months) MONTH) <= DATE '{config.SNAPSHOT}'
        """
    )

    stats = {}
    stats["raw"] = con.execute("SELECT count(*) FROM raw_loans").fetchone()[0]
    stats["resolved"] = con.execute(
        f"SELECT count(*) FROM raw_loans WHERE loan_status IN ({good}, {bad})"
    ).fetchone()[0]
    stats["population"] = con.execute("SELECT count(*) FROM loans").fetchone()[0]
    stats["charged_off"] = con.execute(
        f"SELECT sum({config.TARGET}) FROM loans"
    ).fetchone()[0]
    return stats


def report(con: duckdb.DuckDBPyConnection, stats: dict[str, int]) -> None:
    raw = stats["raw"]
    resolved = stats["resolved"]
    pop = stats["population"]
    bad = stats["charged_off"]

    print()
    print("population funnel")
    print("  all loans in file                 %9d" % raw)
    print("  resolved outcome only             %9d   (-%d still in flight)"
          % (resolved, raw - resolved))
    print("  had full term before snapshot     %9d   (-%d not yet matured)"
          % (pop, resolved - pop))
    print("  charged off                       %9d   (%.2f%% default rate)"
          % (bad, 100.0 * bad / pop))

    print()
    print("what the maturity rule removed, by vintage year")
    rows = con.execute(
        f"""
        WITH resolved AS (
            SELECT
                year(strptime(issue_d, '%b-%Y')::DATE) AS y,
                CASE
                    WHEN (strptime(issue_d, '%b-%Y')::DATE
                          + INTERVAL (CAST(regexp_extract(term, '(\\d+)', 1)
                                           AS INTEGER)) MONTH)
                         <= DATE '{config.SNAPSHOT}'
                    THEN 1 ELSE 0
                END AS matured
            FROM raw_loans
            WHERE loan_status IN ({_quoted_status_list(config.RESOLVED_GOOD)},
                                  {_quoted_status_list(config.RESOLVED_BAD)})
        )
        SELECT y, sum(matured) AS kept, sum(1 - matured) AS dropped, count(*) AS total
        FROM resolved
        GROUP BY y
        ORDER BY y
        """
    ).fetchall()
    print("  year      kept   dropped     total   kept%")
    for y, kept, dropped, total in rows:
        print("  %4d  %8d  %8d  %8d  %5.1f%%"
              % (y, kept, dropped, total, 100.0 * kept / total))

    print()
    print("default rate by vintage year, modelling population")
    rows = con.execute(
        f"""
        SELECT vintage_year, count(*) AS n, avg({config.TARGET}) AS rate
        FROM loans GROUP BY vintage_year ORDER BY vintage_year
        """
    ).fetchall()
    for y, n, rate in rows:
        bar = "#" * int(round(rate * 120))
        print("  %4d  %8d  %5.2f%%  %s" % (y, n, 100 * rate, bar))

    print()
    print("out-of-time split")
    for label, clause in (
        ("train (to %s)" % config.TRAIN_VINTAGE_END,
         "issue_date <= DATE '%s'" % config.TRAIN_VINTAGE_END),
        ("test  (from %s)" % config.TEST_VINTAGE_START,
         "issue_date >= DATE '%s'" % config.TEST_VINTAGE_START),
    ):
        n, rate = con.execute(
            f"SELECT count(*), avg({config.TARGET}) FROM loans WHERE {clause}"
        ).fetchone()
        print("  %-22s %8d rows   %5.2f%% default" % (label, n, 100 * (rate or 0)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=config.RAW_CSV_PATH)
    ap.add_argument("--skip-raw", action="store_true",
                    help="reuse an existing raw_loans table")
    args = ap.parse_args(argv)

    if not os.path.exists(args.csv):
        raise SystemExit("no CSV at %s -- run `python -m src.get_data` first" % args.csv)

    con = connect()
    try:
        if args.skip_raw:
            n = con.execute("SELECT count(*) FROM raw_loans").fetchone()[0]
            print("reusing raw_loans (%d rows)" % n)
        else:
            build_raw(con, args.csv)
        verify_contract(con)
        stats = build_population(con)
        report(con, stats)
    finally:
        con.close()

    print()
    print("warehouse: %s" % config.DUCKDB_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
