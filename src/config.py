"""Paths, constants and the modelling population definition.

Everything that another module might want to hard-code lives here instead, so
there is exactly one place to change a path or a cutoff date.

The population rules in this file are the most consequential decisions in the
whole project, so they are argued for rather than just stated. See
docs/decision-log.md for the longer version.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
WAREHOUSE_DIR = os.path.join(DATA_DIR, "warehouse")
MODEL_DIR = os.path.join(DATA_DIR, "models")
REPORT_DIR = os.path.join(DATA_DIR, "reports")

DOCS_DIR = os.path.join(ROOT, "docs")
ASSET_DIR = os.path.join(DOCS_DIR, "assets")

DUCKDB_PATH = os.path.join(WAREHOUSE_DIR, "credit.duckdb")

RAW_CSV_NAME = "accepted_2007_to_2018Q4.csv"
RAW_CSV_PATH = os.path.join(RAW_DIR, RAW_CSV_NAME)


def ensure_dirs() -> None:
    """Create every directory the pipeline writes to.

    Called at the top of each entry point rather than at import time, because
    importing a config module should not have side effects on the filesystem.
    """
    for path in (RAW_DIR, WAREHOUSE_DIR, MODEL_DIR, REPORT_DIR, ASSET_DIR):
        os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Source data
# ---------------------------------------------------------------------------

# Lending Club's accepted-loan book, 2007 through 2018 Q4. Lending Club took
# their own download page offline years ago, so this mirror is what is actually
# reachable. I verified it serves HTTP 206 ranged requests, which is what makes
# the resumable download in data/get_data.py possible.
SOURCE_URL = (
    "https://huggingface.co/datasets/codesignal/lending-club-loan-accepted"
    "/resolve/main/accepted_2007_to_2018Q4.csv"
)

# Byte length of the file as served. Checked at download time: if the remote
# file is ever replaced, the size check fails loudly instead of the pipeline
# quietly training on different data than the numbers in the README describe.
SOURCE_BYTES = 1_675_133_810

SEED = 20260919


# ---------------------------------------------------------------------------
# The modelling population
# ---------------------------------------------------------------------------

# A loan's outcome is only known once it has finished. These four statuses are
# terminal. The two "does not meet the credit policy" variants are loans LC
# issued under older, looser rules and later disowned; they are still genuine
# resolved outcomes, so they stay in.
RESOLVED_GOOD = [
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
]
RESOLVED_BAD = [
    "Charged Off",
    "Does not meet the credit policy. Status:Charged Off",
]

# Everything else -- Current, Late (16-30), Late (31-120), In Grace Period,
# Default -- describes a loan still in flight. Its eventual outcome is unknown,
# so it cannot be labelled. Dropping these rows is not optional.
#
# "Default" looks like it belongs in RESOLVED_BAD, and a lot of public
# notebooks put it there. It does not belong: in LC's vocabulary Default is a
# transitional state on the way to either Charged Off or recovery, not an
# endpoint.

# The file is a snapshot taken after 2018 Q4. A 60-month loan issued in
# mid-2018 therefore cannot possibly appear as Fully Paid -- it has not had
# time. Keeping only resolved loans without also enforcing maturity quietly
# biases the sample: recent vintages survive only if they defaulted early.
#
# So a loan enters the population only if it had its full term to finish
# before the snapshot. This throws away a lot of 2016-2018 rows, which is
# painful, and it is still the right call.
SNAPSHOT = "2019-03-01"

TARGET = "is_charged_off"
DATE_COL = "issue_d"

# Out-of-time split. Credit models are always deployed on vintages that did
# not exist at training time, so a random split measures the wrong thing. The
# boundary is a date, not a percentage.
TRAIN_VINTAGE_END = "2014-12-31"
TEST_VINTAGE_START = "2015-01-01"
