"""Fetch the Lending Club loan book.

The file is 1.56 GB, which means two things. It has to resume -- a download
that restarts from zero on a dropped connection is useless -- and it has to be
verifiable, because if the mirror ever changes content underneath me then every
number in the README becomes a lie without anything failing.

Usage:
    python -m src.get_data              # full 1.56 GB, resumable
    python -m src.get_data --rows 50000 # quick slice, for CI and smoke tests
    python -m src.get_data --verify     # checksum an existing download
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

from src import config

CHUNK = 16 * 1024 * 1024
UA = "credit-risk-underwriting/1.0 (portfolio project)"
MANIFEST = os.path.join(config.RAW_DIR, "download_manifest.json")


def _open_range(url: str, start: int, end: int | None = None):
    """Request a byte range. `end` is inclusive, per RFC 7233."""
    req = urllib.request.Request(url)
    req.add_header("User-Agent", UA)
    rng = "bytes=%d-" % start if end is None else "bytes=%d-%d" % (start, end)
    req.add_header("Range", rng)
    return urllib.request.urlopen(req, timeout=300)


def remote_size(url: str) -> int:
    """Total length of the remote file.

    Asking for a single byte and reading Content-Range is more reliable than a
    HEAD request here: plenty of CDNs answer HEAD with no length at all.
    """
    with _open_range(url, 0, 0) as r:
        crange = r.headers.get("Content-Range", "")
    if "/" not in crange:
        raise RuntimeError("server did not report a total size: %r" % crange)
    return int(crange.rsplit("/", 1)[1])


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return "%.2f %s" % (n, unit)
        n /= 1024
    return "%.2f GB" % n


def download(url: str, dest: str, expect_bytes: int | None, max_bytes: int | None) -> int:
    """Download `url` to `dest`, resuming if a partial file is already there.

    Returns the number of bytes on disk when finished.
    """
    total = remote_size(url)
    print("remote size: %s (%d bytes)" % (human(total), total))

    if expect_bytes is not None and total != expect_bytes:
        raise SystemExit(
            "Remote file is %d bytes but config.SOURCE_BYTES says %d.\n"
            "The mirror has changed. Every metric in the docs was measured "
            "against the old file, so I am stopping rather than silently "
            "training on something else." % (total, expect_bytes)
        )

    target = total if max_bytes is None else min(max_bytes, total)
    have = os.path.getsize(dest) if os.path.exists(dest) else 0

    if have >= target:
        print("already have %s on disk, nothing to do" % human(have))
        return have
    if have:
        print("resuming from %s" % human(have))

    started = time.time()
    last_report = 0.0
    attempts = 0

    with open(dest, "ab") as fh:
        while have < target:
            end = min(have + CHUNK, target) - 1
            try:
                with _open_range(url, have, end) as r:
                    block = r.read()
                attempts = 0
            except (urllib.error.URLError, TimeoutError) as exc:
                attempts += 1
                if attempts >= 5:
                    raise
                wait = 2 ** attempts
                print("  chunk at %s failed (%s), retrying in %ds"
                      % (human(have), str(exc)[:50], wait))
                time.sleep(wait)
                continue

            if not block:
                print("  server returned an empty chunk, stopping early")
                break

            fh.write(block)
            have += len(block)

            now = time.time()
            if now - last_report > 5 or have >= target:
                elapsed = max(now - started, 0.001)
                pct = 100.0 * have / target
                rate = have / elapsed
                print("  %5.1f%%  %s / %s  (%s/s)"
                      % (pct, human(have), human(target), human(rate)))
                last_report = now

    return have


def checksum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def trim_to_rows(path: str, rows: int) -> None:
    """Cut a partially downloaded file back to a whole number of CSV lines.

    A byte-range download almost always stops mid-line. Pandas would either
    throw or, worse, parse the fragment as a short row.
    """
    keep = []
    with open(path, "rb") as fh:
        for i, line in enumerate(fh):
            if i > rows:
                break
            keep.append(line)
    # The last line read may itself be truncated, so drop it unless the file
    # genuinely ended with a newline.
    if keep and not keep[-1].endswith(b"\n"):
        keep.pop()
    with open(path, "wb") as fh:
        fh.writelines(keep)
    print("trimmed to %d lines (1 header + %d rows)" % (len(keep), len(keep) - 1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=None,
                    help="stop after roughly this many data rows (smoke tests)")
    ap.add_argument("--verify", action="store_true",
                    help="checksum what is already on disk and exit")
    ap.add_argument("--force", action="store_true",
                    help="delete any existing file and start over")
    args = ap.parse_args(argv)

    config.ensure_dirs()
    dest = config.RAW_CSV_PATH

    if args.verify:
        if not os.path.exists(dest):
            print("nothing at %s" % dest)
            return 1
        print("size:   %s" % human(os.path.getsize(dest)))
        print("sha256: %s" % checksum(dest))
        return 0

    if args.force and os.path.exists(dest):
        os.remove(dest)
        print("removed existing file")

    if args.rows:
        # Average row is a bit over 400 bytes across 151 columns. Pulling 800
        # per row gives comfortable headroom, then we trim back exactly.
        budget = 4096 + args.rows * 800
        got = download(config.SOURCE_URL, dest, None, budget)
        trim_to_rows(dest, args.rows)
        got = os.path.getsize(dest)
        expect = None
    else:
        got = download(config.SOURCE_URL, dest, config.SOURCE_BYTES, None)
        expect = config.SOURCE_BYTES
        if got != expect:
            print("WARNING: have %d bytes, expected %d" % (got, expect))
            return 1

    digest = checksum(dest)
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "url": config.SOURCE_URL,
                "bytes": got,
                "sha256": digest,
                "rows_requested": args.rows,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            fh,
            indent=2,
        )

    print()
    print("wrote   %s" % dest)
    print("size    %s" % human(got))
    print("sha256  %s" % digest)
    print("manifest %s" % MANIFEST)
    return 0


if __name__ == "__main__":
    sys.exit(main())
