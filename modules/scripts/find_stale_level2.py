"""
find_stale_level2.py

Find obsids whose Level-2 files were last modified before a user-specified date.

Usage:
    python find_stale_level2.py 2024-01-01
    python find_stale_level2.py 2024-01-01 --db databases/COMAP_manchester.db
    python find_stale_level2.py 2024-01-01 --output stale_obsids.txt
"""

import argparse
import os
import sys
from datetime import datetime, timezone

import sqlalchemy as sa

# ---------------------------------------------------------------------------
# Minimal ORM stub — mirrors what SQLModule defines so we don't import the
# full pipeline stack.
# ---------------------------------------------------------------------------
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class COMAPData(Base):
    __tablename__ = "comap_data"
    obsid: Mapped[int] = mapped_column(primary_key=True)
    level2_path: Mapped[str | None] = mapped_column(nullable=True)


# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="List obsids with stale Level-2 files.")
    p.add_argument(
        "cutoff_date",
        help="Files last modified BEFORE this date are reported (YYYY-MM-DD).",
    )
    p.add_argument(
        "--db",
        default="databases/COMAP_manchester.db",
        help="Path to the SQLite database (default: databases/COMAP_manchester.db).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Optional file path to write the resulting obsid list.",
    )
    p.add_argument(
        "--missing",
        action="store_true",
        help="Also report obsids whose Level-2 file path is set but the file is missing.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Parse cutoff date (treat as UTC midnight)
    try:
        cutoff_dt = datetime.strptime(args.cutoff_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        sys.exit(f"ERROR: Cannot parse date '{args.cutoff_date}'. Use YYYY-MM-DD.")

    cutoff_ts = cutoff_dt.timestamp()
    print(f"Cutoff: {cutoff_dt.strftime('%Y-%m-%d')} UTC  ({cutoff_ts:.0f})")

    if not os.path.exists(args.db):
        sys.exit(f"ERROR: Database not found: {args.db}")

    engine = sa.create_engine(f"sqlite:///{args.db}")
    with engine.connect() as conn:
        rows = conn.execute(
            sa.text(
                "SELECT obsid, level2_path FROM comap_data "
                "WHERE level2_path IS NOT NULL AND level2_path != '' "
                "ORDER BY obsid"
            )
        ).fetchall()

    stale, missing = [], []

    for obsid, path in rows:
        if not os.path.exists(path):
            if args.missing:
                missing.append(obsid)
            continue
        mtime = os.path.getmtime(path)
        if mtime < cutoff_ts:
            stale.append((obsid, path, datetime.fromtimestamp(mtime, tz=timezone.utc)))

    # Report
    print(f"\nFound {len(stale)} obsid(s) with Level-2 files older than {args.cutoff_date}:")
    for obsid, path, mtime in stale:
        print(f"  {obsid:6d}  {mtime.strftime('%Y-%m-%d %H:%M:%S')}  {path}")

    if args.missing and missing:
        print(f"\nFound {len(missing)} obsid(s) with missing Level-2 files:")
        for obsid in missing:
            print(f"  {obsid:6d}")

    if args.output:
        with open(args.output, "w") as f:
            for obsid, _, _ in stale:
                f.write(f"{obsid}\n")
        print(f"\nObsid list written to: {args.output}")

    return stale


if __name__ == "__main__":
    main()
