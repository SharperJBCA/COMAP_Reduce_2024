"""Quick CLI summaries for COMAP SQL database contents."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tabulate import tabulate

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.SQLModule.SQLModule import COMAPData, db


def cmd_summary(show_examples: int) -> None:
    db._connect()

    total = db.session.query(COMAPData.obsid).count()
    with_level1 = db.session.query(COMAPData.obsid).filter(COMAPData.level1_path.isnot(None), COMAPData.level1_path != "").count()
    with_level2 = db.session.query(COMAPData.obsid).filter(COMAPData.level2_path.isnot(None), COMAPData.level2_path != "").count()

    by_source_group = (
        db.session.query(COMAPData.source_group, COMAPData.obsid)
        .order_by(COMAPData.source_group, COMAPData.obsid)
        .all()
    )
    group_counts: dict[str, int] = {}
    for source_group, _ in by_source_group:
        key = source_group or "<NULL>"
        group_counts[key] = group_counts.get(key, 0) + 1

    print(tabulate([
        ["Total obsids", total],
        ["Rows with level1_path", with_level1],
        ["Rows with level2_path", with_level2],
        ["Rows missing level2_path", total - with_level2],
    ], headers=["Metric", "Count"], tablefmt="github"))

    print("\nSource group distribution:")
    print(tabulate(sorted(group_counts.items(), key=lambda x: x[1], reverse=True), headers=["source_group", "rows"], tablefmt="github"))

    if show_examples > 0:
        examples = (
            db.session.query(COMAPData.obsid, COMAPData.source, COMAPData.level1_path, COMAPData.level2_path)
            .order_by(COMAPData.obsid.desc())
            .limit(show_examples)
            .all()
        )
        print(f"\nRecent {show_examples} rows:")
        print(tabulate(examples, headers=["obsid", "source", "level1_path", "level2_path"], tablefmt="github"))

    db._disconnect()


def cmd_missing_level2(limit: int) -> None:
    db._connect()
    rows = (
        db.session.query(COMAPData.obsid, COMAPData.source, COMAPData.level1_path)
        .filter((COMAPData.level2_path.is_(None)) | (COMAPData.level2_path == ""))
        .order_by(COMAPData.obsid)
        .limit(limit if limit > 0 else None)
        .all()
    )
    db._disconnect()

    print(tabulate(rows, headers=["obsid", "source", "level1_path"], tablefmt="github"))


def cmd_path_search(path_contains: str, limit: int) -> None:
    db._connect()
    rows = (
        db.session.query(COMAPData.obsid, COMAPData.level1_path, COMAPData.level2_path)
        .filter(
            (COMAPData.level1_path.contains(path_contains))
            | (COMAPData.level2_path.contains(path_contains))
        )
        .order_by(COMAPData.obsid)
        .limit(limit if limit > 0 else None)
        .all()
    )
    db._disconnect()

    print(tabulate(rows, headers=["obsid", "level1_path", "level2_path"], tablefmt="github"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect COMAP SQL database contents")
    parser.add_argument("--database", required=True, help="Path to sqlite database")

    subparsers = parser.add_subparsers(dest="command", required=True)

    summary = subparsers.add_parser("summary", help="Print high-level database summary")
    summary.add_argument("--show-examples", type=int, default=5, help="Number of recent rows to print")

    missing = subparsers.add_parser("missing-level2", help="List rows that do not have level2_path")
    missing.add_argument("--limit", type=int, default=200, help="Max rows to print (<=0 means no limit)")

    path_search = subparsers.add_parser("path-search", help="Find rows with level paths containing a substring")
    path_search.add_argument("substring", help="Substring to search inside level1_path and level2_path")
    path_search.add_argument("--limit", type=int, default=200, help="Max rows to print (<=0 means no limit)")

    args = parser.parse_args()

    db.connect(args.database)

    if args.command == "summary":
        cmd_summary(args.show_examples)
    elif args.command == "missing-level2":
        cmd_missing_level2(args.limit)
    elif args.command == "path-search":
        cmd_path_search(args.substring, args.limit)


if __name__ == "__main__":
    main()
