"""Reconcile Level-2 files on disk against SQL entries.

This script scans one or more directories for Level-2 files, extracts the obsid
from filenames, and compares the results to SQL entries in `comap_data`.

Default mode is dry-run. Use `--write` to apply updates.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.SQLModule.SQLModule import COMAPData, db

LEVEL2_PATTERN = re.compile(r"Level2_comap-(\d{7})-")


@dataclass
class ReconcileSummary:
    scanned_files: int
    matched_files: int
    unique_obsids: int
    duplicate_obsids: dict[int, list[str]]
    db_level2_rows: int
    db_rows_missing_on_disk: list[tuple[int, str, str | None]]
    db_rows_remapped: list[tuple[int, str | None, str]]
    files_with_obsid_not_in_db: list[tuple[int, str]]


def parse_obsid_from_name(path: Path) -> int | None:
    match = LEVEL2_PATTERN.search(path.name)
    if not match:
        return None
    return int(match.group(1))


def discover_level2_files(search_roots: Iterable[Path]) -> tuple[dict[int, str], dict[int, list[str]], int, int]:
    selected_paths: dict[int, str] = {}
    all_paths_by_obsid: defaultdict[int, list[str]] = defaultdict(list)
    scanned_files = 0
    matched_files = 0

    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("Level2_comap-*"):
            if not path.is_file():
                continue
            scanned_files += 1
            obsid = parse_obsid_from_name(path)
            if obsid is None:
                continue
            matched_files += 1
            path_str = str(path.resolve())
            all_paths_by_obsid[obsid].append(path_str)

    duplicate_obsids = {obsid: paths for obsid, paths in all_paths_by_obsid.items() if len(paths) > 1}

    for obsid, paths in all_paths_by_obsid.items():
        selected_paths[obsid] = sorted(paths)[0]

    return selected_paths, duplicate_obsids, scanned_files, matched_files


def write_list(path: Path, rows: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row}\n")


def reconcile(database: str, roots: list[str], write: bool, output_dir: str) -> ReconcileSummary:
    search_roots = [Path(r).expanduser().resolve() for r in roots]
    discovered, duplicate_obsids, scanned_files, matched_files = discover_level2_files(search_roots)

    db.connect(database)
    db._connect()

    rows = (
        db.session.query(COMAPData.obsid, COMAPData.level1_path, COMAPData.level2_path)
        .filter(COMAPData.level2_path.isnot(None))
        .filter(COMAPData.level2_path != "")
        .all()
    )

    rows_by_obsid = {obsid: (level1_path, level2_path) for obsid, level1_path, level2_path in rows}

    db_rows_missing_on_disk: list[tuple[int, str, str | None]] = []
    db_rows_remapped: list[tuple[int, str | None, str]] = []

    for obsid, (level1_path, level2_path) in rows_by_obsid.items():
        discovered_path = discovered.get(obsid)
        if discovered_path is None:
            db_rows_missing_on_disk.append((obsid, level2_path, level1_path))
            if write:
                db.session.query(COMAPData).filter_by(obsid=obsid).update({"level2_path": None})
            continue

        if level2_path != discovered_path:
            db_rows_remapped.append((obsid, level2_path, discovered_path))
            if write:
                db.session.query(COMAPData).filter_by(obsid=obsid).update({"level2_path": discovered_path})

    db_obsid_rows = db.session.query(COMAPData.obsid).all()
    db_obsid_set = {item[0] for item in db_obsid_rows}

    files_with_obsid_not_in_db = [
        (obsid, path)
        for obsid, path in sorted(discovered.items())
        if obsid not in db_obsid_set
    ]

    if write and (db_rows_missing_on_disk or db_rows_remapped):
        db.session.commit()

    db._disconnect()

    output = Path(output_dir)
    missing_table = [f"{obsid}\t{level2_path}\t{level1_path or ''}" for obsid, level2_path, level1_path in db_rows_missing_on_disk]
    write_list(output / "missing_level2_rows.tsv", missing_table)

    obsids_for_reprocessing = [str(obsid) for obsid, _, _ in db_rows_missing_on_disk]
    write_list(output / "missing_level2_obsids.txt", obsids_for_reprocessing)

    level1_for_reprocessing = [level1_path for _, _, level1_path in db_rows_missing_on_disk if level1_path]
    write_list(output / "missing_level2_level1_paths.txt", level1_for_reprocessing)

    remapped_table = [f"{obsid}\t{old or ''}\t{new}" for obsid, old, new in db_rows_remapped]
    write_list(output / "remapped_level2_rows.tsv", remapped_table)

    orphan_table = [f"{obsid}\t{path}" for obsid, path in files_with_obsid_not_in_db]
    write_list(output / "level2_files_not_in_db.tsv", orphan_table)

    duplicate_table = []
    for obsid, paths in sorted(duplicate_obsids.items()):
        for path in paths:
            duplicate_table.append(f"{obsid}\t{path}")
    write_list(output / "duplicate_level2_files.tsv", duplicate_table)

    return ReconcileSummary(
        scanned_files=scanned_files,
        matched_files=matched_files,
        unique_obsids=len(discovered),
        duplicate_obsids=duplicate_obsids,
        db_level2_rows=len(rows_by_obsid),
        db_rows_missing_on_disk=db_rows_missing_on_disk,
        db_rows_remapped=db_rows_remapped,
        files_with_obsid_not_in_db=files_with_obsid_not_in_db,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile Level-2 file inventory with SQL table entries")
    parser.add_argument("--database", required=True, help="Path to sqlite database")
    parser.add_argument(
        "--search-root",
        nargs="+",
        required=True,
        help="Directories to recursively scan for Level-2 files",
    )
    parser.add_argument("--output-dir", default="reconcile_reports", help="Folder where report files are written")
    parser.add_argument("--write", action="store_true", help="Apply SQL updates/deletions; default is dry-run")
    args = parser.parse_args()

    summary = reconcile(
        database=args.database,
        roots=args.search_root,
        write=args.write,
        output_dir=args.output_dir,
    )

    mode = "WRITE" if args.write else "DRY-RUN"
    print(f"[{mode}] scanned files: {summary.scanned_files}")
    print(f"[{mode}] Level2 filename matches: {summary.matched_files}")
    print(f"[{mode}] unique obsids found on disk: {summary.unique_obsids}")
    print(f"[{mode}] db rows with level2_path: {summary.db_level2_rows}")
    print(f"[{mode}] db rows with missing Level2 file on disk: {len(summary.db_rows_missing_on_disk)}")
    print(f"[{mode}] db rows remapped to discovered Level2 path: {len(summary.db_rows_remapped)}")
    print(f"[{mode}] discovered Level2 files not in DB: {len(summary.files_with_obsid_not_in_db)}")
    print(f"[{mode}] obsids with duplicate Level2 files on disk: {len(summary.duplicate_obsids)}")


if __name__ == "__main__":
    main()
