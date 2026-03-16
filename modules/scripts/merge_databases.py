"""Merge two COMAP SQLite databases.

This utility combines rows from a secondary database into a primary database,
optionally writing to a separate output file. It is designed for cases where the
pipeline was run in multiple locations and each location produced partial updates.

Merge strategy:
- `comap_data` and `old_path_data` are matched by `obsid`.
- `quality_flags` are matched by (`obsid`, `pixel`, `frequency_band`).
- For overlapping rows, non-null values from the preferred database are kept.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path
from typing import Iterable


TABLE_CONFIG = {
    "comap_data": {
        "key_columns": ["obsid"],
        "protected_columns": ["obsid"],
    },
    "quality_flags": {
        "key_columns": ["obsid", "pixel", "frequency_band"],
        "protected_columns": ["id", "obsid", "pixel", "frequency_band"],
    },
    "old_path_data": {
        "key_columns": ["obsid"],
        "protected_columns": ["obsid"],
    },
}


def table_exists(conn: sqlite3.Connection, schema: str, table: str) -> bool:
    row = conn.execute(
        f"SELECT 1 FROM {schema}.sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def table_columns(conn: sqlite3.Connection, schema: str, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA {schema}.table_info({table})").fetchall()
    return [row[1] for row in rows]


def build_upsert_sql(
    table: str,
    columns: Iterable[str],
    key_columns: list[str],
    protected_columns: set[str],
    prefer_secondary: bool,
) -> str:
    insert_cols = ", ".join(columns)
    select_cols = ", ".join([f"s.{c}" for c in columns])
    conflict_cols = ", ".join(key_columns)

    assignments = []
    for col in columns:
        if col in protected_columns:
            continue

        if prefer_secondary:
            # Keep incoming value unless it is NULL.
            expr = f"{col}=COALESCE(excluded.{col}, {table}.{col})"
        else:
            # Keep existing value unless it is NULL.
            expr = f"{col}=COALESCE({table}.{col}, excluded.{col})"

        assignments.append(expr)

    if not assignments:
        # No mutable columns, so no-op update is enough for ON CONFLICT clause.
        assignments = [f"{key_columns[0]}={table}.{key_columns[0]}"]

    update_clause = ", ".join(assignments)

    return (
        f"INSERT INTO {table} ({insert_cols}) "
        f"SELECT {select_cols} FROM secondary.{table} s "
        f"ON CONFLICT ({conflict_cols}) DO UPDATE SET {update_clause}"
    )


def merge_table(conn: sqlite3.Connection, table: str, prefer_secondary: bool) -> tuple[int, int]:
    if not table_exists(conn, "main", table):
        print(f"[skip] main database has no table '{table}'")
        return (0, 0)
    if not table_exists(conn, "secondary", table):
        print(f"[skip] secondary database has no table '{table}'")
        return (0, 0)

    cfg = TABLE_CONFIG[table]
    primary_cols = set(table_columns(conn, "main", table))
    secondary_cols = set(table_columns(conn, "secondary", table))
    shared_cols = [c for c in table_columns(conn, "main", table) if c in secondary_cols]

    if not set(cfg["key_columns"]).issubset(primary_cols) or not set(cfg["key_columns"]).issubset(secondary_cols):
        raise RuntimeError(f"Table '{table}' is missing required key columns: {cfg['key_columns']}")

    if not shared_cols:
        print(f"[skip] no shared columns for '{table}'")
        return (0, 0)

    before = conn.total_changes
    sql = build_upsert_sql(
        table=table,
        columns=shared_cols,
        key_columns=cfg["key_columns"],
        protected_columns=set(cfg["protected_columns"]),
        prefer_secondary=prefer_secondary,
    )
    conn.execute(sql)
    changed = conn.total_changes - before

    src_rows = conn.execute(f"SELECT COUNT(*) FROM secondary.{table}").fetchone()[0]
    return src_rows, changed


def prepare_output(primary_db: Path, output_db: Path, force: bool) -> None:
    if primary_db.resolve() == output_db.resolve():
        return

    if output_db.exists() and not force:
        raise FileExistsError(f"Output database already exists: {output_db}. Use --force to overwrite.")

    output_db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(primary_db, output_db)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge two COMAP SQLite databases")
    parser.add_argument("--primary-db", required=True, help="Path to the base SQLite DB")
    parser.add_argument("--secondary-db", required=True, help="Path to the DB to merge from")
    parser.add_argument(
        "--output-db",
        default=None,
        help="Path for merged DB. Defaults to in-place update of --primary-db",
    )
    parser.add_argument(
        "--prefer",
        choices=["primary", "secondary"],
        default="secondary",
        help="When both DBs have non-null values, choose which one wins (default: secondary)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite --output-db if it exists")
    args = parser.parse_args()

    primary_db = Path(args.primary_db).expanduser().resolve()
    secondary_db = Path(args.secondary_db).expanduser().resolve()
    output_db = Path(args.output_db).expanduser().resolve() if args.output_db else primary_db

    if not primary_db.exists():
        raise FileNotFoundError(f"Primary database not found: {primary_db}")
    if not secondary_db.exists():
        raise FileNotFoundError(f"Secondary database not found: {secondary_db}")

    prepare_output(primary_db, output_db, force=args.force)

    conn = sqlite3.connect(str(output_db))
    try:
        conn.execute(f"ATTACH DATABASE '{secondary_db}' AS secondary")
        prefer_secondary = args.prefer == "secondary"

        print(f"Merging into: {output_db}")
        print(f"Preference on overlap: {args.prefer}")

        for table in ("comap_data", "quality_flags", "old_path_data"):
            src_rows, changed_rows = merge_table(conn, table, prefer_secondary=prefer_secondary)
            print(f"[{table}] source rows: {src_rows}, rows inserted/updated: {changed_rows}")

        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
