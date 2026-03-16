#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Tuple
from sqlalchemy import create_engine, func, case
from sqlalchemy.orm import sessionmaker

# Import your ORM models
from modules.SQLModule.SQLModule import COMAPData

def run(db_path: str, group_by: str, min_obsid: int, csv_out: str | None) -> None:
    """
    Count, per group (source_group or source):
      - total L1 entries (level1_path present)
      - missing L2 entries (level1_path present AND level2_path NULL/empty)
    """
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        grp_col = COMAPData.source_group if group_by == "source_group" else COMAPData.source

        # Only rows that have L1 (we're answering "how many L1s do not have L2")
        base = session.query(
            grp_col.label("group"),
            func.count().label("total_l1"),
            func.sum(
                case(
                    (
                        (COMAPData.level2_path.is_(None)) | (COMAPData.level2_path == ""),
                        1
                    ),
                    else_=0,
                )
            ).label("missing_l2")
        ).filter(
            COMAPData.level1_path.isnot(None), COMAPData.level1_path != "",
            COMAPData.obsid >= min_obsid
        ).group_by(grp_col).order_by(grp_col)

        rows: List[Tuple[str, int, int]] = [(g or "Unknown", t, m) for g, t, m in base.all()]

        # Totals
        total_l1 = sum(r[1] for r in rows)
        total_missing = sum(r[2] for r in rows)

        # Pretty print table
        name = "Source Group" if group_by == "source_group" else "Source"
        colw = max(len(name), *(len(r[0]) for r in rows)) if rows else len(name)
        header = f"{name:<{colw}}  {'Total L1':>9}  {'Missing L2':>11}  {'Pct Missing':>12}"
        print(header)
        print("-" * len(header))
        for g, t, m in rows:
            pct = (m / t * 100.0) if t else 0.0
            print(f"{g:<{colw}}  {t:>9d}  {m:>11d}  {pct:>11.1f}%")
        print("-" * len(header))
        pct_all = (total_missing / total_l1 * 100.0) if total_l1 else 0.0
        print(f"{'TOTAL':<{colw}}  {total_l1:>9d}  {total_missing:>11d}  {pct_all:>11.1f}%")

        # Optional CSV
        if csv_out:
            out = Path(csv_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as f:
                f.write(f"{name},total_l1,missing_l2,pct_missing\n")
                for g, t, m in rows:
                    pct = (m / t * 100.0) if t else 0.0
                    f.write(f"{g},{t},{m},{pct:.3f}\n")
                f.write(f"TOTAL,{total_l1},{total_missing},{pct_all:.3f}\n")
            print(f"\nWrote CSV -> {out}")

    finally:
        session.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Count Level-1 rows lacking Level-2 paths, grouped by source or source_group."
    )
    p.add_argument("--db", default="databases/COMAP_manchester.db", help="SQLite DB path")
    p.add_argument("--group-by", choices=["source_group", "source"], default="source_group",
                   help="Grouping key")
    p.add_argument("--min-obsid", type=int, default=0, help="Only include obsid >= this")
    p.add_argument("--csv", dest="csv_out", default=None, help="Optional CSV output file")
    args = p.parse_args()
    run(args.db, args.group_by, args.min_obsid, args.csv_out)
