#!/usr/bin/env python3
import argparse
from pathlib import Path
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

# Import your models exactly as defined
# (adjust the import path if these classes live elsewhere)
from modules.SQLModule.SQLModule import COMAPData, Base

def main(db_path: str, out_path: str, include_missing: bool) -> None:
    # Connect (read-only is fine; SQLite ignores ro flag unless using URI; keep simple)
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Build query
        if include_missing:
            stmt = select(COMAPData.obsid, 
                         COMAPData.level1_path,
                          COMAPData.level2_path).order_by(COMAPData.obsid)
        else:
            stmt = (select(COMAPData.obsid, 
                           COMAPData.level1_path,
                           COMAPData.level2_path)
                    .where(COMAPData.level2_path.isnot(None))
                    .where(COMAPData.level2_path != "")
                    .order_by(COMAPData.obsid))

        rows = session.execute(stmt).all()

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for obsid, lvl1, lvl2 in rows:
                # Write "obsid<TAB>level2_path" (level2_path may be 'None' if include_missing=True)
                f.write(f"{lvl1} {lvl2 if lvl2 else ''}\n")

        # Quick summary to stdout
        total = session.query(COMAPData).count()
        with_paths = session.query(COMAPData).filter(
            COMAPData.level2_path.isnot(None),
            COMAPData.level2_path != "",
        ).count()

        print(f"Wrote {len(rows)} rows to {out}")
        print(f"DB summary: total={total}, with_level2_path={with_paths}, missing_level2_path={total - with_paths}")

    finally:
        session.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Export Level-2 file paths from COMAP SQLite DB.")
    p.add_argument("--db", default="databases/COMAP_manchester.db",
                   help="Path to SQLite database file")
    p.add_argument("--out", default="level2_paths.txt",
                   help="Output text file")
    p.add_argument("--include-missing", action="store_true",
                   help="Include rows where level2_path is NULL/empty")
    args = p.parse_args()
    main(args.db, args.out, args.include_missing)
