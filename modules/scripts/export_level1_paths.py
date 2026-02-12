#!/usr/bin/env python3
import argparse
from pathlib import Path
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

# Import your models exactly as defined
# (adjust the import path if these classes live elsewhere)
from modules.SQLModule.SQLModule import COMAPData, Base

def main(db_path: str, out_path: str) -> None:
    # Connect (read-only is fine; SQLite ignores ro flag unless using URI; keep simple)
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Build query
        stmt = (select(COMAPData.obsid, COMAPData.level1_path)
                .where(COMAPData.source == "TauA")
                .order_by(COMAPData.obsid))

        rows = session.execute(stmt).all()

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for obsid, lvl1 in rows:
                f.write(f"{lvl1 if lvl1 else ''}\n")

    finally:
        session.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Export Level-2 file paths from COMAP SQLite DB.")
    p.add_argument("--db", default="databases/COMAP_manchester.db",
                   help="Path to SQLite database file")
    p.add_argument("--out", default="level1_paths.txt",
                   help="Output text file")
    args = p.parse_args()
    main(args.db, args.out)
