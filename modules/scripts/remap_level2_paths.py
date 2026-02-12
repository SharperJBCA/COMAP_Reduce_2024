"""Bulk remap Level-2 paths in the SQL database after disk migration."""

import argparse
from pprint import pprint
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.SQLModule.SQLModule import db


def main() -> None:
    parser = argparse.ArgumentParser(description="Remap level2_path prefixes in COMAP SQL database")
    parser.add_argument("--database", required=True, help="Path to sqlite database")
    parser.add_argument("--old-prefix", required=True, help="Old path prefix to replace")
    parser.add_argument("--new-prefix", required=True, help="New path prefix")
    parser.add_argument("--write", action="store_true", help="Apply changes (default is dry-run)")
    parser.add_argument(
        "--require-new-path-exists",
        action="store_true",
        help="Only update entries where the remapped file exists on disk",
    )
    args = parser.parse_args()

    db.connect(args.database)
    summary = db.remap_level2_paths(
        old_prefix=args.old_prefix,
        new_prefix=args.new_prefix,
        dry_run=not args.write,
        require_new_path_exists=args.require_new_path_exists,
    )
    pprint(summary)


if __name__ == "__main__":
    main()
