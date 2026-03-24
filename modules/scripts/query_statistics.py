#!/usr/bin/env python
"""
Query pipeline statistics from the ObservationSummary SQL table.

Answers common questions without opening HDF5 files:
  - Which observations have high/low Tsys?
  - Which observations have high atmospheric tau?
  - Which calibrator observations have poor fits?
  - What is the processing status of observations?
  - Which observations failed and why?

Usage:
    python query_statistics.py <database_path> <command> [options]

Commands:
    summary     Overview of processing status counts
    tsys        List observations by Tsys range
    tau         List observations by atmospheric tau range
    calibrator  List calibrator observations with flux/chi2/pointing info
    failed      List observations that failed processing with error messages
    status      List observations filtered by processing status
"""

import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from modules.SQLModule.SQLModule import SQLModule, COMAPData, ObservationSummary


def cmd_summary(db, args):
    """Print overview of processing status counts."""
    db._connect()
    total = db.session.query(COMAPData).count()
    has_level2 = db.session.query(COMAPData).filter(COMAPData.level2_path.isnot(None)).count()

    from sqlalchemy import func
    status_counts = (db.session.query(ObservationSummary.processing_status, func.count())
                     .group_by(ObservationSummary.processing_status).all())
    has_summary = db.session.query(ObservationSummary).count()
    has_tsys = db.session.query(ObservationSummary).filter(ObservationSummary.median_tsys.isnot(None)).count()
    has_tau = db.session.query(ObservationSummary).filter(ObservationSummary.mean_tau.isnot(None)).count()
    has_cal = db.session.query(ObservationSummary).filter(ObservationSummary.calibrator_flux.isnot(None)).count()
    db._disconnect()

    print(f"Total observations:       {total}")
    print(f"With Level-2 path:        {has_level2}")
    print(f"With summary statistics:  {has_summary}")
    print(f"  - with Tsys:            {has_tsys}")
    print(f"  - with atmosphere tau:  {has_tau}")
    print(f"  - with calibrator fit:  {has_cal}")
    print()
    print("Processing status breakdown:")
    for status, count in status_counts:
        print(f"  {status or 'NULL':15s}  {count}")


def cmd_tsys(db, args):
    """List observations by Tsys range."""
    db._connect()
    query = (db.session.query(COMAPData.obsid, COMAPData.source, COMAPData.utc_start,
                              ObservationSummary.median_tsys)
             .join(ObservationSummary, COMAPData.obsid == ObservationSummary.obsid)
             .filter(ObservationSummary.median_tsys.isnot(None)))
    if args.min_tsys is not None:
        query = query.filter(ObservationSummary.median_tsys >= args.min_tsys)
    if args.max_tsys is not None:
        query = query.filter(ObservationSummary.median_tsys <= args.max_tsys)
    if args.source:
        query = query.filter(COMAPData.source.contains(args.source))
    query = query.order_by(ObservationSummary.median_tsys.desc())

    rows = query.limit(args.limit).all()
    db._disconnect()

    print(f"{'obsid':>8s}  {'source':>15s}  {'utc_start':>20s}  {'median_tsys':>12s}")
    print("-" * 60)
    for obsid, source, utc, tsys in rows:
        print(f"{obsid:8d}  {source or '':>15s}  {utc or '':>20s}  {tsys:12.2f}")
    print(f"\n{len(rows)} observations shown")


def cmd_tau(db, args):
    """List observations by atmospheric tau range."""
    db._connect()
    query = (db.session.query(COMAPData.obsid, COMAPData.source, COMAPData.utc_start,
                              ObservationSummary.mean_tau)
             .join(ObservationSummary, COMAPData.obsid == ObservationSummary.obsid)
             .filter(ObservationSummary.mean_tau.isnot(None)))
    if args.min_tau is not None:
        query = query.filter(ObservationSummary.mean_tau >= args.min_tau)
    if args.max_tau is not None:
        query = query.filter(ObservationSummary.mean_tau <= args.max_tau)
    if args.source:
        query = query.filter(COMAPData.source.contains(args.source))
    query = query.order_by(ObservationSummary.mean_tau.desc())

    rows = query.limit(args.limit).all()
    db._disconnect()

    print(f"{'obsid':>8s}  {'source':>15s}  {'utc_start':>20s}  {'mean_tau':>10s}")
    print("-" * 58)
    for obsid, source, utc, tau in rows:
        print(f"{obsid:8d}  {source or '':>15s}  {utc or '':>20s}  {tau:10.4f}")
    print(f"\n{len(rows)} observations shown")


def cmd_calibrator(db, args):
    """List calibrator observations with fit statistics."""
    db._connect()
    query = (db.session.query(COMAPData.obsid, COMAPData.source, COMAPData.utc_start,
                              ObservationSummary.calibrator_flux,
                              ObservationSummary.calibrator_chi2,
                              ObservationSummary.pointing_offset_az,
                              ObservationSummary.pointing_offset_el)
             .join(ObservationSummary, COMAPData.obsid == ObservationSummary.obsid)
             .filter(ObservationSummary.calibrator_flux.isnot(None)))
    if args.source:
        query = query.filter(COMAPData.source.contains(args.source))
    if args.max_chi2 is not None:
        query = query.filter(ObservationSummary.calibrator_chi2 <= args.max_chi2)
    query = query.order_by(COMAPData.obsid)

    rows = query.limit(args.limit).all()
    db._disconnect()

    print(f"{'obsid':>8s}  {'source':>10s}  {'utc_start':>20s}  {'flux(Jy)':>10s}  {'chi2':>8s}  {'dAz(deg)':>9s}  {'dEl(deg)':>9s}")
    print("-" * 82)
    for obsid, source, utc, flux, chi2, daz, del_ in rows:
        print(f"{obsid:8d}  {source or '':>10s}  {utc or '':>20s}  {flux:10.2f}  {chi2:8.2f}  {daz:9.4f}  {del_:9.4f}")
    print(f"\n{len(rows)} observations shown")


def cmd_failed(db, args):
    """List observations that failed processing."""
    db._connect()
    query = (db.session.query(COMAPData.obsid, COMAPData.source,
                              ObservationSummary.processing_error)
             .join(ObservationSummary, COMAPData.obsid == ObservationSummary.obsid)
             .filter(ObservationSummary.processing_status == 'failed'))
    if args.source:
        query = query.filter(COMAPData.source.contains(args.source))
    query = query.order_by(COMAPData.obsid)

    rows = query.limit(args.limit).all()
    db._disconnect()

    print(f"{'obsid':>8s}  {'source':>15s}  error")
    print("-" * 70)
    for obsid, source, error in rows:
        print(f"{obsid:8d}  {source or '':>15s}  {error or ''}")
    print(f"\n{len(rows)} failed observations")


def cmd_status(db, args):
    """List observations filtered by processing status."""
    db._connect()
    query = (db.session.query(COMAPData.obsid, COMAPData.source, COMAPData.utc_start,
                              ObservationSummary.processing_status,
                              ObservationSummary.n_scans,
                              ObservationSummary.median_tsys)
             .join(ObservationSummary, COMAPData.obsid == ObservationSummary.obsid))
    if args.processing_status:
        query = query.filter(ObservationSummary.processing_status == args.processing_status)
    if args.source:
        query = query.filter(COMAPData.source.contains(args.source))
    query = query.order_by(COMAPData.obsid)

    rows = query.limit(args.limit).all()
    db._disconnect()

    print(f"{'obsid':>8s}  {'source':>15s}  {'utc_start':>20s}  {'status':>10s}  {'n_scans':>8s}  {'tsys':>8s}")
    print("-" * 75)
    for obsid, source, utc, status, n_scans, tsys in rows:
        n_str = f"{n_scans:8d}" if n_scans is not None else "       -"
        t_str = f"{tsys:8.1f}" if tsys is not None else "       -"
        print(f"{obsid:8d}  {source or '':>15s}  {utc or '':>20s}  {status or '':>10s}  {n_str}  {t_str}")
    print(f"\n{len(rows)} observations shown")


def main():
    parser = argparse.ArgumentParser(description="Query COMAP pipeline statistics from SQL")
    parser.add_argument("database", help="Path to SQLite database")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # summary
    subparsers.add_parser("summary", help="Overview of processing status counts")

    # tsys
    p_tsys = subparsers.add_parser("tsys", help="List observations by Tsys range")
    p_tsys.add_argument("--min-tsys", type=float, default=None)
    p_tsys.add_argument("--max-tsys", type=float, default=None)
    p_tsys.add_argument("--source", type=str, default=None)
    p_tsys.add_argument("--limit", type=int, default=100)

    # tau
    p_tau = subparsers.add_parser("tau", help="List observations by atmospheric tau")
    p_tau.add_argument("--min-tau", type=float, default=None)
    p_tau.add_argument("--max-tau", type=float, default=None)
    p_tau.add_argument("--source", type=str, default=None)
    p_tau.add_argument("--limit", type=int, default=100)

    # calibrator
    p_cal = subparsers.add_parser("calibrator", help="List calibrator fit statistics")
    p_cal.add_argument("--source", type=str, default=None)
    p_cal.add_argument("--max-chi2", type=float, default=None)
    p_cal.add_argument("--limit", type=int, default=100)

    # failed
    p_fail = subparsers.add_parser("failed", help="List failed observations with errors")
    p_fail.add_argument("--source", type=str, default=None)
    p_fail.add_argument("--limit", type=int, default=100)

    # status
    p_status = subparsers.add_parser("status", help="List observations by processing status")
    p_status.add_argument("--processing-status", type=str, default=None,
                          choices=["pending", "complete", "failed"])
    p_status.add_argument("--source", type=str, default=None)
    p_status.add_argument("--limit", type=int, default=100)

    args = parser.parse_args()

    db = SQLModule()
    db.connect(args.database)

    commands = {
        "summary": cmd_summary,
        "tsys": cmd_tsys,
        "tau": cmd_tau,
        "calibrator": cmd_calibrator,
        "failed": cmd_failed,
        "status": cmd_status,
    }
    commands[args.command](db, args)


if __name__ == "__main__":
    main()
