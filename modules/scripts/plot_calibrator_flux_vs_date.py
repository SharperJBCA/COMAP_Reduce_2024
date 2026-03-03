#!/usr/bin/env python3
"""Plot COMAP calibrator peak measurements vs date.

This script queries COMAP observations by ObsID range, reads per-observation
calibrator-fit products from Level-2 HDF5 files, and writes:
  1) a CSV table with one row per observation
  2) a date-vs-flux-density plot for selected calibrators

By default, flux density is taken from
  level2/calibrator_fit/flux
and collapsed to a scalar per obsid using the median over feed/band/channel.
The peak amplitude (K) median is also saved to the CSV for quick review.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import and_, create_engine
from sqlalchemy.orm import sessionmaker

from modules.SQLModule.SQLModule import COMAPData

DEFAULT_SOURCES = ["TauA", "CasA", "CygA", "Jupiter"]


@dataclass
class ObsRow:
    obsid: int
    source: str
    date_utc: datetime
    flux_jy: float
    peak_amplitude_k: float
    n_flux_samples: int
    n_amp_samples: int
    level2_path: str


def parse_utc_start(utc_start: str) -> datetime:
    """Parse COMAP utc_start string."""
    return datetime.strptime(utc_start, "%Y-%m-%d-%H:%M:%S")


def collapse_stat(values: np.ndarray, statistic: str) -> float:
    """Collapse an n-D array to one scalar while ignoring NaNs."""
    vals = np.asarray(values, dtype=float)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return np.nan

    if statistic == "median":
        return float(np.nanmedian(vals))
    if statistic == "mean":
        return float(np.nanmean(vals))
    raise ValueError(f"Unsupported statistic: {statistic}")


def read_level2_calibrator_metrics(
    level2_path: str,
    fit_group: str,
    statistic: str,
) -> tuple[float, float, int, int]:
    """Return (flux_jy, amp_k, n_flux_samples, n_amp_samples) for one Level-2 file."""
    grp_name = f"level2/{fit_group}"
    with h5py.File(level2_path, "r") as f:
        if grp_name not in f:
            raise KeyError(f"Missing {grp_name} in {level2_path}")

        grp = f[grp_name]
        if "flux" not in grp:
            raise KeyError(f"Missing {grp_name}/flux in {level2_path}")
        if "amplitudes" not in grp:
            raise KeyError(f"Missing {grp_name}/amplitudes in {level2_path}")

        flux = np.asarray(grp["flux"][:], dtype=float)
        amps = np.asarray(grp["amplitudes"][:], dtype=float)

    flux_jy = collapse_stat(flux, statistic)
    amp_k = collapse_stat(amps, statistic)
    n_flux = int(np.count_nonzero(np.isfinite(flux)))
    n_amp = int(np.count_nonzero(np.isfinite(amps)))
    return flux_jy, amp_k, n_flux, n_amp


def fetch_rows(
    database: str,
    min_obsid: int,
    max_obsid: int,
    sources: Iterable[str],
    fit_group: str,
    statistic: str,
    verbose: bool,
) -> list[ObsRow]:
    engine = create_engine(f"sqlite:///{database}")
    Session = sessionmaker(bind=engine)
    session = Session()

    selected = [s.strip() for s in sources if s.strip()]
    q = (
        session.query(COMAPData)
        .filter(
            and_(
                COMAPData.obsid >= min_obsid,
                COMAPData.obsid <= max_obsid,
                COMAPData.level2_path.isnot(None),
                COMAPData.source.in_(selected),
            )
        )
        .order_by(COMAPData.obsid)
    )

    output: list[ObsRow] = []
    for obs in q.all():
        if not obs.utc_start or not obs.level2_path:
            continue

        level2_path = str(obs.level2_path)
        if not Path(level2_path).exists():
            if verbose:
                print(f"[skip] missing level2 file for obsid={obs.obsid}: {level2_path}")
            continue

        try:
            flux_jy, amp_k, n_flux, n_amp = read_level2_calibrator_metrics(
                level2_path=level2_path,
                fit_group=fit_group,
                statistic=statistic,
            )
        except (OSError, KeyError, ValueError) as exc:
            if verbose:
                print(f"[skip] obsid={obs.obsid}: {exc}")
            continue

        output.append(
            ObsRow(
                obsid=int(obs.obsid),
                source=str(obs.source),
                date_utc=parse_utc_start(str(obs.utc_start)),
                flux_jy=float(flux_jy),
                peak_amplitude_k=float(amp_k),
                n_flux_samples=n_flux,
                n_amp_samples=n_amp,
                level2_path=level2_path,
            )
        )

    session.close()
    return output


def write_csv(rows: list[ObsRow], out_csv: Path) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "obsid": r.obsid,
                "source": r.source,
                "date_utc": r.date_utc,
                "flux_density_jy": r.flux_jy,
                "peak_amplitude_k": r.peak_amplitude_k,
                "n_flux_samples": r.n_flux_samples,
                "n_amp_samples": r.n_amp_samples,
                "level2_path": r.level2_path,
            }
            for r in rows
        ]
    )
    if not df.empty:
        df = df.sort_values(["source", "date_utc", "obsid"]).reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def make_plot(df: pd.DataFrame, out_plot: Path, title: str | None = None) -> None:
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    if df.empty:
        ax.set_title(title or "Calibrator flux density vs date")
        ax.set_xlabel("Date (UTC)")
        ax.set_ylabel("Flux density (Jy)")
        ax.text(0.5, 0.5, "No matching rows", ha="center", va="center", transform=ax.transAxes)
    else:
        for source, g in df.groupby("source"):
            gg = g.sort_values("date_utc")
            ax.plot(
                gg["date_utc"],
                gg["flux_density_jy"],
                marker="o",
                linestyle="-",
                linewidth=1.0,
                markersize=3,
                label=source,
            )

        ax.set_title(title or "Calibrator flux density vs date")
        ax.set_xlabel("Date (UTC)")
        ax.set_ylabel("Flux density (Jy)")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_plot, dpi=180)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Plot calibrator date-vs-flux-density from Level-2 calibrator fit products "
            "and save a per-observation CSV."
        )
    )
    p.add_argument("--database", default="databases/COMAP_manchester.db", help="Path to SQLite database")
    p.add_argument("--min-obsid", type=int, required=True, help="Minimum obsid (inclusive)")
    p.add_argument("--max-obsid", type=int, required=True, help="Maximum obsid (inclusive)")
    p.add_argument(
        "--sources",
        nargs="+",
        default=DEFAULT_SOURCES,
        help="Calibrator sources to include (default: TauA CasA CygA Jupiter)",
    )
    p.add_argument(
        "--fit-group",
        default="calibrator_fit",
        help="Level-2 group under level2/ that stores flux and amplitudes",
    )
    p.add_argument(
        "--statistic",
        choices=["median", "mean"],
        default="median",
        help="How to collapse feed/band/channel values into one value per obsid",
    )
    p.add_argument("--output-csv", default="calibrator_flux_vs_date.csv", help="Output CSV path")
    p.add_argument("--output-plot", default="calibrator_flux_vs_date.png", help="Output plot path")
    p.add_argument("--title", default=None, help="Optional custom plot title")
    p.add_argument("--verbose", action="store_true", help="Print skipped observations")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.min_obsid > args.max_obsid:
        raise ValueError("--min-obsid must be <= --max-obsid")

    rows = fetch_rows(
        database=args.database,
        min_obsid=args.min_obsid,
        max_obsid=args.max_obsid,
        sources=args.sources,
        fit_group=args.fit_group,
        statistic=args.statistic,
        verbose=args.verbose,
    )

    csv_path = Path(args.output_csv)
    plot_path = Path(args.output_plot)

    df = write_csv(rows, csv_path)
    make_plot(df, plot_path, title=args.title)

    print(f"Found {len(df)} matching observations")
    print(f"Wrote CSV:  {csv_path}")
    print(f"Wrote plot: {plot_path}")


if __name__ == "__main__":
    main()
