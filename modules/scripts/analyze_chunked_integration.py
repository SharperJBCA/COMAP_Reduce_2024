#!/usr/bin/env python3
"""Analyze cumulative map depth from chunked map-making runs.

Workflow:
1. Query Level-2 files for a source/obsid window from SQL.
2. Split files into ordered chunks.
3. Run map-making for each chunk via modules/MapMaking/run_map_making.py.
4. Build cumulative weighted maps over chunks.
5. Measure RMS/hits metrics in user-defined regions and plot RMS-vs-hits scaling.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import toml
from astropy.io import fits
from astropy.wcs import WCS
from sqlalchemy import and_, create_engine
from sqlalchemy.orm import sessionmaker

from modules.SQLModule.SQLModule import COMAPData, SQLModule


@dataclass
class ChunkResult:
    chunk_index: int
    n_chunks: int
    n_files: int
    rms_center_mean: float
    rms_center_median: float
    mean_hits_annulus: float


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run chunked map-making and track cumulative RMS vs hits scaling."
        )
    )

    # File selection
    p.add_argument("--database", required=True, help="Path to SQLite database")
    p.add_argument("--source", required=True, help="Source substring to match in SQL")
    p.add_argument("--min-obsid", type=int, required=True, help="Minimum obsid (inclusive)")
    p.add_argument("--max-obsid", type=int, required=True, help="Maximum obsid (inclusive)")
    p.add_argument("--chunk-size", type=int, required=True, help="Files per chunk")

    # Map-making / destriper parameters
    p.add_argument("--band", type=int, required=True, help="Band index (0-7)")
    p.add_argument("--feeds", nargs="+", type=int, default=list(range(1, 20)), help="Feed list")
    p.add_argument("--wcs-def", required=True, help="WCS definition string")
    p.add_argument("--offset-length", type=int, default=100, help="Destriper offset length")
    p.add_argument("--sigma-red-cutoff", type=float, default=0.4, help="Sigma red cutoff")
    p.add_argument("--tod-data-name", default="level2/binned_filtered_data", help="TOD dataset name")
    p.add_argument("--lambda-ridge", type=float, default=1e-4, help="Destriper ridge regularization")
    p.add_argument("--threshold", type=float, default=1e-6, help="CGM stopping threshold")
    p.add_argument("--alpha-grad", type=float, default=0.0, help="Gradient regularization")
    p.add_argument("--alpha-curv", type=float, default=0.0, help="Curvature regularization")
    p.add_argument("--planck-30-path", default=None, help="Optional Planck map path")
    p.add_argument("--calib-path", default=None, help="Optional calibration array path")
    p.add_argument("--jackknife-odd-even", default=None, help="Optional odd/even jackknife mode")
    p.add_argument("--map-name", default="chunked_integration", help="Base map name")
    p.add_argument("--n-processes", type=int, default=1, help="MPI process count")
    p.add_argument("--use-mpirun", action="store_true", help="Run map-making under mpirun")

    # Region selection
    p.add_argument("--center-lon", type=float, required=True, help="Center longitude in map world coords")
    p.add_argument("--center-lat", type=float, required=True, help="Center latitude in map world coords")
    p.add_argument("--center-radius-pix", type=float, required=True, help="Center aperture radius [pix]")
    p.add_argument("--annulus-rin-pix", type=float, required=True, help="Annulus inner radius [pix]")
    p.add_argument("--annulus-rout-pix", type=float, required=True, help="Annulus outer radius [pix]")

    # Output paths
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument("--plot-name", default="chunked_rms_vs_hits.png", help="Output plot filename")
    p.add_argument("--csv-name", default="chunked_rms_vs_hits.csv", help="Output CSV filename")
    p.add_argument("--save-cumulative-npz", action="store_true", help="Save cumulative arrays as NPZ each stage")
    p.add_argument("--save-cumulative-fits", action="store_true", help="Save cumulative FITS each stage")

    return p


def query_level2_files(database: str, source: str, min_obsid: int, max_obsid: int) -> list[tuple[int, str]]:
    """Equivalent SQL query flow to MapMaking file selection, with Level-2 checks."""
    engine = create_engine(f"sqlite:///{database}")
    Session = sessionmaker(bind=engine)
    session = Session()

    sql_db = SQLModule()
    sql_db.connect(database)

    rows = (
        session.query(COMAPData)
        .filter(
            and_(
                COMAPData.obsid >= min_obsid,
                COMAPData.obsid <= max_obsid,
                COMAPData.level2_path.isnot(None),
                COMAPData.source.contains(source),
            )
        )
        .order_by(COMAPData.obsid)
        .all()
    )

    selected: list[tuple[int, str]] = []
    for row in rows:
        if row.level2_path is None:
            continue
        level2_path = str(row.level2_path)
        try:
            flags = sql_db.get_quality_flags(int(row.obsid))
            if flags and all([not v.is_good for _, v in flags.items()]):
                continue
            with h5py.File(level2_path, "r") as h5:
                if (
                    "level2/binned_filtered_data" in h5
                    and "level2_noise_stats/binned_filtered_data/auto_rms" in h5
                ):
                    selected.append((int(row.obsid), level2_path))
        except (OSError, KeyError):
            continue

    session.close()
    return selected


def chunk_list(items: list[str], chunk_size: int) -> list[list[str]]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def run_mapmaking_chunk(args: argparse.Namespace, chunk_files: list[str], chunk_idx: int, work_dir: Path) -> Path:
    file_list = work_dir / f"chunk_{chunk_idx:03d}_files.txt"
    config_file = work_dir / f"chunk_{chunk_idx:03d}_config.toml"

    output_filename = f"{args.map_name}_chunk{chunk_idx:03d}_band{args.band:02d}.fits"
    log_file = Path(args.output_dir).resolve() / f"{args.map_name}_chunk{chunk_idx:03d}.log"
    params = {
        "band": args.band,
        "wcs_def": args.wcs_def,
        "source": args.source,
        "source_group": "Unknown",
        "offset_length": args.offset_length,
        "feeds": list(args.feeds),
        "map_name": f"{args.map_name}_chunk{chunk_idx:03d}",
        "tod_data_name": args.tod_data_name,
        # run_map_making.py expects a path-like log filename and unconditionally
        # creates its parent directory in setup_logging(...).
        "log_file_name": str(log_file),
        "sigma_red_cutoff": args.sigma_red_cutoff,
        "database": str(Path(args.database).resolve()),
        "file_list": str(file_list),
        "output_filename": output_filename,
        "output_dir": str(Path(args.output_dir).resolve()),
        "lambda_ridge": args.lambda_ridge,
        "planck_30_path": args.planck_30_path,
        "calib_path": args.calib_path,
        "jackknife_odd_even": args.jackknife_odd_even,
        "threshold": args.threshold,
        "alpha_grad": args.alpha_grad,
        "alpha_curv": args.alpha_curv,
    }

    with file_list.open("w") as f:
        for path in chunk_files:
            f.write(path + "\n")

    with config_file.open("w") as f:
        toml.dump(params, f)

    run_script = Path("modules/MapMaking/run_map_making.py").resolve()
    command = ["python", str(run_script), str(config_file)]
    if args.use_mpirun or args.n_processes > 1:
        command = ["mpirun", "-n", str(args.n_processes)] + command

    print(f"[chunk {chunk_idx}] running: {' '.join(command)}")
    subprocess.run(command, check=True)

    out_fits = Path(args.output_dir) / output_filename
    if not out_fits.exists():
        raise FileNotFoundError(f"Expected map output not found: {out_fits}")
    return out_fits


def read_map_products(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, fits.Header]:
    with fits.open(path) as hdul:
        primary = np.asarray(hdul[0].data, dtype=float)
        rms = np.asarray(hdul["RMS"].data, dtype=float)
        hits = np.asarray(hdul["HITS"].data, dtype=float)
        header = hdul[0].header.copy()
    return primary, rms, hits, header


def build_region_masks(shape: tuple[int, int], header: fits.Header, lon: float, lat: float, r_center: float, r_in: float, r_out: float) -> tuple[np.ndarray, np.ndarray]:
    wcs = WCS(header)
    cx, cy = wcs.world_to_pixel_values(lon, lat)
    yy, xx = np.indices(shape)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    center_mask = rr <= r_center
    annulus_mask = (rr >= r_in) & (rr <= r_out)
    return center_mask, annulus_mask


def compute_region_stats(cum_rms: np.ndarray, cum_hits: np.ndarray, center_mask: np.ndarray, annulus_mask: np.ndarray) -> tuple[float, float, float]:
    center_vals = cum_rms[center_mask & np.isfinite(cum_rms)]
    annulus_hits = cum_hits[annulus_mask & np.isfinite(cum_hits)]

    center_mean = float(np.nanmean(center_vals)) if center_vals.size > 0 else math.nan
    center_median = float(np.nanmedian(center_vals)) if center_vals.size > 0 else math.nan
    annulus_mean_hits = float(np.nanmean(annulus_hits)) if annulus_hits.size > 0 else math.nan
    return center_mean, center_median, annulus_mean_hits


def save_results_csv(rows: list[ChunkResult], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n_chunks",
            "n_files",
            "rms_center_mean",
            "rms_center_median",
            "mean_hits_annulus",
        ])
        for r in rows:
            writer.writerow([
                r.n_chunks,
                r.n_files,
                r.rms_center_mean,
                r.rms_center_median,
                r.mean_hits_annulus,
            ])


def plot_results(rows: list[ChunkResult], out_plot: Path) -> None:
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    hits = np.array([r.mean_hits_annulus for r in rows], dtype=float)
    rms = np.array([r.rms_center_median for r in rows], dtype=float)
    nfiles = np.array([r.n_files for r in rows], dtype=int)

    valid = np.isfinite(hits) & np.isfinite(rms) & (hits > 0) & (rms > 0)
    ref_y = np.full_like(rms, np.nan)
    if np.any(valid):
        i0 = np.where(valid)[0][0]
        A = rms[i0] * np.sqrt(hits[i0])
        ref_y[valid] = A / np.sqrt(hits[valid])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

    ax0.plot(hits, rms, "o-", label="cumulative")
    if np.any(np.isfinite(ref_y)):
        ax0.plot(hits, ref_y, "--", label="A/sqrt(hits)")
    for x, y, nf in zip(hits, rms, nfiles):
        if np.isfinite(x) and np.isfinite(y):
            ax0.text(x, y, str(nf), fontsize=7)
    ax0.set_xlabel("Mean hits (annulus)")
    ax0.set_ylabel("Median RMS (center)")
    ax0.set_title("Linear scale")
    ax0.grid(alpha=0.3)
    ax0.legend(loc="best")

    ax1.plot(hits[valid], rms[valid], "o-", label="cumulative")
    if np.any(np.isfinite(ref_y)):
        ax1.plot(hits[valid], ref_y[valid], "--", label="A/sqrt(hits)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Mean hits (annulus)")
    ax1.set_ylabel("Median RMS (center)")
    ax1.set_title("Log-log scale")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_plot, dpi=180)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()

    if args.min_obsid > args.max_obsid:
        raise ValueError("--min-obsid must be <= --max-obsid")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")
    if args.annulus_rin_pix >= args.annulus_rout_pix:
        raise ValueError("--annulus-rin-pix must be < --annulus-rout-pix")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    obsid_and_files = query_level2_files(
        database=args.database,
        source=args.source,
        min_obsid=args.min_obsid,
        max_obsid=args.max_obsid,
    )
    if not obsid_and_files:
        raise RuntimeError("No Level-2 files matched requested selection")

    obsid_and_files = sorted(obsid_and_files, key=lambda x: x[0])
    file_list = [f for _, f in obsid_and_files]
    chunks = chunk_list(file_list, args.chunk_size)

    print(f"Selected {len(file_list)} files in {len(chunks)} chunks")

    cumulative_weight = None
    cumulative_signal = None
    cumulative_hits = None
    center_mask = None
    annulus_mask = None
    base_header = None

    results: list[ChunkResult] = []

    with tempfile.TemporaryDirectory(prefix="chunked_mapmaking_", dir=str(output_dir)) as tmpdir:
        work_dir = Path(tmpdir)

        for chunk_idx, chunk_files in enumerate(chunks, start=1):
            out_fits = run_mapmaking_chunk(args, chunk_files, chunk_idx, work_dir)
            map_i, rms_i, hits_i, header = read_map_products(out_fits)

            good = np.isfinite(map_i) & np.isfinite(rms_i) & (rms_i > 0)
            weight_i = np.zeros_like(rms_i, dtype=float)
            weight_i[good] = 1.0 / (rms_i[good] ** 2)
            signal_i = np.zeros_like(map_i, dtype=float)
            signal_i[good] = map_i[good] * weight_i[good]
            hits_i = np.where(np.isfinite(hits_i), hits_i, 0.0)

            if cumulative_weight is None:
                cumulative_weight = np.zeros_like(weight_i)
                cumulative_signal = np.zeros_like(signal_i)
                cumulative_hits = np.zeros_like(hits_i)
                base_header = header
                center_mask, annulus_mask = build_region_masks(
                    shape=map_i.shape,
                    header=header,
                    lon=args.center_lon,
                    lat=args.center_lat,
                    r_center=args.center_radius_pix,
                    r_in=args.annulus_rin_pix,
                    r_out=args.annulus_rout_pix,
                )

            cumulative_weight += weight_i
            cumulative_signal += signal_i
            cumulative_hits += hits_i

            cumulative_map = np.full_like(cumulative_signal, np.nan, dtype=float)
            valid_w = cumulative_weight > 0
            cumulative_map[valid_w] = cumulative_signal[valid_w] / cumulative_weight[valid_w]

            cumulative_rms = np.full_like(cumulative_weight, np.nan, dtype=float)
            cumulative_rms[valid_w] = np.sqrt(1.0 / cumulative_weight[valid_w])

            rms_center_mean, rms_center_median, mean_hits_annulus = compute_region_stats(
                cumulative_rms,
                cumulative_hits,
                center_mask,
                annulus_mask,
            )

            n_files = min(chunk_idx * args.chunk_size, len(file_list))
            results.append(
                ChunkResult(
                    chunk_index=chunk_idx,
                    n_chunks=chunk_idx,
                    n_files=n_files,
                    rms_center_mean=rms_center_mean,
                    rms_center_median=rms_center_median,
                    mean_hits_annulus=mean_hits_annulus,
                )
            )

            if args.save_cumulative_npz:
                np.savez_compressed(
                    output_dir / f"cumulative_chunk{chunk_idx:03d}.npz",
                    map=cumulative_map,
                    rms=cumulative_rms,
                    hits=cumulative_hits,
                    weight=cumulative_weight,
                    signal=cumulative_signal,
                )

            if args.save_cumulative_fits and base_header is not None:
                h0 = fits.PrimaryHDU(cumulative_map)
                hr = fits.ImageHDU(cumulative_rms, name="RMS")
                hh = fits.ImageHDU(cumulative_hits, name="HITS")
                for hdu in (h0, hr, hh):
                    hdu.header.update(WCS(base_header).to_header())
                fits.HDUList([h0, hr, hh]).writeto(
                    output_dir / f"cumulative_chunk{chunk_idx:03d}.fits",
                    overwrite=True,
                )

    csv_path = output_dir / args.csv_name
    plot_path = output_dir / args.plot_name
    save_results_csv(results, csv_path)
    plot_results(results, plot_path)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
