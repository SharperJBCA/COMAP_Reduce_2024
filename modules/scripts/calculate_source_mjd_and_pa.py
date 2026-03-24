#!/usr/bin/env python3
"""Calculate mean MJD and pointing-frame rotation angles for fitted radio sources.

For each source row in the input CSV:
  1. Match the source's reference Galactic position to the nearest COMAP field center.
  2. Query the SQL database for Level-2 HDF5 files observed on that field.
  3. Derive the mean MJD from those files.
  4. Compute two rotation angles required to transform pointing offsets from the
     Galactic (Δl, Δb) frame to the telescope Alt-Az (ΔAz, ΔEl) frame:

       η  (galactic_pa_deg)       — position angle of Galactic North measured East of
                                    North in the ICRS equatorial frame at the source
                                    position (time-independent, Galactic → Equatorial).

       q  (parallactic_angle_deg) — parallactic angle at mean_mjd
                                    (Equatorial → Alt-Az).

     The total rotation angle from Galactic North to the telescope elevation axis is η + q.

Output: the input CSV with five new columns appended:
  matched_field, angular_sep_deg, n_l2_files, mean_mjd,
  galactic_pa_deg, parallactic_angle_deg
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, Galactic
import astropy.units as u

# Allow imports from the project root (two levels up from this script)
sys.path.insert(0, str(Path(__file__).parents[2]))

from modules.utils.Coordinates_py import (
    sex2deg,
    g2e,
    pa,
    AngularSeperation,
    comap_longitude,
    comap_latitude,
)
from modules.SQLModule.SQLModule import SQLModule, COMAPData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ra_sex2deg(ra_str: str) -> float:
    """Convert sexagesimal RA string (HH:MM:SS) to decimal degrees."""
    return sex2deg(ra_str, hours=True)


def _dec_sex2deg(dec_str: str) -> float:
    """Convert sexagesimal Dec string (DD:MM:SS or +DD:MM:SS) to decimal degrees."""
    return sex2deg(dec_str, hours=False)


def load_field_centers(path: Path) -> pd.DataFrame:
    """Load field_centers.csv and attach Galactic l/b columns.

    Accepts files with columns (field_name, ra, dec, ...) where ra and dec are
    sexagesimal strings.  Galactic coordinates are derived from RA/Dec via
    astropy SkyCoord regardless of whether the file already contains l/b.
    """
    df = pd.read_csv(path)
    # Normalise column names
    df.columns = [c.strip() for c in df.columns]

    # Determine the field-name column (first column)
    name_col = df.columns[0]
    df = df.rename(columns={name_col: "field_name"})

    # Identify RA/Dec columns (case-insensitive)
    col_map = {c.lower(): c for c in df.columns}
    ra_col  = col_map.get("ra",  col_map.get("ra [deg]",  None))
    dec_col = col_map.get("dec", col_map.get("dec [deg]", None))
    if ra_col is None or dec_col is None:
        raise ValueError(f"Cannot find RA/Dec columns in {path}. Found: {list(df.columns)}")

    # Convert sexagesimal → decimal degrees
    ra_deg  = df[ra_col].apply(_ra_sex2deg).to_numpy(dtype=float)
    dec_deg = df[dec_col].apply(_dec_sex2deg).to_numpy(dtype=float)

    # Equatorial → Galactic
    sc = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    g  = sc.galactic
    df["field_l"] = g.l.deg
    df["field_b"] = g.b.deg

    return df


def match_source_to_field(
    source_l: float,
    source_b: float,
    field_l: np.ndarray,
    field_b: np.ndarray,
    max_sep: float,
) -> tuple[int, float]:
    """Return (best_index, separation_deg) or (-1, nan) if none within max_sep."""
    seps = AngularSeperation(source_l, source_b, field_l, field_b)
    idx  = int(np.argmin(seps))
    sep  = float(seps[idx])
    if sep > max_sep:
        return -1, float("nan")
    return idx, sep


def get_level2_paths(db_path: str, field_name: str) -> list[str]:
    """Query the SQL database for level-2 paths belonging to field_name."""
    db = SQLModule()
    db.connect(db_path)
    db._connect()
    try:
        rows = (
            db.session.query(COMAPData.level2_path)
            .filter(
                COMAPData.source == field_name,
                COMAPData.level2_path.isnot(None),
                COMAPData.source_group.in_(["Galactic", "Foreground"]),
            )
            .all()
        )
        paths = [r.level2_path for r in rows if r.level2_path]
    finally:
        db._disconnect()
    return paths


def mean_mjd_from_files(paths: list[str]) -> tuple[float, int]:
    """Return (mean_mjd, n_files) by reading only the first and last MJD sample."""
    midpoints: list[float] = []
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            with h5py.File(p, "r") as f:
                mjd_ds = f["spectrometer/MJD"]
                n = mjd_ds.shape[0]
                if n == 0:
                    continue
                first = float(mjd_ds[0])
                last  = float(mjd_ds[-1]) if n > 1 else first
                midpoints.append(0.5 * (first + last))
        except (OSError, KeyError):
            continue
    if not midpoints:
        return float("nan"), 0
    return float(np.mean(midpoints)), len(midpoints)


def galactic_position_angle(source_l: float, source_b: float) -> float:
    """Return η: position angle of Galactic North, East of North in ICRS frame."""
    src_icrs = SkyCoord(l=source_l * u.deg, b=source_b * u.deg, frame="galactic").icrs
    gnp_icrs = SkyCoord(l=0.0 * u.deg, b=90.0 * u.deg, frame="galactic").icrs
    eta = src_icrs.position_angle(gnp_icrs).deg
    return float(eta)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources",       required=True, type=Path, help="Input source CSV")
    p.add_argument("--field-centers", required=True, type=Path, help="Field-centers CSV (field_name, ra, dec, ...)")
    p.add_argument("--database",      required=True, type=str,  help="Path to SQLite database (comap.db)")
    p.add_argument("--output",        required=True, type=Path, help="Output CSV path")
    p.add_argument("--ref-lon-col",   default="lon_paladini [deg]", help="Source table column for reference Galactic longitude")
    p.add_argument("--ref-lat-col",   default="lat_paladini [deg]", help="Source table column for reference Galactic latitude")
    p.add_argument("--max-sep",       default=1.0, type=float, help="Maximum angular separation (deg) for field matching (default: 1.0)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- Load inputs ---
    print(f"Loading source table: {args.sources}")
    sources_df = pd.read_csv(args.sources)
    sources_df.columns = [c.strip() for c in sources_df.columns]

    if args.ref_lon_col not in sources_df.columns:
        raise KeyError(f"Reference longitude column '{args.ref_lon_col}' not found. "
                       f"Available: {list(sources_df.columns)}")
    if args.ref_lat_col not in sources_df.columns:
        raise KeyError(f"Reference latitude column '{args.ref_lat_col}' not found. "
                       f"Available: {list(sources_df.columns)}")

    print(f"Loading field centers: {args.field_centers}")
    fields_df = load_field_centers(args.field_centers)
    field_names = fields_df["field_name"].to_numpy()
    field_l     = fields_df["field_l"].to_numpy(dtype=float)
    field_b     = fields_df["field_b"].to_numpy(dtype=float)

    # Cache: field_name → (mean_mjd, n_files)
    field_mjd_cache: dict[str, tuple[float, int]] = {}

    # Output columns
    matched_field_col   = []
    angular_sep_col     = []
    n_l2_files_col      = []
    mean_mjd_col        = []
    galactic_pa_col     = []
    parallactic_pa_col  = []

    n_rows = len(sources_df)
    print(f"Processing {n_rows} sources …")

    for i, row in sources_df.iterrows():
        src_l = row[args.ref_lon_col]
        src_b = row[args.ref_lat_col]

        if not (np.isfinite(src_l) and np.isfinite(src_b)):
            matched_field_col.append(None)
            angular_sep_col.append(float("nan"))
            n_l2_files_col.append(0)
            mean_mjd_col.append(float("nan"))
            galactic_pa_col.append(float("nan"))
            parallactic_pa_col.append(float("nan"))
            continue

        # --- Match to nearest field ---
        idx, sep = match_source_to_field(src_l, src_b, field_l, field_b, args.max_sep)

        if idx < 0:
            print(f"  Source {i}: no field within {args.max_sep}° (nearest sep={sep:.3f}°)")
            matched_field_col.append(None)
            angular_sep_col.append(sep)
            n_l2_files_col.append(0)
            mean_mjd_col.append(float("nan"))
            galactic_pa_col.append(float("nan"))
            parallactic_pa_col.append(float("nan"))
            continue

        fname = str(field_names[idx])
        matched_field_col.append(fname)
        angular_sep_col.append(sep)

        # --- Get mean MJD (cached per field) ---
        if fname not in field_mjd_cache:
            paths = get_level2_paths(args.database, fname)
            mmjd, nf = mean_mjd_from_files(paths)
            field_mjd_cache[fname] = (mmjd, nf)

        mean_mjd, n_files = field_mjd_cache[fname]
        n_l2_files_col.append(n_files)
        mean_mjd_col.append(mean_mjd)

        # --- Galactic position angle (time-independent) ---
        eta = galactic_position_angle(src_l, src_b)
        galactic_pa_col.append(eta)

        # --- Parallactic angle at mean MJD ---
        if np.isfinite(mean_mjd):
            ra, dec = g2e(src_l, src_b)
            q = pa(np.atleast_1d(ra), np.atleast_1d(dec),
                   np.atleast_1d(mean_mjd),
                   comap_longitude, comap_latitude)
            parallactic_pa_col.append(float(np.mean(q)))
        else:
            parallactic_pa_col.append(float("nan"))

    # --- Attach new columns ---
    sources_df["matched_field"]         = matched_field_col
    sources_df["angular_sep_deg"]       = angular_sep_col
    sources_df["n_l2_files"]            = n_l2_files_col
    sources_df["mean_mjd"]              = mean_mjd_col
    sources_df["galactic_pa_deg"]       = galactic_pa_col
    sources_df["parallactic_angle_deg"] = parallactic_pa_col

    # --- Write output ---
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sources_df.to_csv(args.output, index=False)
    print(f"\nWrote {len(sources_df)} rows → {args.output}")

    matched   = sources_df["matched_field"].notna().sum()
    has_mjd   = sources_df["mean_mjd"].notna().sum()
    print(f"  Matched to a field:       {matched}/{n_rows}")
    print(f"  With mean MJD computed:   {has_mjd}/{n_rows}")


if __name__ == "__main__":
    main()
