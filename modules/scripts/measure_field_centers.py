#!/usr/bin/env python3
"""Measure white-noise RMS and hits at field centres from a FITS map.

Expected FITS HDU layout:
- HDU 0: primary map (for WCS)
- HDU 3: expected white-noise map
- HDU 2: hits map   (NOTE: your original comment said HDU 4, but code uses HDU 2)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

import matplotlib.pyplot as plt

GALACTIC_KEYWORDS = ("fg12", "fg6", "gfield", "field")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map", type=Path, help="Input FITS map")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/field_centers.csv"),
        help="CSV of field centres (default: data/field_centers.csv)",
    )
    parser.add_argument(
        "--source-name",
        type=str,
        default=None,
        help=(
            "Source label used to choose which fields to measure. "
            "If omitted, the map filename is used."
        ),
    )
    parser.add_argument(
        "--aperture-pix",
        type=float,
        default=10.5,
        help="Aperture radius in pixels for robust median summaries (default: 10.5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <map_stem>_field_metrics.csv)",
    )
    return parser.parse_args()


def read_catalog(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def is_galactic_plane_map(source_name: str) -> bool:
    source = source_name.lower()
    return any(key in source for key in GALACTIC_KEYWORDS)


def choose_fields(catalog: list[dict[str, str]], source_name: str) -> list[dict[str, str]]:
    if is_galactic_plane_map(source_name):
        return [row for row in catalog if row["group"].strip().lower() == "galactic_plane"]

    source_lower = source_name.lower()
    exact = [row for row in catalog if row["field_name"].lower() == source_lower]
    if exact:
        return exact

    # Fallback: targeted group only if source not in catalogue
    return [row for row in catalog if row["group"].strip().lower() == "targeted"]


def extract_2d(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Could not reduce array to 2D; got shape={np.shape(data)}")
    return arr


def build_wcs(header) -> WCS:
    return WCS(header).celestial


def circular_mask(shape: tuple[int, int], y0: float, x0: float, r: float) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - y0) ** 2 + (xx - x0) ** 2 <= r**2


def finite_stats(values: np.ndarray) -> tuple[float, float, float]:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.nanmedian(vals)), float(np.nanpercentile(vals, 16)), float(np.nanpercentile(vals, 84))


def measure_fields(
    fields: Iterable[dict[str, str]],
    wcs: WCS,
    rms_map: np.ndarray,
    hits_map: np.ndarray,
    aperture_pix: float,
) -> list[dict[str, float | str | int | bool]]:
    out: list[dict[str, float | str | int | bool]] = []
    ny, nx = rms_map.shape

    for row in fields:
        coord = SkyCoord(row["ra"], row["dec"], unit=(u.hourangle, u.deg), frame="icrs")
        x, y = wcs.world_to_pixel(coord)
        xi, yi = int(np.rint(x)), int(np.rint(y))

        inside = 0 <= xi < nx and 0 <= yi < ny
        result: dict[str, float | str | int | bool] = {
            "field_name": row["field_name"],
            "group": row["group"],
            "ra": row["ra"],
            "dec": row["dec"],
            "x_pix": int(float(x)),
            "y_pix": int(float(y)),
            "inside_map": inside,
            "rms_ap_med": np.nan,
            "hits_ap_med": np.nan,
            "n_ap_pix": 0,
        }

        if inside:
            mask = circular_mask(rms_map.shape, y, x, aperture_pix)
            rms_vals = rms_map[mask]
            hits_vals = hits_map[mask]

            rms_med, _, _ = finite_stats(rms_vals)
            hits_med, _, _ = finite_stats(hits_vals)

            result["rms_ap_med"] = float(rms_med)
            result["hits_ap_med"] = float(hits_med)
            result["n_ap_pix"] = int(mask.sum())

        out.append(result)

    return out


def write_output(path: Path, rows: list[dict[str, float | str | int | bool]]) -> None:
    if not rows:
        raise ValueError("No rows to write")

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rows_to_gal_longitude(rows: list[dict[str, float | str | int | bool]]) -> tuple[np.ndarray, np.ndarray]:
    """Return (l_deg, ok_mask) from rows, computing l from ra/dec."""
    l_list = []
    ok = []

    for r in rows:
        inside = bool(r.get("inside_map", False))
        ra = str(r["ra"])
        dec = str(r["dec"])
        coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame="icrs")
        l_deg = coord.galactic.l.wrap_at(360 * u.deg).deg  # 0..360-ish

        l_list.append(l_deg)
        ok.append(inside)

    return np.asarray(l_list, dtype=float), np.asarray(ok, dtype=bool)


def plot_galactic_plane(rows: list[dict[str, float | str | int | bool]], out_csv: Path) -> None:
    """If rows are galactic_plane, write RMS vs l and hits vs l plots beside the CSV."""
    # Only plot for galactic_plane group rows (and only those inside map with finite values)
    #groups = {str(r.get("group", "")).strip().lower() for r in rows}
    #if groups != {"galactic_plane"} and "galactic_plane" not in groups:
    #    return

    l_deg, inside = _rows_to_gal_longitude(rows)

    names = np.asarray([str(r["field_name"]) for r in rows])
    rms = np.asarray([float(r["rms_ap_med"]) for r in rows], dtype=float)
    hits = np.asarray([float(r["hits_ap_med"]) for r in rows], dtype=float)

    ok_rms = inside & np.isfinite(l_deg) & np.isfinite(rms)
    ok_hits = inside & np.isfinite(l_deg) & np.isfinite(hits)

    # Sort by longitude for nicer labeling/reading
    def _sorted(arr_mask: np.ndarray):
        idx = np.argsort(l_deg[arr_mask])
        return (
            l_deg[arr_mask][idx],
            idx,
        )

    # --- RMS vs l ---
    if np.any(ok_rms):
        l1, idx1 = _sorted(ok_rms)
        rms1 = rms[ok_rms][idx1]
        names1 = names[ok_rms][idx1]

        plt.figure(figsize=(24,8))
        plt.scatter(l1, rms1)
        for x, y, n in zip(l1, rms1, names1):
            plt.annotate(n, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        plt.xlabel("Galactic longitude l (deg)")
        plt.ylabel("RMS aperture median")
        plt.title("Galactic plane: RMS vs Galactic longitude")
        plt.grid(True, alpha=0.3)

        out_png = out_csv.with_name(out_csv.stem.replace("_field_metrics", "") + "_rms_vs_galactic_longitude.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

    # --- Hits vs l ---
    if np.any(ok_hits):
        l2, idx2 = _sorted(ok_hits)
        hits2 = hits[ok_hits][idx2]
        names2 = names[ok_hits][idx2]

        plt.figure(figsize=(24,8))
        plt.scatter(l2, hits2)
        for x, y, n in zip(l2, hits2, names2):
            plt.annotate(n, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        plt.xlabel("Galactic longitude l (deg)")
        plt.ylabel("Hits aperture median")
        plt.title("Galactic plane: Hits vs Galactic longitude")
        plt.grid(True, alpha=0.3)

        out_png = out_csv.with_name(out_csv.stem.replace("_field_metrics", "") + "_hits_vs_galactic_longitude.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()


def main() -> None:
    args = parse_args()

    source_name = args.source_name or args.map.stem
    catalog = read_catalog(args.catalog)
    fields = choose_fields(catalog, source_name)

    with fits.open(args.map) as hdul:
        wcs = build_wcs(hdul[0].header)
        rms_map = extract_2d(hdul[3].data)
        hits_map = extract_2d(hdul[2].data)

    rows = measure_fields(fields, wcs, rms_map, hits_map, args.aperture_pix)
    output = args.output or args.map.with_name(f"{args.map.stem}_field_metrics.csv")
    write_output(output, rows)

    print(rows)
    # If the selection corresponds to galactic plane, output the two plots
    if is_galactic_plane_map(source_name):
        plot_galactic_plane(rows, output)

    print(f"Measured {len(rows)} fields from source selector '{source_name}'.")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()