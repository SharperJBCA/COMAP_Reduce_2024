#!/usr/bin/env python3
"""Measure white-noise RMS and hits at field centres from a FITS map.

Expected FITS HDU layout:
- HDU 0: primary map (for WCS)
- HDU 3: expected white-noise map
- HDU 4: hits map
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u

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
        default=1.5,
        help="Aperture radius in pixels for robust median summaries (default: 1.5)",
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
            "x_pix": float(x),
            "y_pix": float(y),
            "inside_map": inside,
            "rms_center": np.nan,
            "hits_center": np.nan,
            "rms_ap_med": np.nan,
            "rms_ap_p16": np.nan,
            "rms_ap_p84": np.nan,
            "hits_ap_med": np.nan,
            "hits_ap_p16": np.nan,
            "hits_ap_p84": np.nan,
            "n_ap_pix": 0,
        }

        if inside:
            result["rms_center"] = float(rms_map[yi, xi])
            result["hits_center"] = float(hits_map[yi, xi])

            mask = circular_mask(rms_map.shape, y, x, aperture_pix)
            rms_vals = rms_map[mask]
            hits_vals = hits_map[mask]
            rms_med, rms_p16, rms_p84 = finite_stats(rms_vals)
            hits_med, hits_p16, hits_p84 = finite_stats(hits_vals)
            result["rms_ap_med"] = rms_med
            result["rms_ap_p16"] = rms_p16
            result["rms_ap_p84"] = rms_p84
            result["hits_ap_med"] = hits_med
            result["hits_ap_p16"] = hits_p16
            result["hits_ap_p84"] = hits_p84
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


def main() -> None:
    args = parse_args()

    source_name = args.source_name or args.map.stem
    catalog = read_catalog(args.catalog)
    fields = choose_fields(catalog, source_name)

    with fits.open(args.map) as hdul:
        wcs = build_wcs(hdul[0].header)
        rms_map = extract_2d(hdul[3].data)
        hits_map = extract_2d(hdul[4].data)

    rows = measure_fields(fields, wcs, rms_map, hits_map, args.aperture_pix)
    output = args.output or args.map.with_name(f"{args.map.stem}_field_metrics.csv")
    write_output(output, rows)

    print(f"Measured {len(rows)} fields from source selector '{source_name}'.")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
