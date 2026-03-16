#!/usr/bin/env python3
"""Build a Galactic map of per-pixel median MJD from COMAP Level-2 pointing."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import healpy as hp
import numpy as np
import toml
from astropy.io import fits
from astropy.io.fits import Header
from astropy.wcs import WCS

from modules.utils.Coordinates_py import comap_latitude, comap_longitude, h2e_full


DEFAULT_WCS_PATHS = [
    Path("map_making_configs/wcs_definitions.toml"),
    Path("modules/MapMaking/map_making_configs/wcs_definitions.toml"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-list", required=True, type=Path, help="Text file containing Level-2 HDF5 paths")
    parser.add_argument("--wcs-def", required=True, help="Key in wcs_definitions.toml")
    parser.add_argument("--output", required=True, type=Path, help="Output FITS file path")
    parser.add_argument(
        "--feeds",
        default=None,
        help="Optional feed subset, e.g. '1,2,5'",
    )
    parser.add_argument(
        "--coord-frame",
        default="C",
        help="Input coordinate frame code for RA/Dec before Galactic rotation (default: C)",
    )
    return parser.parse_args()


def parse_feeds(feeds_arg: str | None) -> set[int] | None:
    if feeds_arg is None:
        return None
    feeds = {int(tok.strip()) for tok in feeds_arg.split(",") if tok.strip()}
    if not feeds:
        raise ValueError("--feeds was provided but no valid feed IDs were parsed")
    return feeds


def resolve_wcs_config() -> Path:
    for path in DEFAULT_WCS_PATHS:
        if path.exists():
            return path
    joined = ", ".join(str(p) for p in DEFAULT_WCS_PATHS)
    raise FileNotFoundError(f"Could not locate WCS config. Tried: {joined}")


def load_wcs(wcs_def: str) -> tuple[WCS, Header, int, int, Path]:
    config_path = resolve_wcs_config()
    cfg_all = toml.load(config_path)
    if wcs_def not in cfg_all:
        raise KeyError(f"WCS definition '{wcs_def}' not found in {config_path}")

    header = Header(cfg_all[wcs_def])
    wcs = WCS(header)
    ny = int(header["NAXIS2"])
    nx = int(header["NAXIS1"])
    return wcs, header, ny, nx, config_path


def read_file_list(path: Path) -> list[Path]:
    files = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        files.append(Path(line))
    if not files:
        raise ValueError(f"No input files found in {path}")
    return files


def coordinate_transform(lon_in: np.ndarray, lat_in: np.ndarray, input_coord: str, output_coord: str = "G") -> tuple[np.ndarray, np.ndarray]:
    if input_coord == output_coord:
        return np.mod(lon_in, 360.0), lat_in
    rot = hp.rotator.Rotator(coord=[input_coord, output_coord])
    lon, lat = rot(lon_in, lat_in, lonlat=True)
    return np.mod(lon, 360.0), lat


def main() -> None:
    args = parse_args()

    file_paths = read_file_list(args.file_list)
    feeds_keep = parse_feeds(args.feeds)

    wcs, wcs_header, ny, nx, _ = load_wcs(args.wcs_def)
    npix = ny * nx

    mjd_per_pixel: dict[int, list[float]] = defaultdict(list)
    hits = np.zeros(npix, dtype=np.int64)

    for file_path in file_paths:
        with h5py.File(file_path, "r") as f:
            mjd = f["spectrometer/MJD"][...]
            feeds = f["spectrometer/feeds"][...].astype(int)
            az_all = f["spectrometer/pixel_pointing/pixel_az"][...]
            el_all = f["spectrometer/pixel_pointing/pixel_el"][...]

            n_feed = min(len(feeds), az_all.shape[0], el_all.shape[0])
            for ifeed in range(n_feed):
                feed = int(feeds[ifeed])
                if feeds_keep is not None and feed not in feeds_keep:
                    continue

                az = np.asarray(az_all[ifeed], dtype=np.float64)
                el = np.asarray(el_all[ifeed], dtype=np.float64)

                nsamp = min(mjd.size, az.size, el.size)
                if nsamp == 0:
                    continue

                mjd_s = np.asarray(mjd[:nsamp], dtype=np.float64)
                az_s = az[:nsamp]
                el_s = el[:nsamp]

                finite = np.isfinite(mjd_s) & np.isfinite(az_s) & np.isfinite(el_s)
                if not finite.any():
                    continue

                mjd_s = mjd_s[finite]
                az_s = az_s[finite]
                el_s = el_s[finite]

                ra, dec = h2e_full(az_s, el_s, mjd_s, comap_longitude, comap_latitude)
                gl, gb = coordinate_transform(ra, dec, args.coord_frame, "G")

                xpix, ypix = wcs.all_world2pix(gl, gb, 0)
                in_bounds = (xpix >= 0) & (xpix < nx) & (ypix >= 0) & (ypix < ny)
                if not in_bounds.any():
                    continue

                xi = xpix[in_bounds].astype(int)
                yi = ypix[in_bounds].astype(int)
                mjd_good = mjd_s[in_bounds]
                pix = np.ravel_multi_index((yi, xi), (ny, nx))

                for p, m in zip(pix, mjd_good, strict=False):
                    mjd_per_pixel[int(p)].append(float(m))
                    hits[int(p)] += 1

    median_map = np.full(npix, np.nan, dtype=np.float32)
    for p, values in mjd_per_pixel.items():
        median_map[p] = np.float32(np.median(np.asarray(values, dtype=np.float64)))

    median_map_2d = median_map.reshape(ny, nx)
    hits_2d = hits.reshape(ny, nx)

    primary_hdu = fits.PrimaryHDU(data=median_map_2d)
    primary_hdu.header.extend(wcs_header, update=True)
    primary_hdu.header["PRODUCT"] = "MEDIAN_MJD"
    primary_hdu.header["WCS_DEF"] = args.wcs_def
    primary_hdu.header["FILELIST"] = str(args.file_list)
    primary_hdu.header["DATE"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    hits_hdu = fits.ImageHDU(data=hits_2d, name="HITS")

    hdul = fits.HDUList([primary_hdu, hits_hdu])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hdul.writeto(args.output, overwrite=True)

    print(f"Wrote median-MJD map: {args.output}")
    print(f"Input files: {len(file_paths)}")
    print(f"Filled pixels: {np.count_nonzero(np.isfinite(median_map_2d))}")


if __name__ == "__main__":
    main()
