"""
fit_hii_pointing.py

For each HII region map produced by the LST-grouped map-making pipeline:

  1. Read the FITS map and its MEAN_AZ, MEAN_EL, MEAN_MJD header keywords.
  2. Fit a 2D Gaussian + planar background to the source at the expected
     Galactic (l, b) position from the HII catalogue.
  3. Convert the fitted Galactic peak to (RA, Dec) and then to (az, el) at
     the mean MJD of the observation, giving the *fitted* az/el.
  4. Compute delta_az = fitted_az - mean_az, delta_el = fitted_el - mean_el.
  5. Collect all results into a pointing-offset table (CSV).

Usage:
    python modules/scripts/fit_hii_pointing.py \\
        --map-dir hii_field_tables/maps \\
        --summary hii_field_match_summary.csv \\
        --output  pointing_offsets.csv

    # Override search box half-width and initial FWHM guess
    python modules/scripts/fit_hii_pointing.py \\
        --map-dir hii_field_tables/maps \\
        --summary hii_field_match_summary.csv \\
        --box-radius 0.5 --init-fwhm 0.1
"""

import argparse
import glob
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, Galactic, ICRS
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import least_squares
import astropy.units as u

# COMAP site coordinates
COMAP_LON = -(118 + 16.0 / 60.0 + 56.0 / 3600.0)  # deg West (negative)
COMAP_LAT = 37.0 + 14.0 / 60.0 + 2.0 / 3600.0      # deg North
COMAP_HEIGHT = 1222.0                                  # metres

from astropy.coordinates import AltAz, EarthLocation
from astropy.time import Time

COMAP_LOCATION = EarthLocation.from_geodetic(
    COMAP_LON * u.deg, COMAP_LAT * u.deg, COMAP_HEIGHT * u.m
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 2-D Gaussian + plane model
# ---------------------------------------------------------------------------

def gaussian2d_plane(params, x, y):
    """
    Model: 2-D Gaussian peak + tilted plane background.

    params = [amplitude, x0, y0, sigma_x, sigma_y, theta,
              bg_offset, bg_grad_x, bg_grad_y]

    (x, y) are pixel coordinates.
    theta is the position angle of the major axis (radians).
    """
    amp, x0, y0, sx, sy, theta, bg0, bg_gx, bg_gy = params
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    dx = x - x0
    dy = y - y0
    a = (cos_t / sx) ** 2 + (sin_t / sy) ** 2
    b = 2 * sin_t * cos_t * (1 / sx**2 - 1 / sy**2)
    c = (sin_t / sx) ** 2 + (cos_t / sy) ** 2
    gauss = amp * np.exp(-0.5 * (a * dx**2 + b * dx * dy + c * dy**2))
    plane = bg0 + bg_gx * dx + bg_gy * dy
    return gauss + plane


def residuals(params, x, y, data, weights):
    return (gaussian2d_plane(params, x, y) - data) * weights


# ---------------------------------------------------------------------------
# Fitting driver
# ---------------------------------------------------------------------------

def fit_source_in_map(
    map_data: np.ndarray,
    wcs: WCS,
    source_glon: float,
    source_glat: float,
    rms_data: np.ndarray = None,
    box_radius_deg: float = 0.5,
    init_fwhm_deg: float = 0.1,
) -> dict:
    """
    Fit a 2-D Gaussian + plane to the source at (source_glon, source_glat).

    Returns a dict with fitted parameters or None on failure.
    """
    ny, nx = map_data.shape

    # Source pixel
    sx_pix, sy_pix = wcs.all_world2pix(source_glon, source_glat, 0)
    sx_pix, sy_pix = float(sx_pix), float(sy_pix)

    if not (0 <= sx_pix < nx and 0 <= sy_pix < ny):
        log.warning("Source (%.4f, %.4f) falls outside map — skipping.", source_glon, source_glat)
        return None

    # Pixel scale (degrees)
    cdelt = abs(float(wcs.wcs.cdelt[0]))

    # Cutout box
    box_pix = int(np.ceil(box_radius_deg / cdelt))
    x0 = max(0, int(sx_pix) - box_pix)
    x1 = min(nx, int(sx_pix) + box_pix + 1)
    y0 = max(0, int(sy_pix) - box_pix)
    y1 = min(ny, int(sy_pix) + box_pix + 1)

    cutout = map_data[y0:y1, x0:x1]
    if rms_data is not None:
        rms_cut = rms_data[y0:y1, x0:x1]
    else:
        rms_cut = None

    # Mask NaNs
    valid = np.isfinite(cutout)
    if rms_cut is not None:
        valid &= np.isfinite(rms_cut) & (rms_cut > 0)
    if valid.sum() < 10:
        log.warning("Too few valid pixels (%d) in cutout — skipping.", valid.sum())
        return None

    yy, xx = np.mgrid[0 : cutout.shape[0], 0 : cutout.shape[1]]
    # Centre of source in cutout coordinates
    cx = sx_pix - x0
    cy = sy_pix - y0

    data_flat = cutout[valid]
    x_flat = xx[valid].astype(float)
    y_flat = yy[valid].astype(float)

    if rms_cut is not None:
        w_flat = 1.0 / rms_cut[valid]
    else:
        w_flat = np.ones_like(data_flat)

    # Initial guess
    init_sigma = init_fwhm_deg / (2.355 * cdelt)  # FWHM → sigma in pixels
    peak_val = float(np.nanmax(cutout[valid]))
    bg_est = float(np.nanmedian(cutout[valid]))
    p0 = [
        peak_val - bg_est,  # amplitude
        cx,                 # x0
        cy,                 # y0
        init_sigma,         # sigma_x
        init_sigma,         # sigma_y
        0.0,                # theta
        bg_est,             # bg_offset
        0.0,                # bg_grad_x
        0.0,                # bg_grad_y
    ]

    try:
        result = least_squares(
            residuals,
            p0,
            args=(x_flat, y_flat, data_flat, w_flat),
            method="lm",
            max_nfev=5000,
        )
    except Exception as exc:
        log.warning("Fit failed: %s", exc)
        return None

    if not result.success:
        log.warning("Fit did not converge: %s", result.message)
        return None

    pfit = result.x

    # Parameter covariance from the Jacobian
    # J^T J ≈ Hessian/2 → cov ≈ s² (J^T J)^{-1}
    J = result.jac
    residual_variance = float(np.sum(result.fun**2) / max(len(result.fun) - len(pfit), 1))
    try:
        cov = np.linalg.inv(J.T @ J) * residual_variance
        perr = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        perr = np.full(len(pfit), np.nan)

    # Convert fitted peak pixel back to world coordinates
    fit_xpix = pfit[1] + x0
    fit_ypix = pfit[2] + y0
    fit_glon, fit_glat = wcs.all_pix2world(fit_xpix, fit_ypix, 0)
    fit_glon, fit_glat = float(fit_glon), float(fit_glat)

    # Fitted FWHM in degrees
    fwhm_x = abs(pfit[3]) * cdelt * 2.355
    fwhm_y = abs(pfit[4]) * cdelt * 2.355

    # Build full 2-D model and residual images on the cutout grid
    model_2d = gaussian2d_plane(pfit, xx.astype(float), yy.astype(float))
    resid_2d = cutout - model_2d
    resid_2d[~valid] = np.nan

    # Compute residual RMS inside the cutout
    resid_rms = float(np.sqrt(np.nanmean(resid_2d[valid] ** 2)))

    # World-coordinate extent of cutout [l_min, l_max, b_min, b_max]
    l_lo, b_lo = wcs.all_pix2world(x0, y0, 0)
    l_hi, b_hi = wcs.all_pix2world(x1 - 1, y1 - 1, 0)
    wcs_extent = [float(l_hi), float(l_lo), float(b_lo), float(b_hi)]

    # Position uncertainties: pixel → degrees
    # perr[1] = sigma(x0_pix), perr[2] = sigma(y0_pix)
    pos_err_lon_deg = float(perr[1]) * cdelt  # GLON error
    pos_err_lat_deg = float(perr[2]) * cdelt  # GLAT error

    return {
        "fit_glon": fit_glon,
        "fit_glat": fit_glat,
        "amplitude": float(pfit[0]),
        "amplitude_err": float(perr[0]),
        "fwhm_x_deg": fwhm_x,
        "fwhm_y_deg": fwhm_y,
        "theta_deg": float(np.degrees(pfit[5])),
        "bg_offset": float(pfit[6]),
        "resid_rms": resid_rms,
        "pos_err_lon_deg": pos_err_lon_deg,
        "pos_err_lat_deg": pos_err_lat_deg,
        "fit_xpix": float(fit_xpix),
        "fit_ypix": float(fit_ypix),
        # For plotting
        "cutout": cutout,
        "model_2d": model_2d,
        "resid_2d": resid_2d,
        "wcs_extent": wcs_extent,
        "source_glon": source_glon,
        "source_glat": source_glat,
    }


# ---------------------------------------------------------------------------
# Coordinate conversions
# ---------------------------------------------------------------------------

def galactic_to_radec(glon_deg: float, glat_deg: float):
    """Galactic (l, b) → ICRS (ra, dec) in degrees."""
    sc = SkyCoord(l=glon_deg * u.deg, b=glat_deg * u.deg, frame=Galactic())
    icrs = sc.icrs
    return float(icrs.ra.deg), float(icrs.dec.deg)


def radec_to_azel(ra_deg: float, dec_deg: float, mjd: float):
    """ICRS (ra, dec) → horizon (az, el) at COMAP site for the given MJD."""
    t = Time(mjd, format="mjd")
    sc = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame=ICRS())
    altaz = sc.transform_to(AltAz(obstime=t, location=COMAP_LOCATION))
    return float(altaz.az.deg), float(altaz.alt.deg)


def radec_to_azel_from_lst(ra_deg: float, dec_deg: float, lst_hours: float):
    """
    Convert RA/Dec to az/el using the Local Sidereal Time directly.

    This avoids the problem of MEAN_MJD being scrambled when observations
    span many calendar dates at the same LST — the sidereal time at the
    averaged MJD bears no relation to the actual common LST of the group.

    Uses standard spherical-trig (HA, Dec, Lat) → (az, el) formulas.
    Azimuth is measured from North through East (astronomical convention).
    """
    ha_deg = lst_hours * 15.0 - ra_deg
    ha_rad = np.radians(ha_deg)
    dec_rad = np.radians(dec_deg)
    lat_rad = np.radians(COMAP_LAT)

    sin_alt = (np.sin(dec_rad) * np.sin(lat_rad)
               + np.cos(dec_rad) * np.cos(lat_rad) * np.cos(ha_rad))
    alt_rad = np.arcsin(np.clip(sin_alt, -1.0, 1.0))

    cos_az = ((np.sin(dec_rad) - np.sin(alt_rad) * np.sin(lat_rad))
              / (np.cos(alt_rad) * np.cos(lat_rad)))
    cos_az = np.clip(cos_az, -1.0, 1.0)
    az_rad = np.arccos(cos_az)

    # If HA > 0 (source is west of meridian), az > 180°
    if np.sin(ha_rad) > 0:
        az_rad = 2.0 * np.pi - az_rad

    return np.degrees(az_rad), np.degrees(alt_rad)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_fit(fit: dict, plot_path: str, title: str = ""):
    """
    Three-panel figure: data cutout | model | residual.
    Each panel shows Galactic coordinate axes and marks the catalogue
    and fitted source positions.
    """
    cutout = fit["cutout"].copy()
    model = fit["model_2d"]
    resid = fit["resid_2d"]
    extent = fit["wcs_extent"]  # [l_hi, l_lo, b_lo, b_hi]

    # Shared colour scale from data
    vmin = np.nanpercentile(cutout[np.isfinite(cutout)], 1)
    vmax = np.nanpercentile(cutout[np.isfinite(cutout)], 99)
    # Residual colour scale (symmetric)
    resid_lim = np.nanpercentile(np.abs(resid[np.isfinite(resid)]), 99)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    panels = [
        (cutout, "Data", vmin, vmax, "RdBu_r"),
        (model, "Model", vmin, vmax, "RdBu_r"),
        (resid, "Residual", -resid_lim, resid_lim, "RdBu_r"),
    ]

    src_l = fit["source_glon"]
    src_b = fit["source_glat"]
    fit_l = fit["fit_glon"]
    fit_b = fit["fit_glat"]

    for ax, (data, label, v0, v1, cmap) in zip(axes, panels):
        im = ax.imshow(
            data,
            origin="lower",
            extent=extent,
            aspect="equal",
            vmin=v0,
            vmax=v1,
            cmap=cmap,
            interpolation="nearest",
        )
        fig.colorbar(im, ax=ax, shrink=0.85, label="K")
        # Mark catalogue position
        ax.plot(src_l, src_b, "x", color="black", ms=8, mew=1.5, label="Catalogue")
        # Mark fitted position
        ax.plot(fit_l, fit_b, "+", color="lime", ms=10, mew=2, label="Fit")
        # FWHM ellipse on data panel
        if label == "Data":
            ell = Ellipse(
                (fit_l, fit_b),
                width=fit["fwhm_x_deg"],
                height=fit["fwhm_y_deg"],
                angle=fit["theta_deg"],
                edgecolor="lime",
                facecolor="none",
                lw=1.5,
                ls="--",
                label="FWHM",
            )
            ax.add_patch(ell)
            ax.legend(loc="upper right", fontsize=7)
        ax.set_xlabel("GLON [deg]")
        ax.set_ylabel("GLAT [deg]")
        ax.set_title(label)

    fig.suptitle(title, fontsize=10)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    log.info("    Plot saved: %s", plot_path)


def plot_offsets_vs_azel(df: pd.DataFrame, plot_dir: str):
    """
    Four-panel summary: delta_az vs az, delta_az vs el,
                        delta_el vs az, delta_el vs el.
    All offsets in arcmin with error bars.
    """
    os.makedirs(plot_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    panels = [
        (axes[0, 0], "pred_az", "delta_az_arcmin", "err_az_arcmin",
         "Azimuth [deg]", r"$\Delta$az $\cos$(el) [arcmin]"),
        (axes[0, 1], "pred_el", "delta_az_arcmin", "err_az_arcmin",
         "Elevation [deg]", r"$\Delta$az $\cos$(el) [arcmin]"),
        (axes[1, 0], "pred_az", "delta_el_arcmin", "err_el_arcmin",
         "Azimuth [deg]", r"$\Delta$el [arcmin]"),
        (axes[1, 1], "pred_el", "delta_el_arcmin", "err_el_arcmin",
         "Elevation [deg]", r"$\Delta$el [arcmin]"),
    ]

    for ax, xcol, ycol, ecol, xlabel, ylabel in panels:
        ax.errorbar(
            df[xcol], df[ycol], yerr=df[ecol],
            fmt="o", ms=4, capsize=2, elinewidth=0.8, color="C0",
        )
        ax.axhline(0, color="grey", ls="--", lw=0.8)
        med = df[ycol].median()
        ax.axhline(med, color="C1", ls=":", lw=1, label=f"median = {med:.2f}'")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.set_ylim(-2,2)

    fig.suptitle(
        f"Pointing offsets ({len(df)} fits)  |  "
        f"median daz*cos={df['delta_az_arcmin'].median():.2f}'  "
        f"del={df['delta_el_arcmin'].median():.2f}'",
        fontsize=11,
    )

    out = os.path.join(plot_dir, "pointing_offsets_summary.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Summary plot saved: %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Fit HII sources in epoch maps and compute az/el pointing offsets."
    )
    p.add_argument(
        "--map-dir",
        required=True,
        help="Directory containing the hii_*_epoch*.fits maps.",
    )
    p.add_argument(
        "--summary",
        default="hii_field_match_summary.csv",
        help="Match summary CSV from match_hii_to_fields.py.",
    )
    p.add_argument(
        "--lst-metadata",
        default=None,
        help="LST metadata CSV (map_name, lst_center_hours) from match_hii_to_fields.py. "
             "If provided, uses mean LST for source az/el prediction instead of MEAN_MJD.",
    )
    p.add_argument(
        "--output",
        default="pointing_offsets.csv",
        help="Output CSV of pointing offsets (default: pointing_offsets.csv).",
    )
    p.add_argument(
        "--box-radius",
        type=float,
        default=0.5,
        help="Half-width of the fitting box around the source in degrees (default: 0.5).",
    )
    p.add_argument(
        "--init-fwhm",
        type=float,
        default=0.1,
        help="Initial FWHM guess in degrees (default: 0.1).",
    )
    p.add_argument(
        "--plot-dir",
        default="pointing_fit_plots",
        help="Directory for diagnostic plots (default: pointing_fit_plots/).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Load the HII → field match summary
    if not os.path.exists(args.summary):
        sys.exit(f"ERROR: Summary file not found: {args.summary}")
    summary = pd.read_csv(args.summary)
    log.info("Loaded %d source–field matches from %s", len(summary), args.summary)

    # Load LST metadata if provided (map_name → mean LST in hours)
    lst_lookup = {}
    if args.lst_metadata:
        if not os.path.exists(args.lst_metadata):
            sys.exit(f"ERROR: LST metadata file not found: {args.lst_metadata}")
        lst_meta = pd.read_csv(args.lst_metadata)
        for _, row in lst_meta.iterrows():
            lst_lookup[row["map_name"]] = float(row["lst_center_hours"])
        log.info("Loaded LST metadata for %d maps from %s",
                 len(lst_lookup), args.lst_metadata)

    # Build a lookup: field_name → list of (glon, glat) sources
    field_sources = {}
    for _, row in summary.iterrows():
        field = row["matched_field"]
        field_sources.setdefault(field, []).append(
            (float(row["glon"]), float(row["glat"]))
        )

    # Find all maps (LST group or legacy epoch naming)
    map_files = sorted(glob.glob(os.path.join(args.map_dir, "hii_*_lstgrp*_band*.fits")))
    if not map_files:
        map_files = sorted(glob.glob(os.path.join(args.map_dir, "hii_*_lstgrp*.fits")))
    if not map_files:
        # Fall back to legacy epoch naming
        map_files = sorted(glob.glob(os.path.join(args.map_dir, "hii_*_epoch*_band*.fits")))
    if not map_files:
        map_files = sorted(glob.glob(os.path.join(args.map_dir, "hii_*_epoch*.fits")))
    if not map_files:
        sys.exit(f"ERROR: No HII maps found in {args.map_dir}")
    log.info("Found %d map(s) in %s", len(map_files), args.map_dir)

    results = []

    for fitsfile in map_files:
        basename = os.path.basename(fitsfile)
        log.info("Processing %s", basename)

        # Parse field name and group from filename:
        #   hii_<Field>_lstgrp<NN>_band...fits   (LST-grouped)
        #   hii_<Field>_epoch<NN>_band...fits     (legacy epoch)
        parts = basename.replace(".fits", "").split("_")
        group_idx = None
        for i, tok in enumerate(parts):
            if tok.startswith("lstgrp") or tok.startswith("epoch"):
                group_idx = i
                break
        if group_idx is None:
            log.warning("Cannot parse group/epoch from filename %s — skipping.", basename)
            continue
        # Field name is everything between "hii" and the group token
        field_name = "_".join(parts[1:group_idx])
        epoch_str = parts[group_idx]

        if field_name not in field_sources:
            log.warning("Field '%s' not in summary — skipping %s.", field_name, basename)
            continue

        # Read FITS
        with fits.open(fitsfile) as hdul:
            header = hdul[0].header
            map_data = hdul[0].data
            wcs = WCS(header)

            # Read RMS if available
            rms_data = None
            for hdu in hdul:
                if hdu.name == "RMS":
                    rms_data = hdu.data
                    break

            mean_az = header.get("MEAN_AZ", None)
            mean_el = header.get("MEAN_EL", None)
            mean_mjd = header.get("MEAN_MJD", None)

        # Determine the mean LST for this map
        # Extract the map_name stem: everything before _band (or the full stem)
        stem = basename.replace(".fits", "")
        # Strip _band* and _feed* suffixes to get the map_name used in metadata
        map_name_key = stem
        for suffix_prefix in ("_band", "_feed"):
            idx = map_name_key.find(suffix_prefix)
            if idx >= 0:
                map_name_key = map_name_key[:idx]

        mean_lst = lst_lookup.get(map_name_key, None)

        if mean_lst is not None:
            log.info("  Using mean LST=%.4fh from metadata for az/el prediction", mean_lst)
        elif mean_mjd is not None:
            log.warning(
                "  No LST metadata for '%s' — falling back to MEAN_MJD=%.6f",
                map_name_key, mean_mjd,
            )
        else:
            log.warning(
                "Missing both LST metadata and MEAN_MJD in %s — skipping.", basename
            )
            continue

        # Fit each source expected in this field
        for source_glon, source_glat in field_sources[field_name]:
            log.info(
                "  Fitting source at (l=%.4f, b=%.4f) in %s",
                source_glon, source_glat, basename,
            )

            fit = fit_source_in_map(
                map_data,
                wcs,
                source_glon,
                source_glat,
                rms_data=rms_data,
                box_radius_deg=args.box_radius,
                init_fwhm_deg=args.init_fwhm,
            )
            if fit is None:
                log.warning("  Fit failed — skipping this source.")
                continue

            # Predicted source az/el from catalogue (l,b) → RA/Dec → az/el
            cat_ra, cat_dec = galactic_to_radec(source_glon, source_glat)
            if mean_lst is not None:
                pred_az, pred_el = radec_to_azel_from_lst(cat_ra, cat_dec, mean_lst)
            else:
                pred_az, pred_el = radec_to_azel(cat_ra, cat_dec, mean_mjd)

            if pred_el < 0:
                log.warning(
                    "  Predicted el=%.2f < 0 for (l=%.4f, b=%.4f) "
                    "— source below horizon, skipping. "
                    "(mean_lst=%.4fh, MEAN_AZ=%.2f, MEAN_EL=%.2f from header)",
                    pred_el, source_glon, source_glat,
                    mean_lst if mean_lst is not None else -1.0,
                    mean_az if mean_az is not None else -1.0,
                    mean_el if mean_el is not None else -1.0,
                )
                continue

            # Fitted source az/el from fitted (l,b) → RA/Dec → az/el
            fit_ra, fit_dec = galactic_to_radec(fit["fit_glon"], fit["fit_glat"])
            if mean_lst is not None:
                fit_az, fit_el = radec_to_azel_from_lst(fit_ra, fit_dec, mean_lst)
            else:
                fit_az, fit_el = radec_to_azel(fit_ra, fit_dec, mean_mjd)

            # Pointing offsets: fitted − predicted (degrees)
            # delta_az is on-sky: multiply by cos(el) for cross-elevation
            delta_az_deg = (fit_az - pred_az) * np.cos(np.radians(pred_el))
            delta_el_deg = fit_el - pred_el

            # Position uncertainties → offset uncertainties (approx same angular scale)
            # lon error projects onto az via cos(el), lat error maps to el directly
            err_az_deg = fit["pos_err_lon_deg"] * np.cos(np.radians(pred_el))
            err_el_deg = fit["pos_err_lat_deg"]

            # Convert to arcmin
            delta_az_arcmin = delta_az_deg * 60.0
            delta_el_arcmin = delta_el_deg * 60.0
            err_az_arcmin = err_az_deg * 60.0
            err_el_arcmin = err_el_deg * 60.0

            row = {
                "field": field_name,
                "group": epoch_str,
                "fits_file": basename,
                "source_glon": source_glon,
                "source_glat": source_glat,
                "fit_glon": fit["fit_glon"],
                "fit_glat": fit["fit_glat"],
                "cat_ra": cat_ra,
                "cat_dec": cat_dec,
                "fit_ra": fit_ra,
                "fit_dec": fit_dec,
                "amplitude": fit["amplitude"],
                "amplitude_err": fit["amplitude_err"],
                "fwhm_x_deg": fit["fwhm_x_deg"],
                "fwhm_y_deg": fit["fwhm_y_deg"],
                "theta_deg": fit["theta_deg"],
                "resid_rms": fit["resid_rms"],
                "mean_lst": mean_lst,
                "mean_az": mean_az,
                "mean_el": mean_el,
                "mean_mjd": mean_mjd,
                "pred_az": pred_az,
                "pred_el": pred_el,
                "fit_az": fit_az,
                "fit_el": fit_el,
                "delta_az_arcmin": delta_az_arcmin,
                "delta_el_arcmin": delta_el_arcmin,
                "err_az_arcmin": err_az_arcmin,
                "err_el_arcmin": err_el_arcmin,
            }
            results.append(row)
            log.info(
                "    fit (l=%.4f, b=%.4f)  amp=%.4g±%.2g K  "
                "daz*cos=%.2f±%.2f'  del=%.2f±%.2f'",
                fit["fit_glon"], fit["fit_glat"],
                fit["amplitude"], fit["amplitude_err"],
                delta_az_arcmin, err_az_arcmin,
                delta_el_arcmin, err_el_arcmin,
            )

            # Diagnostic plot
            os.makedirs(args.plot_dir, exist_ok=True)
            plot_name = f"{field_name}_{epoch_str}_l{source_glon:.2f}_b{source_glat:.2f}.png"
            plot_path = os.path.join(args.plot_dir, plot_name)
            plot_title = (
                f"{field_name} {epoch_str}  "
                f"l={fit['fit_glon']:.4f} b={fit['fit_glat']:.4f}  "
                f"amp={fit['amplitude']:.3g} K  "
                f"daz*cos={delta_az_arcmin:.2f}' del={delta_el_arcmin:.2f}'"
            )
            plot_fit(fit, plot_path, title=plot_title)

    # Write output
    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        log.info("Wrote %d offset(s) to %s", len(df), args.output)

        # Print summary (arcmin)
        print("\n" + "=" * 90)
        print("POINTING OFFSET SUMMARY (arcmin)")
        print("=" * 90)
        cols = [
            "field", "group", "source_glon", "source_glat",
            "amplitude",
            "mean_lst", "pred_az", "pred_el",
            "delta_az_arcmin", "err_az_arcmin",
            "delta_el_arcmin", "err_el_arcmin",
        ]
        print(df[cols].to_string(index=False, float_format="%.4f"))
        print("=" * 90)
        print(f"\nMedian offsets:  delta_az*cos(el) = {df['delta_az_arcmin'].median():.2f}'"
              f"   delta_el = {df['delta_el_arcmin'].median():.2f}'")
        print(f"Scatter (std):  delta_az*cos(el) = {df['delta_az_arcmin'].std():.2f}'"
              f"   delta_el = {df['delta_el_arcmin'].std():.2f}'")

        # Summary plot: offsets vs az/el
        plot_offsets_vs_azel(df, args.plot_dir)
    else:
        log.warning("No successful fits — no output written.")


if __name__ == "__main__":
    main()
