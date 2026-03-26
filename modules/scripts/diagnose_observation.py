"""Diagnostic script for inspecting a single COMAP Level-2 observation.

Produces:
  1. Per-feed TOD plots (one panel per scan)
  2. Per-feed 1/f noise power spectra with fitted model overlay
  3. Per-feed system temperature, gain, and atmosphere fit plots
  4. CSV summary of noise statistics

Usage:
  # By filename:
  python modules/scripts/diagnose_observation.py --file /path/to/level2.h5

  # By obsid (requires database):
  python modules/scripts/diagnose_observation.py --obsid 57205 --database databases/COMAP_manchester.db

  # Custom output directory:
  python modules/scripts/diagnose_observation.py --file /path/to/level2.h5 --output-dir outputs/diagnostics
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py

GROUND_FEED = 20
NBANDS = 4


# ──────────────────────────────────────────────────────────────────────
# Noise analysis helpers (mirroring NoiseStats module)
# ──────────────────────────────────────────────────────────────────────

def auto_rms(data: np.ndarray) -> float:
    N = len(data) // 2 * 2
    return float(np.nanstd(data[:N:2] - data[1:N:2]) / np.sqrt(2))


def power_spectrum(data: np.ndarray, sample_frequency: float = 50.0):
    N = len(data)
    d = data - np.nanmedian(data)
    nans = np.isnan(d)
    if np.any(nans):
        d[nans] = 0.0
    Pk = np.abs(np.fft.fft(d)[:N // 2]) ** 2 / N
    k = np.fft.fftfreq(N, d=1.0 / sample_frequency)[:N // 2]

    N_bins = 100
    edges = np.logspace(np.log10(k[1]), np.log10(k[-1]), N_bins + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    weights = np.histogram(k, bins=edges)[0]
    Pk_binned = np.histogram(k, bins=edges, weights=Pk)[0]
    good = weights > 0
    Pk_binned[good] /= weights[good]
    mask = good & np.isfinite(Pk_binned)
    return centres[mask], Pk_binned[mask], weights[mask]


def noise_model(k, sigma_white, sigma_red, alpha, k_ref=0.1):
    return sigma_red**2 * np.abs(k / k_ref) ** alpha + sigma_white**2


def fit_noise(data: np.ndarray, k_ref: float = 0.1):
    k, Pk, w = power_spectrum(data)
    if len(Pk) == 0:
        return (np.nan, np.nan, np.nan), np.array([]), np.array([])
    rms = auto_rms(data)
    ref_amp = np.mean(Pk[np.argmin(np.abs(k - k_ref))]) ** 0.5

    def chi2(params):
        mdl = noise_model(k, *params, k_ref=k_ref)
        if np.any(mdl <= 0):
            return 1e30
        return np.sum((np.log(Pk) - np.log(mdl)) ** 2 * w**0.5)

    res = minimize(chi2, [rms, ref_amp, -1.5], bounds=[(0, None), (0, None), (-4, 4)])
    params = tuple(res.x) if res.success else (np.nan, np.nan, np.nan)
    return params, k, Pk


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────

def plot_tod_per_scan(h5, feeds, scan_edges, output_dir, obsid):
    """One figure per feed: panels showing the binned filtered TOD per scan."""
    n_scans = len(scan_edges)
    dset_name = "level2/binned_filtered_data"
    if dset_name not in h5:
        print(f"  [skip] {dset_name} not found — cannot plot TOD")
        return

    for feed in feeds:
        fig, axes = plt.subplots(n_scans, 1, figsize=(14, 3 * n_scans),
                                 sharex=False, squeeze=False)
        fig.suptitle(f"Obsid {obsid} — Feed {feed:02d} — Filtered TOD per scan", y=1.0)

        # Read full feed data: shape (NBANDS, n_channels, n_tod)
        data = h5[dset_name][feed - 1, ...]

        for iscan, (s0, s1) in enumerate(scan_edges):
            ax = axes[iscan, 0]
            # Average over frequency channels for a summary trace per band
            for iband in range(data.shape[0]):
                trace = np.nanmean(data[iband, :, s0:s1], axis=0)
                ax.plot(np.arange(s0, s1), trace, lw=0.4, label=f"Band {iband}")
            ax.set_ylabel("T [K]")
            ax.set_title(f"Scan {iscan} (samples {s0}–{s1})", fontsize=9)
            if iscan == 0:
                ax.legend(fontsize=7, ncol=4, loc="upper right")

        axes[-1, 0].set_xlabel("Sample index")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{obsid}_feed{feed:02d}_tod.png"), dpi=120)
        plt.close(fig)


def plot_noise_spectra(h5, feeds, scan_edges, output_dir, obsid):
    """One figure per feed: power spectrum + fit for each scan, band-averaged."""
    dset_name = "level2/binned_filtered_data"
    if dset_name not in h5:
        print(f"  [skip] {dset_name} not found — cannot plot noise spectra")
        return

    n_scans = len(scan_edges)
    for feed in feeds:
        data_full = h5[dset_name][feed - 1, ...]  # (NBANDS, nch, ntod)
        n_bands = data_full.shape[0]

        fig, axes = plt.subplots(n_scans, n_bands, figsize=(5 * n_bands, 3 * n_scans),
                                 squeeze=False, sharex=True)
        fig.suptitle(f"Obsid {obsid} — Feed {feed:02d} — Noise power spectra", y=1.0)

        for iscan, (s0, s1) in enumerate(scan_edges):
            for iband in range(n_bands):
                ax = axes[iscan, iband]
                # Average channels in this band for one summary spectrum
                trace = np.nanmean(data_full[iband, :, s0:s1], axis=0)
                if np.all(np.isnan(trace)):
                    ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
                    continue
                params, k, Pk = fit_noise(trace)
                if len(k) > 0:
                    ax.loglog(k, Pk, color="C0", lw=0.6, alpha=0.7)
                    if np.isfinite(params[0]):
                        ax.loglog(k, noise_model(k, *params), color="C3", lw=1.5,
                                  label=(f"$\\sigma_w$={params[0]:.3g}\n"
                                         f"$\\sigma_r$={params[1]:.3g}\n"
                                         f"$\\alpha$={params[2]:.2f}"))
                        ax.legend(fontsize=6, loc="upper right")
                if iscan == 0:
                    ax.set_title(f"Band {iband}", fontsize=9)
                if iband == 0:
                    ax.set_ylabel(f"Scan {iscan}", fontsize=9)

        for ax in axes[-1, :]:
            ax.set_xlabel("Frequency [Hz]")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{obsid}_feed{feed:02d}_noise_psd.png"), dpi=120)
        plt.close(fig)


def plot_tsys_gain_atmos(h5, feeds, scan_edges, output_dir, obsid):
    """One figure per feed with three rows: Tsys, Gain, Atmosphere (tau & offset)."""
    has_tsys = "level2/vane/system_temperature" in h5
    has_gain = "level2/vane/gain" in h5
    has_atmos = "level2/atmosphere/tau" in h5

    if not (has_tsys or has_gain or has_atmos):
        print("  [skip] No Tsys/Gain/Atmosphere datasets found")
        return

    freq = None
    if "spectrometer/frequency" in h5:
        freq = h5["spectrometer/frequency"][:].reshape(-1)  # (NBANDS*NCHANNELS,)
        sort_idx = np.argsort(freq)

    for feed in feeds:
        ifeed = feed - 1
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
        fig.suptitle(f"Obsid {obsid} — Feed {feed:02d} — Calibration diagnostics", y=0.98)

        x_label = "Channel index"
        x_vals = None
        if freq is not None:
            x_vals = freq[sort_idx]
            x_label = "Frequency [GHz]"

        # --- Row 0: System temperature ---
        if has_tsys:
            tsys = h5["level2/vane/system_temperature"][0, ifeed, :, :]  # (NBANDS, NCHANNELS)
            ax = fig.add_subplot(gs[0, 0])
            vals = tsys.reshape(-1)
            xv = x_vals if x_vals is not None else np.arange(len(vals))
            idx = sort_idx if freq is not None else np.arange(len(vals))
            for iband in range(tsys.shape[0]):
                band_slice = slice(iband * tsys.shape[1], (iband + 1) * tsys.shape[1])
                ax.plot(xv[band_slice] if freq is not None else np.arange(tsys.shape[1]),
                        tsys[iband, :], lw=0.6, label=f"Band {iband}")
            ax.set_ylabel("Tsys [K]")
            ax.set_xlabel(x_label)
            ax.set_title("System Temperature")
            ax.legend(fontsize=7, ncol=4)

        # --- Row 0 right: Gain ---
        if has_gain:
            gain = h5["level2/vane/gain"][0, ifeed, :, :]  # (NBANDS, NCHANNELS)
            ax = fig.add_subplot(gs[0, 1])
            for iband in range(gain.shape[0]):
                ax.plot(np.arange(gain.shape[1]), gain[iband, :], lw=0.6, label=f"Band {iband}")
            ax.set_ylabel("Gain [counts/K]")
            ax.set_xlabel("Channel index")
            ax.set_title("Vane Gain")
            ax.legend(fontsize=7, ncol=4)

        # --- Row 1: Atmosphere tau per scan ---
        if has_atmos:
            tau = h5["level2/atmosphere/tau"][ifeed, :, :, :]    # (NBANDS, NCHANNELS, n_scans)
            offsets = h5["level2/atmosphere/offsets"][ifeed, :, :, :]

            n_scans = tau.shape[-1]
            ax_tau = fig.add_subplot(gs[1, 0])
            ax_off = fig.add_subplot(gs[1, 1])

            for iscan in range(n_scans):
                for iband in range(tau.shape[0]):
                    ax_tau.plot(np.arange(tau.shape[1]),
                                tau[iband, :, iscan], lw=0.5, alpha=0.7,
                                label=f"B{iband} S{iscan}" if iband == 0 else None)
                    ax_off.plot(np.arange(offsets.shape[1]),
                                offsets[iband, :, iscan], lw=0.5, alpha=0.7)

            ax_tau.set_ylabel("Tau coefficient [K]")
            ax_tau.set_xlabel("Channel index")
            ax_tau.set_title("Atmosphere: tau")
            ax_tau.legend(fontsize=6, ncol=min(n_scans, 4))

            ax_off.set_ylabel("Offset [K]")
            ax_off.set_xlabel("Channel index")
            ax_off.set_title("Atmosphere: offsets")

        # --- Row 2: Stored noise stats summary (if available) ---
        stats_group = None
        for gname in ["level2_noise_stats/binned_filtered_data",
                       "level2_noise_stats/binned_data"]:
            if gname in h5:
                stats_group = gname
                break

        if stats_group is not None:
            sigma_r = h5[f"{stats_group}/sigma_red"][ifeed, ...]    # (NBANDS, nch, n_scans)
            alpha_n = h5[f"{stats_group}/alpha"][ifeed, ...]
            arms = h5[f"{stats_group}/auto_rms"][ifeed, ...]

            ax_sr = fig.add_subplot(gs[2, 0])
            ax_al = fig.add_subplot(gs[2, 1])

            for iband in range(sigma_r.shape[0]):
                sr_mean = np.nanmean(sigma_r[iband, :, :], axis=-1)
                al_mean = np.nanmean(alpha_n[iband, :, :], axis=-1)
                ax_sr.plot(sr_mean, lw=0.8, label=f"Band {iband}")
                ax_al.plot(al_mean, lw=0.8, label=f"Band {iband}")

            ax_sr.set_ylabel("sigma_red (scan-avg)")
            ax_sr.set_xlabel("Channel bin")
            ax_sr.set_title("Red noise amplitude")
            ax_sr.legend(fontsize=7, ncol=4)

            ax_al.set_ylabel("alpha (scan-avg)")
            ax_al.set_xlabel("Channel bin")
            ax_al.set_title("Noise spectral index")
            ax_al.legend(fontsize=7, ncol=4)

        fig.savefig(os.path.join(output_dir, f"{obsid}_feed{feed:02d}_calibration.png"), dpi=120)
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# CSV summary
# ──────────────────────────────────────────────────────────────────────

def write_noise_csv(h5, feeds, scan_edges, output_dir, obsid):
    """Write a CSV with per-feed, per-band, per-scan noise statistics."""
    rows = []

    # Collect stored stats if present
    stored_stats = {}
    for dset_label, gname in [("filtered", "level2_noise_stats/binned_filtered_data"),
                               ("unfiltered", "level2_noise_stats/binned_data")]:
        if gname in h5:
            stored_stats[dset_label] = {
                k: h5[f"{gname}/{k}"][...] for k in h5[gname].keys()
            }

    # Tsys summary
    has_tsys = "level2/vane/system_temperature" in h5
    tsys_data = h5["level2/vane/system_temperature"][0, ...] if has_tsys else None  # (NFEEDS, NBANDS, NCHANNELS)

    # Atmosphere summary
    has_atmos = "level2/atmosphere/tau" in h5
    tau_data = h5["level2/atmosphere/tau"][...] if has_atmos else None  # (NFEEDS, NBANDS, NCHANNELS, n_scans)

    for feed in feeds:
        ifeed = feed - 1
        for iband in range(NBANDS):
            for iscan, (s0, s1) in enumerate(scan_edges):
                row = {
                    "obsid": obsid,
                    "feed": feed,
                    "band": iband,
                    "scan": iscan,
                    "scan_start": int(s0),
                    "scan_end": int(s1),
                    "scan_length": int(s1 - s0),
                }

                # Median Tsys across channels
                if tsys_data is not None:
                    row["median_tsys_K"] = f"{np.nanmedian(tsys_data[ifeed, iband, :]):.3f}"
                else:
                    row["median_tsys_K"] = ""

                # Mean tau across channels for this scan
                if tau_data is not None and iscan < tau_data.shape[-1]:
                    row["mean_tau"] = f"{np.nanmean(tau_data[ifeed, iband, :, iscan]):.4f}"
                else:
                    row["mean_tau"] = ""

                # Stored noise stats (averaged over channels)
                for label, stats in stored_stats.items():
                    for key in ["sigma_white", "sigma_red", "alpha", "auto_rms"]:
                        val = stats[key][ifeed, iband, :, iscan]
                        nonzero = val[val != 0]
                        mean_val = np.nanmean(nonzero) if len(nonzero) > 0 else np.nan
                        row[f"{label}_{key}"] = f"{mean_val:.6g}" if np.isfinite(mean_val) else ""

                # Live auto_rms from TOD if binned_filtered_data exists
                dset_name = "level2/binned_filtered_data"
                if dset_name in h5:
                    trace = np.nanmean(h5[dset_name][ifeed, iband, :, s0:s1], axis=0)
                    row["live_auto_rms"] = f"{auto_rms(trace):.6g}" if not np.all(np.isnan(trace)) else ""
                else:
                    row["live_auto_rms"] = ""

                rows.append(row)

    if not rows:
        print("  [skip] No data for CSV")
        return

    csv_path = os.path.join(output_dir, f"{obsid}_noise_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV written: {csv_path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def resolve_file(args):
    """Return (level2_path, obsid) from CLI args."""
    if args.file:
        level2_path = args.file
        with h5py.File(level2_path, "r") as h5:
            obsid = int(h5["comap"].attrs.get("obsid", 0))
        return level2_path, obsid

    if args.obsid:
        if not args.database:
            print("ERROR: --database is required when using --obsid", file=sys.stderr)
            sys.exit(1)
        from modules.SQLModule.SQLModule import db
        db.connect(args.database)
        db._connect()
        from modules.SQLModule.SQLModule import COMAPData
        row = db.session.query(COMAPData).filter_by(obsid=args.obsid).first()
        db._disconnect()
        if row is None:
            print(f"ERROR: obsid {args.obsid} not found in database", file=sys.stderr)
            sys.exit(1)
        if not row.level2_path:
            print(f"ERROR: obsid {args.obsid} has no level2_path", file=sys.stderr)
            sys.exit(1)
        return row.level2_path, args.obsid

    print("ERROR: supply --file or --obsid", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic plots and noise summary for a single COMAP Level-2 observation"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Path to Level-2 HDF5 file")
    group.add_argument("--obsid", type=int, help="Observation ID (requires --database)")
    parser.add_argument("--database", type=str, help="Path to SQLite database (required with --obsid)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: outputs/diagnostics/<obsid>)")
    parser.add_argument("--feeds", type=int, nargs="+", default=None,
                        help="Specific feeds to plot (default: all science feeds)")
    args = parser.parse_args()

    level2_path, obsid = resolve_file(args)
    print(f"Diagnosing obsid {obsid}: {level2_path}")

    output_dir = args.output_dir or f"outputs/diagnostics/{obsid}"
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(level2_path, "r") as h5:
        all_feeds = h5["spectrometer/feeds"][:]
        science_feeds = sorted([int(f) for f in all_feeds if f != GROUND_FEED])
        feeds = args.feeds if args.feeds else science_feeds

        if "level2/scan_edges" not in h5:
            print("ERROR: level2/scan_edges not found — file may not be fully processed", file=sys.stderr)
            sys.exit(1)
        scan_edges = h5["level2/scan_edges"][...]

        print(f"  Feeds: {feeds}")
        print(f"  Scans: {len(scan_edges)}")
        print(f"  Output: {output_dir}")

        print("  Plotting TOD per scan...")
        plot_tod_per_scan(h5, feeds, scan_edges, output_dir, obsid)

        print("  Plotting noise power spectra...")
        plot_noise_spectra(h5, feeds, scan_edges, output_dir, obsid)

        print("  Plotting Tsys / Gain / Atmosphere...")
        plot_tsys_gain_atmos(h5, feeds, scan_edges, output_dir, obsid)

        print("  Writing noise summary CSV...")
        write_noise_csv(h5, feeds, scan_edges, output_dir, obsid)

    print(f"Done. All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
