"""Matplotlib plotting helpers for the diagnostics notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


N_FEEDS = 19
N_BANDS = 8


def plot_timeseries(
    df: pd.DataFrame,
    stat: str,
    group_by: str = "feed",
    feed: int | None = None,
    band: int | None = None,
    source_group: str | None = None,
    ax=None,
    log_y: bool = False,
):
    """Stat vs MJD scatter, coloured by feed or band.

    Filters by `feed` / `band` / `source_group` if provided. `group_by` picks
    the colour axis ("feed" or "band").
    """
    sub = df.dropna(subset=["mjd", stat]).copy()
    if source_group is not None:
        sub = sub[sub["source_group"] == source_group]
    if feed is not None:
        sub = sub[sub["feed"] == feed]
    if band is not None:
        sub = sub[sub["band"] == band]

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4))

    if sub.empty:
        ax.set_title(f"{stat} — no data")
        return ax

    groups = sorted(sub[group_by].unique())
    cmap = plt.get_cmap("viridis", max(len(groups), 1))
    for i, g in enumerate(groups):
        seg = sub[sub[group_by] == g]
        ax.scatter(seg["mjd"], seg[stat], s=6, alpha=0.45,
                   color=cmap(i), label=f"{group_by}={g}")
    ax.set_xlabel("MJD")
    ax.set_ylabel(stat)
    if log_y:
        ax.set_yscale("log")
    title = stat
    if source_group:
        title += f" — {source_group}"
    if feed is not None:
        title += f" — feed {feed}"
    if band is not None:
        title += f" — band {band}"
    ax.set_title(title)
    if len(groups) <= 12:
        ax.legend(fontsize=7, ncol=2, loc="best")
    return ax


def plot_feed_band_heatmap(
    df: pd.DataFrame,
    stat: str,
    agg: str = "median",
    source_group: str | None = None,
    ax=None,
    cmap: str = "viridis",
    log_color: bool = False,
):
    """Aggregate `stat` per (feed, band) and plot a 19 x 8 grid."""
    sub = df.dropna(subset=[stat])
    if source_group is not None:
        sub = sub[sub["source_group"] == source_group]
    grid = np.full((N_FEEDS, N_BANDS), np.nan)
    counts = np.zeros((N_FEEDS, N_BANDS), dtype=int)
    if not sub.empty:
        agg_func = {"median": np.median, "mean": np.mean, "std": np.std}[agg]
        for (f, b), seg in sub.groupby(["feed", "band"]):
            if 0 <= f < N_FEEDS and 0 <= b < N_BANDS:
                grid[int(f), int(b)] = agg_func(seg[stat])
                counts[int(f), int(b)] = len(seg)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))

    plot_data = grid
    if log_color:
        with np.errstate(divide="ignore", invalid="ignore"):
            plot_data = np.where(grid > 0, np.log10(grid), np.nan)

    im = ax.imshow(plot_data, aspect="auto", origin="lower", cmap=cmap)
    ax.set_xlabel("band")
    ax.set_ylabel("feed (0-indexed)")
    title = f"{agg}({stat})"
    if source_group:
        title += f" — {source_group}"
    if log_color:
        title += " (log10)"
    ax.set_title(title)
    ax.set_xticks(range(N_BANDS))
    ax.set_yticks(range(N_FEEDS))
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(stat)

    for f in range(N_FEEDS):
        for b in range(N_BANDS):
            if counts[f, b] > 0:
                ax.text(b, f, str(counts[f, b]), ha="center", va="center",
                        fontsize=6, color="white")
    return ax


def plot_source_group_distributions(
    df: pd.DataFrame,
    stat: str,
    ax=None,
    log_y: bool = False,
):
    """Violin/box of `stat` grouped by source_group."""
    sub = df.dropna(subset=[stat, "source_group"])
    groups = sorted(sub["source_group"].dropna().unique())
    data = [sub[sub["source_group"] == g][stat].values for g in groups]

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))

    if not data or all(len(d) == 0 for d in data):
        ax.set_title(f"{stat} — no data")
        return ax

    ax.boxplot(data, tick_labels=groups, showfliers=False)
    if log_y:
        ax.set_yscale("log")
    ax.set_ylabel(stat)
    ax.set_title(f"{stat} by source group")
    ax.tick_params(axis="x", rotation=30)
    return ax


def plot_per_scan(
    per_scan_df: pd.DataFrame,
    stat: str,
    band: int = 0,
    channel: int = 0,
    ax=None,
):
    """Plot per-scan stat for one band/channel, one line per feed.

    Expects a DataFrame from `load_per_scan_noise_from_hdf5`.
    """
    sub = per_scan_df[
        (per_scan_df["band"] == band) & (per_scan_df["channel"] == channel)
    ]
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4))
    if sub.empty or stat not in sub.columns:
        ax.set_title(f"{stat} — no data")
        return ax
    cmap = plt.get_cmap("viridis", N_FEEDS)
    for feed, seg in sub.groupby("feed"):
        ax.plot(seg["scan"], seg[stat], color=cmap(int(feed)),
                alpha=0.7, label=f"feed {feed}")
    ax.set_xlabel("scan index")
    ax.set_ylabel(stat)
    ax.set_title(f"{stat} per scan — band {band} chan {channel}")
    ax.legend(fontsize=6, ncol=4, loc="best")
    return ax


def plot_calibrator_timeseries(
    cal_df: pd.DataFrame,
    calibrator: str,
    feed: int,
    band: int,
    channel: int = 0,
    model_df: pd.DataFrame | None = None,
    ax=None,
):
    """Calibrator measured flux vs MJD for one feed/band/channel.

    Overlays the active calibration_model_fit curve if `model_df` is provided.
    """
    sub = cal_df[
        (cal_df.get("source") == calibrator)
        & (cal_df["feed"] == feed)
        & (cal_df["band"] == band)
        & (cal_df["channel"] == channel)
    ].dropna(subset=["measured_flux", "mjd"])

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    if sub.empty:
        ax.set_title(f"{calibrator} feed {feed} band {band} chan {channel} — no data")
        return ax

    ax.errorbar(
        sub["mjd"], sub["measured_flux"], yerr=sub.get("flux_error"),
        fmt="o", ms=3, alpha=0.6, label="measured"
    )

    if model_df is not None and not model_df.empty:
        row = model_df[
            (model_df["feed"] == feed)
            & (model_df["band"] == band)
            & (model_df["channel"] == channel)
        ]
        if len(row) == 1:
            r = row.iloc[0]
            mjd_grid = np.linspace(sub["mjd"].min(), sub["mjd"].max(), 200)
            curve = _evaluate_model(r, mjd_grid)
            if curve is not None:
                ax.plot(mjd_grid, curve, "-", color="C3",
                        label=f"model ({r.get('model_type')})")

    ax.set_xlabel("MJD")
    ax.set_ylabel("flux (Jy)")
    ax.set_title(f"{calibrator} feed {feed} band {band} chan {channel}")
    ax.legend(fontsize=8)
    return ax


def _evaluate_model(row: pd.Series, mjd: np.ndarray):
    """Evaluate a calibration_model_fit row at a grid of MJDs."""
    mtype = row.get("model_type")
    if mtype in ("polynomial", "mean"):
        coeffs = row.get("poly_coeffs")
        if not coeffs:
            return None
        t = (mjd - 59000.0) / 365.25
        return np.polyval(list(reversed(coeffs)), t)
    if mtype == "nearest":
        mjds = row.get("nearest_mjds")
        gains = row.get("nearest_gains")
        if not mjds or not gains:
            return None
        mjds = np.asarray(mjds)
        gains = np.asarray(gains)
        idx = np.array([np.argmin(np.abs(mjds - m)) for m in mjd])
        return gains[idx]
    return None
