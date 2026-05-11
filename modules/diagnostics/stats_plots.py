"""Matplotlib plotting helpers for the diagnostics notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


N_FEEDS = 19
N_BANDS = 8


# Default 19-feed hex layout (rows of 3-4-5-4-3). Coordinates are in
# "hex units": x is horizontal, y is vertical, even rows offset by 0.5.
# Feed indices are 0-based. Override by passing your own dict to
# `plot_feed_hex_boxplots(..., hex_layout=YOUR_DICT)`.
DEFAULT_HEX_LAYOUT: dict[int, tuple[float, float]] = {
    0: (1.0,  2.0), 1: (2.0,  2.0), 2: (3.0,  2.0),
    3: (0.5,  1.0), 4: (1.5,  1.0), 5: (2.5,  1.0), 6: (3.5,  1.0),
    7: (0.0,  0.0), 8: (1.0,  0.0), 9: (2.0,  0.0), 10: (3.0, 0.0), 11: (4.0, 0.0),
    12: (0.5, -1.0), 13: (1.5, -1.0), 14: (2.5, -1.0), 15: (3.5, -1.0),
    16: (1.0, -2.0), 17: (2.0, -2.0), 18: (3.0, -2.0),
}


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


def mapmaking_pass_obsids(df: pd.DataFrame) -> pd.Index:
    """Obsids that would survive MapMaking's obsid-level flag check.

    Reproduces the SQL-checkable parts of the gate in
    `modules/MapMaking/MapMaking.py`: an obsid passes iff it has a
    non-null `level2_path` AND at least one (feed, band) row with
    `is_good=True`. Returns a pandas Index of unique passing obsids.

    Caveats — this is an *upper bound*:
      - HDF5 dataset presence (`level2/binned_filtered_data`, etc.)
        and per-(feed, band) gating happen later in ReadData.
      - SkyDip should be excluded by the caller before calling this.
    """
    if df.empty:
        return pd.Index([], name="obsid")
    has_path = df["level2_path"].notna()
    sub = df[has_path]
    any_good = sub.groupby("obsid")["is_good"].any()
    return any_good[any_good].index


def plot_obsids_per_source_group(
    df: pd.DataFrame,
    exclude_groups: tuple[str, ...] = ("SkyDip",),
    ax=None,
    annotate: bool = True,
):
    """Bar chart: number of unique obsids per source_group.

    `df` is expected to have `obsid` and `source_group` columns (e.g. from
    `load_quality_flags`). SkyDip is excluded by default.
    """
    sub = df.dropna(subset=["source_group"])
    if exclude_groups:
        sub = sub[~sub["source_group"].isin(exclude_groups)]
    counts = (sub.drop_duplicates("obsid")
                .groupby("source_group").size()
                .sort_values(ascending=False))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    if counts.empty:
        ax.set_title("No observations after filtering")
        return ax

    bars = ax.bar(counts.index.astype(str), counts.values, color="C0")
    ax.set_ylabel("# observations")
    excl = ", ".join(exclude_groups) if exclude_groups else "none"
    ax.set_title(f"Observations per source group (excluding: {excl})")
    ax.tick_params(axis="x", rotation=20)
    if annotate:
        for b, v in zip(bars, counts.values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{int(v):,}",
                    ha="center", va="bottom", fontsize=8)
    return ax


def plot_obsids_per_source(
    df: pd.DataFrame,
    source_group: str,
    ax=None,
    top_n: int | None = None,
    log_y: bool = False,
    annotate: bool = True,
    exclude_groups: tuple[str, ...] = ("SkyDip",),
    show_mapmaking_pass: bool = False,
):
    """Bar chart: number of unique obsids per `source` within one source_group.

    SkyDip is always excluded via `exclude_groups`, regardless of the
    requested group, as a safety net.

    If `show_mapmaking_pass=True`, overlays a second (narrower) bar per
    source showing how many obsids would survive MapMaking's obsid-level
    flag check (see `mapmaking_pass_obsids`). The annotation switches to
    "pass/total".
    """
    sub = df.dropna(subset=["source", "source_group"])
    if exclude_groups:
        sub = sub[~sub["source_group"].isin(exclude_groups)]
    sub = sub[sub["source_group"] == source_group]

    totals = (sub.drop_duplicates("obsid")
                .groupby("source").size()
                .sort_values(ascending=False))
    if top_n is not None:
        totals = totals.head(top_n)

    if show_mapmaking_pass:
        passing = mapmaking_pass_obsids(sub)
        sub_pass = sub[sub["obsid"].isin(passing)]
        pass_counts = (sub_pass.drop_duplicates("obsid")
                       .groupby("source").size())
        pass_counts = pass_counts.reindex(totals.index).fillna(0).astype(int)

    if ax is None:
        width = max(6.0, 0.35 * max(len(totals), 1) + 2.0)
        fig, ax = plt.subplots(figsize=(width, 4))
    if totals.empty:
        ax.set_title(f"{source_group} — no observations")
        return ax

    x = np.arange(len(totals))
    labels = totals.index.astype(str)
    if show_mapmaking_pass:
        ax.bar(x, totals.values, width=0.8, color="C2",
               alpha=0.4, label="total")
        bars_pass = ax.bar(x, pass_counts.values, width=0.55, color="C2",
                           label="passes MapMaking gate")
        ax.legend(fontsize=8, loc="upper right")
    else:
        bars_pass = ax.bar(x, totals.values, width=0.8, color="C2")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, fontsize=7)
    ax.set_ylabel("# observations")
    title = f"Observations per source — {source_group}"
    if top_n is not None and len(totals) == top_n:
        title += f" (top {top_n})"
    if show_mapmaking_pass:
        title += "  [MapMaking-gate SQL upper bound]"
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    if annotate:
        for i, (b, tot) in enumerate(zip(bars_pass, totals.values)):
            if show_mapmaking_pass:
                txt = f"{int(pass_counts.values[i])}/{int(tot)}"
            else:
                txt = f"{int(tot)}"
            ax.text(b.get_x() + b.get_width() / 2, tot, txt,
                    ha="center", va="bottom", fontsize=7)
    return ax


def plot_feed_hex_boxplots(
    df: pd.DataFrame,
    stat: str,
    hex_layout: dict[int, tuple[float, float]] | None = None,
    band: int | None = None,
    source_group: str | None = None,
    log_y: bool = False,
    share_y: bool = True,
    showfliers: bool = False,
    fig=None,
    figsize: tuple[float, float] = (11.0, 9.0),
    pct_clip: tuple[float, float] = (1.0, 99.0),
):
    """Box plot of `stat` per feed, with one mini-axes per feed placed
    in a hexagon layout.

    Parameters
    ----------
    df : DataFrame
        Per-(obsid, feed, band) rows, e.g. quality_flags loaded via
        `stats_queries.load_quality_flags`.
    stat : str
        Column name to summarise.
    hex_layout : dict[int, (x, y)] or None
        Mapping feed -> hex coordinate. If None, uses DEFAULT_HEX_LAYOUT.
        Customise this to match the physical pixel arrangement.
    band, source_group : optional filters.
    log_y : if True, y axis log-scaled and non-positive values dropped.
    share_y : if True, all sub-axes share a y range based on the
        `pct_clip` percentiles across all feeds.
    showfliers : passed to matplotlib's boxplot.
    fig, figsize : control the parent figure.
    """
    layout = hex_layout if hex_layout is not None else DEFAULT_HEX_LAYOUT

    sub = df.dropna(subset=[stat]).copy()
    if source_group is not None:
        sub = sub[sub["source_group"] == source_group]
    if band is not None:
        sub = sub[sub["band"] == band]
    if log_y:
        sub = sub[sub[stat] > 0]

    if fig is None:
        fig = plt.figure(figsize=figsize)

    xs = [p[0] for p in layout.values()]
    ys = [p[1] for p in layout.values()]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    nx = xmax - xmin + 1.0
    ny = ymax - ymin + 1.0

    # Reserve outer margins for the suptitle and labels.
    L, B, W, H = 0.06, 0.05, 0.90, 0.86
    cell_w = W / nx
    cell_h = H / ny
    ax_w = cell_w * 0.78
    ax_h = cell_h * 0.72

    ylim = None
    if share_y and not sub.empty:
        vals = sub[stat].values
        if len(vals) > 0:
            lo, hi = np.nanpercentile(vals, pct_clip)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                ylim = (lo, hi)

    axes: dict[int, plt.Axes] = {}
    for feed, (x, y) in layout.items():
        left = L + (x - xmin) * cell_w + (cell_w - ax_w) * 0.5
        bottom = B + (y - ymin) * cell_h + (cell_h - ax_h) * 0.5
        ax = fig.add_axes([left, bottom, ax_w, ax_h])

        seg = sub[sub["feed"] == feed][stat].values
        seg = seg[np.isfinite(seg)]
        if len(seg) > 0:
            ax.boxplot(seg, showfliers=showfliers, widths=0.7)
            n = len(seg)
        else:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7, color="grey")
            n = 0

        if log_y and n > 0:
            ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title(f"F{feed} (n={n})", fontsize=8, pad=2)
        axes[feed] = ax

    title = f"{stat} per feed (hex layout)"
    if source_group:
        title += f" — {source_group}"
    if band is not None:
        title += f" — band {band}"
    fig.suptitle(title, fontsize=11)
    return fig, axes


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
