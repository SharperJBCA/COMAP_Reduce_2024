"""Pandas-based loaders for the COMAP diagnostics notebook.

All queries are read-only and operate directly on the SQLite file. Optional
tables (observation_summary, calibration_flux, calibration_model_fit) may not
exist in older databases; these helpers return empty DataFrames in that case.
"""

from __future__ import annotations

import os
import sqlite3
import json
import numpy as np
import pandas as pd


NOISE_COLUMNS = [
    "filtered_red_noise",
    "filtered_white_noise",
    "filtered_auto_rms",
    "filtered_noise_index",
    "unfiltered_red_noise",
    "unfiltered_white_noise",
    "unfiltered_auto_rms",
    "unfiltered_noise_index",
    "n_spikes",
    "n_nan_values",
    "mean_atm_temp",
]


def _connect(database_path: str) -> sqlite3.Connection:
    if not os.path.exists(database_path):
        raise FileNotFoundError(database_path)
    return sqlite3.connect(database_path)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    )
    return cur.fetchone() is not None


def _utc_to_mjd(utc_series: pd.Series) -> pd.Series:
    """Parse comap_data.utc_start ("YYYY-MM-DD-HH:MM:SS") to MJD."""
    dt = pd.to_datetime(utc_series, format="%Y-%m-%d-%H:%M:%S", errors="coerce")
    epoch = pd.Timestamp("1858-11-17")  # MJD 0
    mjd = (dt - epoch).dt.total_seconds() / 86_400.0
    return mjd.where(dt.notna())


def load_quality_flags(
    database_path: str,
    source_groups: list[str] | None = None,
    mjd_range: tuple[float, float] | None = None,
    min_obsid: int = 0,
    max_obsid: int = 10_000_000,
) -> pd.DataFrame:
    """Join quality_flags to comap_data.

    Returns one row per (obsid, pixel, frequency_band). Adds an `mjd` column
    parsed from `utc_start` and renames pixel/frequency_band to feed/band for
    readability. Note: the DB stores pixel 0-18 (0-indexed) and band 0-7.
    """
    conn = _connect(database_path)
    try:
        query = """
            SELECT q.obsid, q.pixel AS feed, q.frequency_band AS band,
                   q.is_good, q.comment,
                   q.filtered_red_noise, q.filtered_white_noise,
                   q.filtered_auto_rms, q.filtered_noise_index,
                   q.unfiltered_red_noise, q.unfiltered_white_noise,
                   q.unfiltered_auto_rms, q.unfiltered_noise_index,
                   q.n_spikes, q.n_nan_values, q.mean_atm_temp,
                   c.source, c.source_group, c.utc_start, c.level2_path
            FROM quality_flags q
            JOIN comap_data c ON c.obsid = q.obsid
            WHERE q.obsid BETWEEN ? AND ?
        """
        df = pd.read_sql_query(query, conn, params=(min_obsid, max_obsid))
    finally:
        conn.close()

    df["mjd"] = _utc_to_mjd(df["utc_start"])
    df["is_good"] = df["is_good"].astype(bool)

    if source_groups is not None:
        df = df[df["source_group"].isin(source_groups)]
    if mjd_range is not None:
        lo, hi = mjd_range
        df = df[(df["mjd"] >= lo) & (df["mjd"] <= hi)]
    return df.reset_index(drop=True)


def load_observation_summary(database_path: str) -> pd.DataFrame:
    """Return observation_summary joined with comap_data.

    Empty DataFrame if the optional table does not yet exist in this DB.
    """
    conn = _connect(database_path)
    try:
        if not _table_exists(conn, "observation_summary"):
            return pd.DataFrame()
        query = """
            SELECT s.*, c.source, c.source_group, c.utc_start, c.level2_path
            FROM observation_summary s
            JOIN comap_data c ON c.obsid = s.obsid
        """
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    df["mjd"] = _utc_to_mjd(df["utc_start"])
    return df


def load_calibration_flux(
    database_path: str, calibrator: str | None = None
) -> pd.DataFrame:
    """Return per-feed/band/channel calibrator flux measurements."""
    conn = _connect(database_path)
    try:
        if not _table_exists(conn, "calibration_flux"):
            return pd.DataFrame()
        if calibrator is None:
            df = pd.read_sql_query("SELECT * FROM calibration_flux", conn)
        else:
            df = pd.read_sql_query(
                "SELECT * FROM calibration_flux WHERE source = ?",
                conn,
                params=(calibrator,),
            )
    finally:
        conn.close()
    return df


def load_calibration_model(database_path: str) -> pd.DataFrame:
    """Return calibration_model_fit. JSON columns are parsed eagerly."""
    conn = _connect(database_path)
    try:
        if not _table_exists(conn, "calibration_model_fit"):
            return pd.DataFrame()
        df = pd.read_sql_query("SELECT * FROM calibration_model_fit", conn)
    finally:
        conn.close()

    for col in ("poly_coeffs", "nearest_mjds", "nearest_gains"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda s: json.loads(s) if isinstance(s, str) and s else None
            )
    return df


def summarize_coverage(qf: pd.DataFrame, summary: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per source_group, fraction of (obsid, feed, band) rows with populated stats.

    The denominator is the total quality_flags rows in each group (one per
    feed/band/obsid). For observation-level fields (Tsys, tau, calibrator),
    a separate breakdown is appended if `summary` is provided.
    """
    if qf.empty:
        return pd.DataFrame()

    by_group = qf.groupby("source_group", dropna=False)
    rows = []
    for group, sub in by_group:
        row = {
            "source_group": group,
            "n_obsids": sub["obsid"].nunique(),
            "n_rows": len(sub),
            "frac_bad_flag": (~sub["is_good"]).mean(),
        }
        for col in NOISE_COLUMNS:
            row[f"frac_{col}"] = sub[col].notna().mean()
        rows.append(row)
    coverage = pd.DataFrame(rows).set_index("source_group").sort_index()

    if summary is not None and not summary.empty:
        obs_cols = [
            c for c in ("median_tsys", "mean_tau", "calibrator_flux",
                        "calibrator_chi2", "n_scans")
            if c in summary.columns
        ]
        obs_rows = []
        for group, sub in summary.groupby("source_group", dropna=False):
            row = {"source_group": group}
            for col in obs_cols:
                row[f"frac_{col}"] = sub[col].notna().mean()
            obs_rows.append(row)
        obs = pd.DataFrame(obs_rows).set_index("source_group")
        coverage = coverage.join(obs, how="outer")

    return coverage


def load_per_scan_noise_from_hdf5(level2_path: str) -> pd.DataFrame:
    """Read the per-scan noise statistics arrays from a single Level-2 file.

    Returns one row per (feed, band, channel, scan).
    """
    import h5py

    rows = []
    with h5py.File(level2_path, "r") as f:
        if "level2_noise_stats" not in f:
            return pd.DataFrame()
        grp = f["level2_noise_stats"]
        keys = [k for k in grp.keys()]
        arrays = {k: np.asarray(grp[k]) for k in keys}

    if not arrays:
        return pd.DataFrame()

    first = next(iter(arrays.values()))
    if first.ndim != 4:
        return pd.DataFrame()
    nfeed, nband, nchan, nscan = first.shape

    idx = np.indices((nfeed, nband, nchan, nscan)).reshape(4, -1)
    df = pd.DataFrame(
        {
            "feed": idx[0],
            "band": idx[1],
            "channel": idx[2],
            "scan": idx[3],
        }
    )
    for k, arr in arrays.items():
        if arr.shape == (nfeed, nband, nchan, nscan):
            df[k] = arr.reshape(-1)
    return df
