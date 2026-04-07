"""
match_hii_to_fields.py

Match HII region fitted positions (Galactic coordinates from
ancillary_data/HII_region_fitted_positions.csv) to the nearest COMAP
field centre (celestial coordinates from data/field_centers.csv), then
produce per-field, per-epoch obsid lists from the database.

An epoch is a run of observations for a field where no consecutive pair
is separated by more than --epoch-gap days (default 7).

Usage:
    python match_hii_to_fields.py
    python match_hii_to_fields.py --hii ancillary_data/HII_region_fitted_positions.csv
                                  --fields data/field_centers.csv
                                  --db databases/COMAP_manchester.db
                                  --max-sep 2.0
                                  --epoch-gap 7
                                  --output-dir hii_field_tables/
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import sqlalchemy as sa
from astropy.coordinates import SkyCoord, Galactic, ICRS
import astropy.units as u

UTC_FMT = "%Y-%m-%d-%H:%M:%S"


def parse_utc(s: str) -> datetime:
    return datetime.strptime(s, UTC_FMT)


def assign_epochs(obs: pd.DataFrame, gap_days: float) -> pd.DataFrame:
    """
    Sort observations by date and assign an epoch number.
    A new epoch starts whenever the gap to the previous observation
    exceeds gap_days.

    Expects obs to have a 'datetime' column (Python datetime objects).
    Returns obs with an added 'epoch' column (1-based integer).
    """
    obs = obs.sort_values("datetime").copy()
    epoch = 1
    epochs = [epoch]
    threshold = timedelta(days=gap_days)
    prev_dt = obs["datetime"].iloc[0]
    for dt in obs["datetime"].iloc[1:]:
        if (dt - prev_dt) > threshold:
            epoch += 1
        epochs.append(epoch)
        prev_dt = dt
    obs["epoch"] = epochs
    return obs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_sexagesimal_ra(s: str) -> float:
    """Parse HH:MM:SS.ss → decimal degrees."""
    parts = s.strip().split(":")
    h, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
    return (h + m / 60.0 + sec / 3600.0) * 15.0


def parse_sexagesimal_dec(s: str) -> float:
    """Parse ±DD:MM:SS.ss → decimal degrees."""
    s = s.strip()
    sign = -1.0 if s.startswith("-") else 1.0
    parts = s.lstrip("+-").split(":")
    d, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
    return sign * (d + m / 60.0 + sec / 3600.0)


def load_field_centers(path: str) -> pd.DataFrame:
    """Load field_centers.csv and add columns ra_deg, dec_deg, glon, glat."""
    df = pd.read_csv(path)
    df["ra_deg"]  = df["ra"].apply(parse_sexagesimal_ra)
    df["dec_deg"] = df["dec"].apply(parse_sexagesimal_dec)

    coords = SkyCoord(ra=df["ra_deg"].values * u.deg,
                      dec=df["dec_deg"].values * u.deg,
                      frame=ICRS())
    galactic = coords.galactic
    df["glon"] = galactic.l.deg
    df["glat"] = galactic.b.deg
    return df


def load_hii_regions(path: str) -> pd.DataFrame:
    """Load HII region CSV and return rows with valid lon_CO26/lat_CO26."""
    df = pd.read_csv(path)
    lon_col = "lon_CO26 [deg]"
    lat_col = "lat_CO26 [deg]"
    if lon_col not in df.columns or lat_col not in df.columns:
        sys.exit(f"ERROR: Expected columns '{lon_col}' and '{lat_col}' in {path}")
    df = df.dropna(subset=[lon_col, lat_col]).copy()
    df = df.rename(columns={lon_col: "glon", lat_col: "glat"})
    return df


def match_sources_to_fields(hii: pd.DataFrame,
                             fields: pd.DataFrame,
                             max_sep_deg: float) -> pd.DataFrame:
    """
    For each HII region find the nearest field centre.

    Returns a DataFrame with the original HII columns plus:
        matched_field  — field_name of the nearest field
        separation_deg — angular separation in degrees
    """
    hii_coords = SkyCoord(l=hii["glon"].values * u.deg,
                          b=hii["glat"].values * u.deg,
                          frame=Galactic())

    field_coords = SkyCoord(l=fields["glon"].values * u.deg,
                             b=fields["glat"].values * u.deg,
                             frame=Galactic())

    # match_to_catalog_sky finds the nearest field for each HII region
    idx, sep2d, _ = hii_coords.match_to_catalog_sky(field_coords)

    result = hii.copy()
    result["matched_field"]  = fields["field_name"].values[idx]
    result["separation_deg"] = sep2d.deg

    if max_sep_deg > 0:
        n_before = len(result)
        result = result[result["separation_deg"] <= max_sep_deg].copy()
        n_dropped = n_before - len(result)
        if n_dropped:
            print(f"  Dropped {n_dropped} HII region(s) with separation > {max_sep_deg} deg")

    return result


def query_observations(db_path: str, field_names: list[str]) -> pd.DataFrame:
    """
    Query the database for obsids, dates, and level2 paths for the given
    field names (matched against the 'source' column, case-insensitive).
    """
    engine = sa.create_engine(f"sqlite:///{db_path}")
    sql = f"""
        SELECT obsid,
               utc_start AS date,
               source    AS field_name,
               level2_path
        FROM   comap_data
        WHERE  LOWER(source) IN ({", ".join(f"lower('{f}')" for f in field_names)})
          AND  level2_path IS NOT NULL
          AND  level2_path != ''
        ORDER  BY source, obsid
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(sa.text(sql), conn)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Match HII regions to COMAP fields and build per-field, per-epoch obsid lists."
    )
    p.add_argument("--hii",    default="ancillary_data/HII_region_fitted_positions.csv")
    p.add_argument("--fields", default="data/field_centers.csv")
    p.add_argument("--db",     default="databases/COMAP_manchester.db")
    p.add_argument(
        "--max-sep", type=float, default=2.0,
        help="Maximum angular separation [deg] to accept a match (0 = no limit, default: 2.0).",
    )
    p.add_argument(
        "--epoch-gap", type=float, default=7.0,
        help="Gap in days that defines a new epoch (default: 7).",
    )
    p.add_argument(
        "--output-dir", default="hii_field_tables",
        help="Directory for output files (default: hii_field_tables/).",
    )
    p.add_argument(
        "--summary", default="hii_field_match_summary.csv",
        help="CSV file summarising which HII region matched which field.",
    )
    p.add_argument(
        "--toml-out", default=None,
        help="Generate a map-making TOML pipeline file (default: <output-dir>/hii_mapmaking.toml).",
    )
    p.add_argument(
        "--band", type=int, default=0,
        help="Band index for generated TOML pipeline entries (default: 0).",
    )
    p.add_argument(
        "--feeds", default="1,3,4,5,6,8,10,11,12,13,14,15,16,17,18,19",
        help="Comma-separated feed list for generated TOML (default: standard 16-feed set).",
    )
    p.add_argument(
        "--n-processes", type=int, default=1,
        help="MPI processes per map-making job in generated TOML (default: 1).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    for path in [args.hii, args.fields, args.db]:
        if not os.path.exists(path):
            sys.exit(f"ERROR: File not found: {path}")

    # ---- Load and match ------------------------------------------------
    print(f"Loading HII regions from {args.hii} ...")
    hii = load_hii_regions(args.hii)
    print(f"  {len(hii)} HII regions with valid CO26 positions")

    print(f"Loading field centres from {args.fields} ...")
    fields = load_field_centers(args.fields)
    print(f"  {len(fields)} field centres")

    print("Matching HII regions to nearest field centre ...")
    matched = match_sources_to_fields(hii, fields, args.max_sep)
    print(f"  {len(matched)} HII regions matched within {args.max_sep} deg")

    if matched.empty:
        print("No matches found — nothing to do.")
        return

    # ---- Summary table -------------------------------------------------
    summary_cols = ["glon", "glat", "matched_field", "separation_deg"]
    # Include Paladini reference coords if present
    for col in ["lon_paladini [deg]", "lat_paladini [deg]"]:
        if col in matched.columns:
            summary_cols = [col] + summary_cols
    summary = matched[summary_cols].sort_values("matched_field").reset_index(drop=True)
    summary.to_csv(args.summary, index=False)
    print(f"\nMatch summary written to: {args.summary}")
    print(summary.to_string(index=False))

    # ---- Query database for each matched field -------------------------
    unique_fields = sorted(matched["matched_field"].unique())
    print(f"\n{len(unique_fields)} unique field(s) matched: {unique_fields}")

    print(f"\nQuerying database ({args.db}) ...")
    obs_df = query_observations(args.db, unique_fields)
    print(f"  {len(obs_df)} observations found with level2 files")

    if obs_df.empty:
        print("No observations with Level-2 files found for matched fields.")
        return

    # ---- Parse dates and assign epochs ---------------------------------
    obs_df["datetime"] = obs_df["date"].apply(parse_utc)
    os.makedirs(args.output_dir, exist_ok=True)

    all_rows = []   # accumulates every row for the combined output

    for field in unique_fields:
        field_obs = obs_df[obs_df["field_name"].str.lower() == field.lower()].copy()
        if field_obs.empty:
            continue

        field_obs = assign_epochs(field_obs, args.epoch_gap)

        hii_sources = matched.loc[
            matched["matched_field"].str.lower() == field.lower(),
            ["glon", "glat", "separation_deg"],
        ].reset_index(drop=True)

        n_epochs = field_obs["epoch"].max()
        print(f"\n  {field}: {len(field_obs)} obs, {n_epochs} epoch(s)")
        print(f"    HII source(s) matched:")
        for _, row in hii_sources.iterrows():
            print(f"      glon={row['glon']:.4f}  glat={row['glat']:.4f}  "
                  f"sep={row['separation_deg']:.3f} deg")

        # --- per-field CSV (obsid, date, level2_path, epoch) ------------
        out_csv = os.path.join(args.output_dir, f"{field}_observations.csv")
        field_obs.drop(columns=["datetime"]).to_csv(out_csv, index=False)
        print(f"    CSV  → {out_csv}")

        # --- per-field plain-text summary, one epoch per block -----------
        out_txt = os.path.join(args.output_dir, f"{field}_obsids_by_epoch.txt")
        with open(out_txt, "w") as fh:
            fh.write(f"# Field: {field}\n")
            for _, src in hii_sources.iterrows():
                fh.write(f"# HII source: glon={src['glon']:.4f}  "
                         f"glat={src['glat']:.4f}  sep={src['separation_deg']:.3f} deg\n")
            fh.write(f"# Epoch gap: {args.epoch_gap} days  |  {n_epochs} epoch(s)\n")
            for ep in range(1, n_epochs + 1):
                ep_obs = field_obs[field_obs["epoch"] == ep].sort_values("datetime")
                start  = ep_obs["datetime"].iloc[0].strftime("%Y-%m-%d")
                end    = ep_obs["datetime"].iloc[-1].strftime("%Y-%m-%d")
                fh.write(f"\n[epoch {ep}]  {start} – {end}  ({len(ep_obs)} obs)\n")
                for _, row in ep_obs.iterrows():
                    fh.write(f"{row['obsid']}  {row['date']}  {row['level2_path']}\n")
                print(f"      epoch {ep}: {start} – {end}  ({len(ep_obs)} obs)  "
                      f"obsids: {ep_obs['obsid'].tolist()}")
        print(f"    TXT  → {out_txt}")

        # --- per-epoch obsid-only files (one obsid per line) -----------
        epoch_dir = os.path.join(args.output_dir, "obsid_lists")
        os.makedirs(epoch_dir, exist_ok=True)
        for ep in range(1, n_epochs + 1):
            ep_obs = field_obs[field_obs["epoch"] == ep].sort_values("datetime")
            epoch_file = os.path.join(epoch_dir, f"{field}_epoch{ep:02d}.txt")
            with open(epoch_file, "w") as fh:
                for obsid in ep_obs["obsid"]:
                    fh.write(f"{obsid}\n")
        print(f"    Obsid lists → {epoch_dir}/{field}_epoch*.txt")

        # accumulate for combined output
        field_obs["hii_glon"] = hii_sources["glon"].iloc[0]
        field_obs["hii_glat"] = hii_sources["glat"].iloc[0]
        field_obs["separation_deg"] = hii_sources["separation_deg"].iloc[0]
        all_rows.append(field_obs)

    # ---- Combined CSV (all fields, all epochs) -------------------------
    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined = combined.drop(columns=["datetime"])
        combined_path = os.path.join(args.output_dir, "all_fields_observations.csv")
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined table written to: {combined_path}")

    # ---- Generate map-making TOML --------------------------------------
    toml_out = args.toml_out or os.path.join(args.output_dir, "hii_mapmaking.toml")
    generate_mapmaking_toml(
        toml_out,
        unique_fields,
        all_rows,
        args,
    )


def generate_mapmaking_toml(
    toml_path: str,
    fields: list,
    all_rows: list,
    args,
):
    """
    Write a map-making TOML with one pipeline entry per field+epoch,
    each pointing at the per-epoch obsid list file via obsid_list_file.
    """
    feeds_str = f"[{args.feeds}]"
    epoch_dir = os.path.join(args.output_dir, "obsid_lists")
    # Use absolute paths so the TOML works from any working directory
    abs_epoch_dir = os.path.abspath(epoch_dir)

    pipeline_entries = []
    for field_df in all_rows:
        if field_df.empty:
            continue
        field = field_df["field_name"].iloc[0]
        n_epochs = field_df["epoch"].max()
        for ep in range(1, n_epochs + 1):
            ep_obs = field_df[field_df["epoch"] == ep]
            obsid_file = os.path.join(abs_epoch_dir, f"{field}_epoch{ep:02d}.txt")
            map_name = f"hii_{field}_epoch{ep:02d}"
            entry = (
                f'modules.MapMaking.MapMakingModules.CreateMap.CreateMap('
                f'feeds={feeds_str}, '
                f'band={args.band}, '
                f'map_name="{map_name}", '
                f'obsid_list_file="{obsid_file}", '
                f'tod_data_name="level2/binned_filtered_data", '
                f'database_file="{args.db}", '
                f'n_processes={args.n_processes}, '
                f'write_mean_azel=True'
                f')'
            )
            pipeline_entries.append(entry)

    with open(toml_path, "w") as fh:
        fh.write("[Master]\n\n")
        fh.write(f"log_file = 'logs/hii_mapmaking.log'\n\n")
        fh.write(f"sql_database = '{args.db}'\n\n")
        fh.write("pipeline = [\n")
        for entry in pipeline_entries:
            fh.write(f"    '{entry}',\n")
        fh.write("]\n\n")
        fh.write("[modules.MapMaking.MapMakingModules.CreateMap.CreateMap]\n\n")
        maps_dir = os.path.join(args.output_dir, "maps")
        fh.write(f"output_dir = '{maps_dir}'\n")
        fh.write("wcs_def = 'dynamic'\n")
        fh.write("dynamic_wcs_cdelt = 0.0166666\n")
        fh.write("dynamic_wcs_padding = 0.25\n")
        fh.write("offset_length = 100\n")
        fh.write("write_mean_azel = true\n")

    print(f"\nMap-making TOML written to: {toml_path}")
    print(f"  {len(pipeline_entries)} map-making job(s)")
    print(f"  Run with: python run.py {toml_path}")


if __name__ == "__main__":
    main()
