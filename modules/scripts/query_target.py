import argparse
from sqlalchemy import and_, or_
import sqlalchemy as sa
import sys 
import h5py
from astropy.coordinates import SkyCoord
import os 
from astropy.time import Time, TimezoneInfo 
import astropy.units as u 
from datetime import datetime

from pathlib import Path
sys.path.append(str(Path(__file__).parents[2])) 
from modules.SQLModule.SQLModule import SQLModule, COMAPData, QualityFlag, db
from tqdm import tqdm
from tabulate import tabulate

timezone_pacific = TimezoneInfo(utc_offset=-8*u.hour)

def check_stats_exist(flag_stats):
    """Check if any noise statistics exist for this observation-pixel-band"""
    stat_fields = [
        'filtered_red_noise', 'filtered_white_noise', 'filtered_auto_rms', 'filtered_noise_index',
        'unfiltered_red_noise', 'unfiltered_white_noise', 'unfiltered_auto_rms', 'unfiltered_noise_index'
    ]
    return any(getattr(flag_stats, field) is not None for field in stat_fields)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Query COMAP database for observations of a specific target with Level2 data')
    parser.add_argument('source', type=str, help='Target source name (e.g., "CygA")')
    parser.add_argument('--database', type=str, default='comap.db',
                       help='Path to database file (default: comap.db)')
    parser.add_argument('--max-obsid', type=int, default=None, 
                        help='Maximum ObsID to query (default: None)')
    parser.add_argument('--min-obsid', type=int, default=None,
                        help='Minimum ObsID to query (default: None)')
    parser.add_argument('--show-coordinates', action='store_true', default=False)
    parser.add_argument('--show-all', action='store_true', default=False)
    parser.add_argument('--show-path', action='store_true', default=False)
    parser.add_argument('--show-level2-path', action='store_true', default=False)
    parser.add_argument('--show-local-time', action='store_true', default=False)
    parser.add_argument('--include-skydips', action='store_true', default=False)
    args = parser.parse_args()

    # Connect to database
    db = SQLModule()
    db.connect(args.database)

    if args.show_all:
        conditions = and_(COMAPData.source.contains(f'%{args.source}%'),
                         or_(COMAPData.source_group.like('Foreground'), COMAPData.source_group.like('Galactic'), COMAPData.source_group.like('Calibrator')))
    else:
        conditions = and_(COMAPData.source.contains(f'%{args.source}%'), 
                          COMAPData.level2_path.isnot(None),
                         or_(COMAPData.source_group.like('Foreground'), COMAPData.source_group.like('Galactic'), COMAPData.source_group.like('Calibrator')))

    # Query for observations matching criteria
    observations = (db.session.query(COMAPData)
                   .filter(conditions)
                   .order_by(COMAPData.obsid)
                   .all())
    
    if args.max_obsid:
        observations = [obs for obs in observations if obs.obsid <= args.max_obsid]

    if args.min_obsid:
        observations = [obs for obs in observations if obs.obsid >= args.min_obsid]
        
    # Prepare data for output
    table_data = []
    has_level2_data_count = 0
    for obs in tqdm(observations):
        # Check if any pixel-band combination has statistics
        has_stats = False
        has_level2_data = False 
        quality_flags = (db.session.query(QualityFlag)
                        .filter_by(obsid=obs.obsid)
                        .all())
        
        if quality_flags:
            has_stats = any(check_stats_exist(flag) for flag in quality_flags)

        if not os.path.exists(obs.level1_path):
            print(f"Level1 path {obs.level1_path} does not exist")

        try:
            with h5py.File(obs.level1_path) as f:

                if args.show_coordinates:
                    ra = f['spectrometer/pixel_pointing/pixel_ra'][0,0]
                    dec = f['spectrometer/pixel_pointing/pixel_dec'][0,0]
                    # convert to galactic 
                    c = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')
                    gl = c.galactic.l.deg
                    gb = c.galactic.b.deg

                utc_start = f['comap'].attrs['utc_start']
                local_start_time = Time(datetime.strptime(utc_start, '%Y-%m-%d-%H:%M:%S'))
                local_start_time = local_start_time.to_datetime(timezone=timezone_pacific).strftime('%Y-%m-%d %H:%M:%S')
        except (KeyError,OSError): 
            continue 

        try:
            if obs.level2_path and os.path.exists(obs.level2_path):
                with h5py.File(obs.level2_path) as f:
                    if 'level2/binned_filtered_data' in f:
                        has_level2_data_count += 1
                        has_level2_data = True
        except (KeyError,OSError):
            continue 


        output_info = [
            obs.obsid,
            obs.source,
            obs.source_group,
            "Yes" if has_stats else "No",
            "Yes" if has_level2_data else "No",
        ]
        if args.show_coordinates:
            output_info.append(gl)
            output_info.append(gb)
        if args.show_path:
            output_info.append(obs.level1_path)
        if args.show_local_time:
            output_info.append(local_start_time)
        if args.show_level2_path:
            output_info.append(obs.level2_path)

        table_data.append(output_info)

    # Print results
    if table_data:
        print(f"\nFound {len(observations)} observations of {args.source} with Level2 data:")
        headers = ["ObsID", "Source", "Source Group", "Has Stats", "Has Level2 Data"]
        if args.show_coordinates:
            headers.append('GL')
            headers.append('GB')
        if args.show_path:
            headers.append("Level1 Path")
        if args.show_local_time:
            headers.append("Local Start Time")
        if args.show_level2_path:
            headers.append("Level2 Path")

        print(tabulate(table_data, 
                      headers=headers,
                      tablefmt="grid"))
        with open(f'{args.source}_obs.txt', 'w') as f:
            f.write(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        print(f"\nNo observations found for {args.source} with Level2 data")

    # Print summary
    total_obs = db.session.query(COMAPData).filter(and_(COMAPData.source.contains(f'%{args.source}%'),
                                                       or_(COMAPData.source_group.like('Foreground'), 
                                                           COMAPData.source_group.like('Galactic'),
                                                           COMAPData.source_group.like('Calibrator')))).count()
    obs_with_stats = sum(1 for row in table_data if row[2] == "Yes")
    
    print(f"\nSummary:")
    print(f"Total observations of {args.source}: {total_obs}")
    print(f"Observations with Level2 data: {has_level2_data_count}")
    print(f"Observations without Level2 data: {total_obs - has_level2_data_count}")
    print(f"Observations with noise statistics: {obs_with_stats}")
    print(f"Observations without noise statistics: {len(observations) - obs_with_stats}")
    db.disconnect()

if __name__ == "__main__":
    main()