import argparse
from typing import Any, Dict
from tabulate import tabulate  # for nice table formatting
import sqlalchemy as sa
import sys 
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2])) 
from modules.SQLModule.SQLModule import SQLModule, COMAPData, QualityFlag, db

def print_dict_nicely(d: Dict[str, Any], indent: int = 0) -> None:
    """Print dictionary contents with nice formatting"""
    for key, value in d.items():
        # Skip SQLAlchemy internal attributes
        if key.startswith('_'):
            continue
        print(" " * indent + f"{key}: {value}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Query COMAP database for obsid information')
    parser.add_argument('obsid', type=int, help='Observation ID to query')
    parser.add_argument('--database', type=str, default='comap.db',
                       help='Path to database file (default: comap.db)')
    args = parser.parse_args()

    # Connect to database
    #db = SQLModule()
    print(args.database)
    db.connect(args.database)
    inspector = sa.inspect(db.database)
    existing_columns = {col['name'] for col in inspector.get_columns('quality_flags')}
    #sys.exit()

    # Get basic observation data
    obs_data = db.query_data(args.obsid)
    if not obs_data:
        print(f"No data found for obsid {args.obsid}")
        return

    # Print observation data
    print("\n=== COMAP Observation Data ===")
    print_dict_nicely(obs_data)

    # Get quality flags
    flags = (db.session.query(QualityFlag)
            .filter_by(obsid=args.obsid)
            .all())

    if flags:
        print("\n=== Quality Flags ===")
        # Convert to list of dictionaries for tabulate
        flag_data = []
        for flag in flags:
            if not flag.is_good:  # Only show statistics for bad data
                row = {
                    'Pixel': flag.pixel,
                    'Band': flag.frequency_band,
                    'Good': flag.is_good,
                    'Comment': flag.comment or '',
                    'Filtered RN': flag.filtered_red_noise,
                    'Filtered WN': flag.filtered_white_noise,
                    'Filtered RMS': flag.filtered_auto_rms,
                    'Filtered Index': flag.filtered_noise_index,
                    'Unfiltered RN': flag.unfiltered_red_noise,
                    'Unfiltered WN': flag.unfiltered_white_noise,
                    'Unfiltered RMS': flag.unfiltered_auto_rms,
                    'Unfiltered Index': flag.unfiltered_noise_index,
                    'N Spikes': flag.n_spikes,
                    'N NaNs': flag.n_nan_values,
                    'Mean Atm T': flag.mean_atm_temp
                }
                flag_data.append(row)
        
        if flag_data:
            print("\nBAD DATA POINTS:")
            print(tabulate(flag_data, headers='keys', tablefmt='grid'))
        else:
            print("All data points marked as good")

        # Print summary statistics
        total_flags = len(flags)
        bad_flags = sum(1 for f in flags if not f.is_good)
        print(f"\nSummary:")
        print(f"Total pixel-frequency combinations: {total_flags}")
        print(f"Bad data points: {bad_flags}")
        print(f"Good data points: {total_flags - bad_flags}")
        
    db.disconnect()

if __name__ == "__main__":
    main()