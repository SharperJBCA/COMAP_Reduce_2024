import h5py
import numpy as np
from pathlib import Path
import sys 
sys.path.append(str(Path(__file__).parents[2])) 
from modules.SQLModule.SQLModule import SQLModule, COMAPData, QualityFlag, db

def update_noise_stats_from_level2(db: SQLModule, min_obsid: int = None, max_obsid: int = None):
    """
    Update noise statistics in SQL database from existing level 2 files
    
    Args:
        db: SQLModule instance
        min_obsid: Optional minimum obsid to process
        max_obsid: Optional maximum obsid to process
    """
    # Query all observations with level2 files
    query = db.session.query(COMAPData).filter(COMAPData.level2_path.isnot(None))
    if min_obsid:
        query = query.filter(COMAPData.obsid >= min_obsid)
    if max_obsid:
        query = query.filter(COMAPData.obsid <= max_obsid)
    
    observations = query.all()
    
    # Define the mapping between HDF5 paths and SQL columns
    key2sql_map = {
        ('binned_data', 'sigma_white'): 'unfiltered_white_noise',
        ('binned_data', 'sigma_red'): 'unfiltered_red_noise',
        ('binned_data', 'alpha'): 'unfiltered_noise_index',
        ('binned_data', 'auto_rms'): 'unfiltered_auto_rms',
        ('binned_filtered_data', 'sigma_white'): 'filtered_white_noise',
        ('binned_filtered_data', 'sigma_red'): 'filtered_red_noise',
        ('binned_filtered_data', 'alpha'): 'filtered_noise_index',
        ('binned_filtered_data', 'auto_rms'): 'filtered_auto_rms'
    }
    
    NFEEDS = 19  # Assuming these are the constants from your NoiseStats class
    NBANDS = 8
    n_channels = 1  # Adjust if different
    
    for obs in observations:
        print(f"Processing obsid {obs.obsid}")
        if not Path(obs.level2_path).exists():
            print(f"Warning: Level2 file not found: {obs.level2_path}")
            continue
            
        try:
            with h5py.File(obs.level2_path, 'r') as f:
                # Check if noise statistics exist
                if 'NoiseStatistics' not in f:
                    print(f"No noise statistics found for obsid {obs.obsid}")
                    continue
                
                stats_dict = {}  # feed -> band -> stat
                
                # Process each dataset type (binned_data and binned_filtered_data)
                for dataset_base in ['binned_data', 'binned_filtered_data']:
                    dataset_path = f'NoiseStatistics/{dataset_base}'
                    if dataset_path not in f:
                        continue
                        
                    # Process each statistic type
                    for stat_name in ['sigma_white', 'sigma_red', 'alpha', 'auto_rms']:
                        if stat_name not in f[dataset_path]:
                            continue
                            
                        stat_data = f[dataset_path][stat_name][:]
                        
                        # Map the data to feeds and bands
                        for feed in range(1, NFEEDS+1):
                            for band in range(NBANDS * n_channels):
                                if feed not in stats_dict:
                                    stats_dict[feed] = {}
                                if band not in stats_dict[feed]:
                                    stats_dict[feed][band] = {}
                                    
                                key_name = key2sql_map[(dataset_base, stat_name)]
                                iband, ichannel = np.unravel_index(band, (NBANDS, n_channels))
                                stats_dict[feed][band][key_name] = stat_data[feed-1, iband, ichannel]
                
                # Update database with collected statistics
                for feed in stats_dict:
                    for band in stats_dict[feed]:
                        db.update_quality_statistics(
                            obs.obsid,
                            pixel=feed,
                            frequency_band=band,
                            **stats_dict[feed][band]
                        )
                
                print(f"Successfully updated statistics for obsid {obs.obsid}")
                
        except Exception as e:
            print(f"Error processing obsid {obs.obsid}: {str(e)}")
            continue

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Update noise statistics in SQL database from level 2 files')
    parser.add_argument('--database', type=str, default='comap.db',
                       help='Path to database file')
    parser.add_argument('--min-obsid', type=int, help='Minimum obsid to process')
    parser.add_argument('--max-obsid', type=int, help='Maximum obsid to process')
    
    args = parser.parse_args()
    
    db = SQLModule()
    db.connect(args.database)
    
    update_noise_stats_from_level2(db, args.min_obsid, args.max_obsid)
    
    db.disconnect()