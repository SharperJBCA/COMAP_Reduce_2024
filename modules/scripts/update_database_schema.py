import shutil
import os
from datetime import datetime
from sqlalchemy.orm import Session
import sqlalchemy as sa

import sys 
from pathlib import Path 
sys.path.append(str(Path(__file__).parents[2])) 
from modules.SQLModule.SQLModule import SQLModule, COMAPData, QualityFlag, FileFlag

def backup_and_update_database(original_db_path: str):
    """
    Create backup of database and add QualityFlag table with default values
    """
    # 1. Create backup with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{original_db_path}.backup_{timestamp}"
    
    print(f"Creating backup at: {backup_path}")
    shutil.copy2(original_db_path, backup_path)
    
    # 2. Load original database
    print("Connecting to original database...")
    db = SQLModule()
    db.connect(original_db_path)
    
    # 3. Check existing columns and add missing ones
    inspector = sa.inspect(db.database)
    existing_columns = {col['name'] for col in inspector.get_columns('quality_flags')}
    
    # Define all the noise statistic columns we want to add
    new_columns = {
        'filtered_red_noise': sa.Float,
        'filtered_white_noise': sa.Float,
        'filtered_auto_rms': sa.Float,
        'filtered_noise_index': sa.Float,
        'unfiltered_red_noise': sa.Float,
        'unfiltered_white_noise': sa.Float,
        'unfiltered_auto_rms': sa.Float,
        'unfiltered_noise_index': sa.Float,
        'n_spikes': sa.Integer,
        'n_nan_values': sa.Integer,
        'mean_atm_temp': sa.Float
    }
    
    # Add any missing columns
    with db.database.begin() as conn:
        for col_name, col_type in new_columns.items():
            if col_name not in existing_columns:
                print(f"Adding column {col_name}")
                conn.execute(sa.text(
                    f'ALTER TABLE quality_flags ADD COLUMN {col_name} {col_type.__visit_name__}'
                ))
    sys.exit() 

    # 3. Create new QualityFlag table
    print("Creating QualityFlag table...")
    QualityFlag.__table__.create(db.database, checkfirst=True)
    
    # 4. Add default quality flags for all existing observations
    print("Adding default quality flags...")
    with Session(db.database) as session:
        # Get all existing obsids
        obsids = session.query(COMAPData.obsid).all()
        
        # For each obsid, create default good flags for all pixel/frequency combinations
        for (obsid,) in obsids:
            print(f"Processing obsid: {obsid}")
            flags = []
            for pixel in range(19):  # 19 pixels
                for freq in range(8):  # 8 frequency bands
                    flags.append(
                        QualityFlag(
                            obsid=obsid,
                            pixel=pixel,
                            frequency_band=freq,
                            is_good=True,
                            comment=None,
                            filtered_red_noise=None,
                            filtered_white_noise=None,
                            filtered_auto_rms=None,
                            filtered_noise_index=None,
                            unfiltered_red_noise=None,
                            unfiltered_white_noise=None,
                            unfiltered_auto_rms=None,
                            unfiltered_noise_index=None,
                            n_spikes=None,
                            n_nan_values=None,
                            mean_atm_temp=None
                        )
                    )
            
            # Add flags in batches to improve performance
            session.bulk_save_objects(flags)
            session.commit()
    
    print("Database update complete!")
    print(f"Original database: {original_db_path}")
    print(f"Backup created at: {backup_path}")

if __name__ == "__main__":
    # Replace with your database path
    db_path = "databases/COMAP_manchester.db"
    backup_and_update_database(db_path)