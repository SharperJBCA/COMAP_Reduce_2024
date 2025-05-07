# ObserverFlags.py 
# 
# Description: 
#  Want to read in the observer flags csv file and update flags in the SQL database
#  Steps:
#   1) Read in the csv file into a dictionary of obsids, each with a sub dictionary containing a boolean for each feed/band 
#   2) Update the SQL database QualityFlag (quality_flags) table which needs:
#       - obsid
#       - feed/pixel
#       - band
#       - is_good (boolean) -> We will use this for observer flags. 

import logging 

import subprocess
from datetime import datetime
import numpy as np
import os
import stat
import pwd
import grp
import glob
from tqdm import tqdm 
import sys 
import multiprocessing
import time 
import csv 
from sqlalchemy import and_, or_

try:
    from modules.SQLModule.SQLModule import db, COMAPData, QualityFlag
except ModuleNotFoundError:
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))    
    from modules.SQLModule.SQLModule import db, COMAPData, QualityFlag

class ObserverFlags:

    def __init__(self, observer_flags_csv_file='', dbconnect=None, **kwargs):
        self.observer_flags_csv_file = observer_flags_csv_file
        self.db=dbconnect

    def run(self): 
        logging.info('Running ObserverFlags') 

        # Step 1) Read in the observer flags csv file
        observer_flags_list = self.read_observer_flags_csv(self.observer_flags_csv_file)

        # Step 1b) Set all observer flags to True 
        self.clean_sql_database()

        # Step 2) Update the SQL database quality_flags table 
        self.update_sql_database(observer_flags_list)

    def read_observer_flags_csv(self, file_path):
        """
        Read observer flags csv into a list of dictionaries.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            list: List of dictionaries with keys 'obsid_start', 'obsid_end', 'pixels', 'sidebands'
        """
        result = []
        
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Process obsid_start and obsid_end
                obsid_start = row['ObsID start']
                obsid_end = row['ObsID end']
                
                # Convert to integers if not empty
                if obsid_start:
                    obsid_start = int(obsid_start)
                if obsid_end:
                    if isinstance(obsid_end, str):
                        if obsid_end == 'ongoing':
                            obsid_end = -1
                        else:
                            obsid_end = int(obsid_end)
                else:
                    obsid_end = None
                # Process pixels
                pixels_value = row['Pixels'].strip()
                pixels = []
                
                if pixels_value.lower() == 'all' or not any(c.isdigit() for c in pixels_value):
                    # If 'all' or any non-numeric value, return all pixels 1-19
                    pixels = list(range(1, 20))
                else:
                    # Remove quotes and split by comma
                    pixels_value = pixels_value.replace('"', '')
                    for item in pixels_value.split(','):
                        item = item.strip()
                        if item and item.isdigit():
                            pixels.append(int(item))
                
                # Process sidebands
                sideband_value = row['Sidebands'].strip().lower()
                sidebands = []
                
                if sideband_value == 'all':
                    sidebands = [0, 1, 2, 3]
                elif sideband_value == 'banda':
                    sidebands = [0, 1]
                elif sideband_value == 'bandb':
                    sidebands = [2, 3]
                    
                # Create dictionary and append to result
                entry = {
                    'obsid_start': obsid_start,
                    'obsid_end': obsid_end,
                    'pixels': pixels,
                    'sidebands': sidebands
                }
                result.append(entry)
        
        return result

    def clean_sql_database(self):
        """
        Set all the quality_flags is_good values to True before setting the observer flags. 
        """

        self.db._connect()
        self.db.session.query(QualityFlag).update(
            {"is_good": True, "comment": "Observer Flag"}, 
            synchronize_session=False
        )
        self.db.session.commit()
        self.db._disconnect()

    def update_sql_database(self,observations_list):
        """
        Update the quality_flags table based on the parsed CSV data.
        
        Args:
            db: SQLModule instance connected to the database
            observations_list: List of dictionaries containing obsid_start, obsid_end, pixels, and sidebands
            reason: Optional reason to include in the comment field
        
        Returns:
            dict: Statistics about the update operation
        """
        max_obsid = max(db.query_all_obsids())
        updates_by_obsid = {}
        for obs in tqdm(observations_list, desc='Updating SQL Database'):
            obsid_start = obs['obsid_start']
            obsid_end = obs['obsid_end']
            pixels = obs['pixels']
            sidebands = obs['sidebands']
            
            # Handle case where only a single obsid is specified
            if not obsid_end and obsid_start:
                obsid_end = obsid_start
            if obsid_end == -1:
                obsid_end = max_obsid 

            # Skip entries with no obsid_start
            if not obsid_start:
                continue
                
            # Determine range of observation IDs to update
            obsid_range = range(obsid_start, obsid_end + 1) if obsid_end else [obsid_start]

            
            self.db._connect()
            valid_obsids = self.db.session.query(COMAPData.obsid).filter(
                COMAPData.obsid >= obsid_start,
                COMAPData.obsid <= obsid_end
            ).order_by(COMAPData.obsid).all()
            self.db._disconnect()
            # Flatten the result tuples into a simple list
            valid_obsids = [obsid[0] for obsid in valid_obsids]
                    
            # Process each observation ID in the range
            for obsid in valid_obsids:
                # Check if this obsid exists in the database
                if not db.obsid_exists(obsid):
                    #print(f"Warning: ObsID {obsid} does not exist in the database, skipping")
                    continue
                
                if obsid not in updates_by_obsid:
                    updates_by_obsid[obsid] = set() 

                for pixel in pixels:
                    for sideband in sidebands:
                        updates_by_obsid[obsid].add((pixel, sideband))

        # Proces updates in batches by obsid 
        print(f"Updating {len(updates_by_obsid)} observations")
        total_updates = 0 

        # Use transaction for all updates
        for obsid, flag_combinations in tqdm(updates_by_obsid.items(), desc='Updating Observations'):
            if not flag_combinations:
                continue 

            # build a filter for this obsid and all pixel/sideband combinations 
            conditions = [] 
            for pixel, sideband in flag_combinations:
                conditions.append(
                    and_(QualityFlag.pixel == pixel, QualityFlag.frequency_band == sideband)
                )

            if conditions:
                self.db._connect()
                result = self.db.session.query(QualityFlag).filter(
                    and_(QualityFlag.obsid == obsid, or_(*conditions))
                ).update(
                    {"is_good": False, "comment": "Observer Flag"}, 
                    synchronize_session=False
                )
                self.db._disconnect() 
                total_updates += result
        print(f"Updated {total_updates} quality flags across {len(updates_by_obsid)} observations")

        

if __name__ == "__main__": 
    # update path 
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))    
    print(str(Path(__file__).parent.parent))
    db.connect(sys.argv[2])

    ob = ObserverFlags(observer_flags_csv_file=sys.argv[1],dbconnect=db) 
    ob.run()
    db.disconnect() 