# DataAssess.py 
#
# Description:
#  Check what data files already exist on local system and
#  compare with data files in the SQL database.
#  Update any missing files to database. 

import logging
import h5py 
import glob 
from tqdm import tqdm 
from modules.SQLModule.SQLModule import db, COMAPData

def get_obsid_from_path(file_path: str) -> int:
    """
    Get the obsid from the file path
    """
    if 'Level2' in file_path:
        obsid = file_path.split('/')[-1].split('_')[-1].split('-')[1]
    else: 
        obsid = file_path.split('/')[-1].split('-')[1]
    return int(obsid)

class DataAssess: 

    def __init__(self, level1_data_paths=[], level2_data_paths=[], overwrite_existing=False, **kwargs) -> None:
        logging.info("Initializing DataAssess")

        self.overwrite_existing = overwrite_existing
        self.level1_data_paths = level1_data_paths
        self.level2_data_paths = level2_data_paths
        for k,v in kwargs.items():
            logging.info(f'{k}: {v}')
            setattr(self, k, v)

    def run(self) -> None:
        logging.info("Running DataAssess")

        # Step 1) Check which files we already have
        logging.info('Checking which files we already have')
        level1_files = self.get_file_paths_metadata(self.level1_data_paths)
        level2_files = self.get_file_paths_metadata(self.level2_data_paths, is_level2=True)

        # Step 2) Cross reference with SQL database
        level1_missing_obsids = self.check_sql_database(level1_files, level2_files, 
                                                        overwrite_existing=self.overwrite_existing)

        # Step 3) Update SQL database with missing files
        self.update_sql_database(level1_missing_obsids, level1_files)
        self.update_sql_database(level1_missing_obsids, level2_files)

    @staticmethod
    def update_sql_database(obsids: list, data_files: dict) -> None:
        """
        Update the SQL database with missing files
        """
        for obsid in obsids:
            if obsid in data_files:
                db.insert_or_update_data(data_files[obsid])
            else:
                logging.error(f'Error: Obsid {obsid} not found in level1 or level2 files')

    def check_sql_database(self, level1_files: dict, level2_files: dict, overwrite_existing: bool = False) -> list: 
        """
        Query the SQL database to check which files already exist 
        """
        level1_obsids = list(level1_files.keys())
        level1_result = db.query_obsid_list(level1_obsids)

        if overwrite_existing:
            return level1_obsids
        
        level1_missing_obsids = [obsid for obsid in level1_obsids if obsid not in level1_result]

        return level1_missing_obsids 


    @staticmethod
    def get_file_paths_metadata(data_paths : list, is_level2=False) -> None:
        """
        Find all of the data files and return a dictionary of metadata + file path
        """
        if is_level2:
            condition = lambda f:'Level2' in f
            file_path_key = 'level2_path'
            logging.info('Getting level 2 files')
        else:
            condition = lambda f:not 'Level2' in f
            file_path_key = 'level1_path'
            logging.info('Getting level 1 files')
        file_info = {}
        for path in data_paths:
            file_paths = glob.glob(path + '/./**/*comap*.hd5',recursive=True)
            logging.info(f'Found {len(file_paths)} files in {path}')
            for file_path in tqdm(file_paths, desc='Reading files'):
                try:
                    obsid_from_path = get_obsid_from_path(file_path)
                except ValueError as e:
                    print(e)
                    print(file_path) 
                    continue
                if db.obsid_exists(obsid_from_path):
                    logging.info(f'Obsid {obsid_from_path} already in database')
                    continue

                if condition(file_path.rsplit('/')[-1]):
                    try:
                        with h5py.File(file_path, 'r') as f:
                            if not 'comap' in f:
                                logging.info(f'No comap group in file: {file_path}')
                                continue
                            comap_metadata = {k:v for k,v in f['comap'].attrs.items()} 
                            if not 'obsid' in comap_metadata:
                                logging.info(f'No obsid in file: {file_path}')
                                continue
                            comap_metadata[file_path_key] = file_path
                            comap_metadata['obsid'] = int(comap_metadata['obsid'])
                            comap_metadata['source'], comap_metadata['source_group'] = COMAPData.get_source_info(comap_metadata) 
                    except OSError as e:
                        logging.error(f'Error reading file: {file_path}')
                        continue
                    file_info[comap_metadata['obsid']] = comap_metadata 
                
        return file_info
    
