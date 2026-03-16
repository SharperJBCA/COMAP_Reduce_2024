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

    def __init__(self, level1_data_paths=[], level2_data_paths=[], overwrite_existing=False, 
                 validate_level2_paths=True, clear_quality_stats_on_missing=False,**kwargs) -> None:
        logging.info("Initializing DataAssess")

        self.overwrite_existing = overwrite_existing
        self.level1_data_paths = level1_data_paths
        self.level2_data_paths = level2_data_paths
        self.validate_level2_paths = validate_level2_paths
        self.clear_quality_stats_on_missing = clear_quality_stats_on_missing

        for k,v in kwargs.items():
            logging.info(f'{k}: {v}')
            setattr(self, k, v)

    def run(self) -> None:
        logging.info("Running DataAssess")


        # Step 0) Validate that current DB Level-2 paths exist
        if self.validate_level2_paths:
            logging.info("Validating existing Level-2 paths in DB...")
            self.validate_existing_level2_paths()

        # Step 1) Check which files we already have
        logging.info('Checking which files we already have')

        level1_files = self.get_file_paths_metadata(self.level1_data_paths, is_level2=False)
        level2_files = self.get_file_paths_metadata(self.level2_data_paths, is_level2=True)
        # Step 2) Cross reference with SQL database
        #level1_missing_obsids = self.check_sql_database(level1_files, level2_files, 
        #                                                overwrite_existing=self.overwrite_existing)
        discovered = sorted(set(level1_files) | set(level2_files))
        if self.overwrite_existing:
            missing_obsids = discovered
        else:
            existing = db.query_obsid_list(discovered)  # {obsid: rowdict}
            missing_obsids = [o for o in discovered if o not in existing]

        # Step 3) Update SQL database with missing files
        self.update_sql_database(missing_obsids, level1_files)
        self.update_sql_database(missing_obsids, level2_files)

        if not self.overwrite_existing:
            to_fix = []
            for obsid, meta in level2_files.items():
                row = existing.get(obsid)
                if not row:
                    continue
                if (row.get('level2_path') in (None, "",)) and ('level2_path' in meta):
                    to_fix.append(obsid)
        else:
            to_fix = list(level2_files.keys())

        self.update_sql_database(to_fix, level2_files)

    def validate_existing_level2_paths(self) -> None:
        """
        Check all DB rows with a Level-2 path. If a file is missing on disk,
        clear Level-2 info (and optionally flag stats).
        """
        missing = db.scrub_missing_level2_paths(clear_quality_stats=self.clear_quality_stats_on_missing)
        if missing:
            logging.warning(f"Cleared Level-2 info for {len(missing)} obsids with missing files: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
        else:
            logging.info("All existing Level-2 paths in DB point to files that exist.")

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
        discovered_obsids = sorted(set(level1_files.keys()) | set(level2_files.keys()))
        if overwrite_existing:
            return discovered_obsids

        # query DB only once
        result_existing = db.query_obsid_list(discovered_obsids)
        missing = [obsid for obsid in discovered_obsids if obsid not in result_existing]
        return missing


    @staticmethod
    def get_file_paths_metadata(data_paths : list, is_level2=False) -> None:
        """
        Find all of the data files and return a dictionary of metadata + file path
        """
        file_path_key = 'level2_path' if is_level2 else 'level1_path'
        logging.info(f'Getting {"level 2" if is_level2 else "level 1"} files')

        file_info = {}
        for path in data_paths:
            file_paths = glob.glob(path + '/./**/*comap*.hd5',recursive=True)
            logging.info(f'Found {len(file_paths)} files in {path}')
            for file_path in tqdm(file_paths, desc='Reading files'):
                try:
                    obsid_from_path = get_obsid_from_path(file_path)
                except ValueError as e:
                    logging.debug(f'Error getting obsid from path: {file_path}')
                    logging.debug(e)
                    continue

                try:
                    with h5py.File(file_path, 'r') as f:
                        if 'comap' not in f:
                            logging.info(f'No comap group in file: {file_path}')
                            continue
                        comap_metadata = {k: v for k, v in f['comap'].attrs.items()}
                        if 'obsid' not in comap_metadata:
                            logging.info(f'No obsid in file: {file_path}')
                            continue
                        comap_metadata['obsid'] = int(comap_metadata['obsid'])
                        # Only accept entries that match the obsid we parsed from the filename,
                        # if you want an extra sanity check:
                        if comap_metadata['obsid'] != obsid_from_path:
                            logging.debug(f'obsid mismatch: attrs={comap_metadata["obsid"]} path={obsid_from_path}')
                        comap_metadata[file_path_key] = file_path
                        comap_metadata['source'], comap_metadata['source_group'] = COMAPData.get_source_info(comap_metadata)
                except OSError:
                    logging.error(f'Error reading file: {file_path}')
                    continue
                file_info[comap_metadata['obsid']] = comap_metadata 
        return file_info
    
