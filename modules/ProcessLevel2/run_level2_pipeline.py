import os 
import matplotlib
matplotlib.use('agg')

from astropy.io import fits 
import sys 
sys.path.append(os.getcwd())
import logging
import argparse
import toml
import numpy as np
import time 
from modules.SQLModule.SQLModule import db 
import importlib
from modules.pipeline_control.Pipeline import BadCOMAPFile, update_log_variable,setup_logging
from modules.SQLModule.SQLModule import QualityFlag, FileFlag
from datetime import datetime



def process_files(filelist: list, parameters: dict, rank: int, base_delay : float = 2 ) -> None:
    """
    Process a list of files
    """

    process_pid = os.getpid()
    delay = rank * base_delay
    time.sleep(delay) 
    for file in filelist:
        # Loop over the Master pipeline 
        update_log_variable('{:06d} Rank {:02d} PID {}'.format(file.obsid, rank, process_pid))
        logging.info(f"Processing file: {os.path.basename(file.level1_path)}")
        for module_info in parameters['Master']['_pipeline']:
            logging.info(f"Running module: {module_info['module']}")
            package = module_info['package']
            module = module_info['module']
            args = module_info['args']
            
            # Import the module
            logging.info(f"Importing module: {package}.{module}")
            module = getattr(importlib.import_module(package), module)
            module = module(**args)
            module.set_lock_file_path(parameters['Master']['lock_files_folder'])
            logging.info(f"Module imported: {module.__class__.__name__}")
            # Execute the module
            try:
                module.run(file)
            except BadCOMAPFile as e:
                logging.error(f"Error processing file: {file.level1_path}")
                logging.error(e)
                db.update_quality_flag_all(file.obsid, FileFlag.BAD_FILE)
                break
    update_log_variable('None')
    print('Main loop done')
    
def main():
    
    parser = argparse.ArgumentParser(description='Run the map making code')
    parser.add_argument('file_list', type=str, help='Path to the file list text file')
    parser.add_argument('config_file', type=str, help='Path to the configuration file')
    parser.add_argument('rank', type=int, help='Rank of the process')
    parser.add_argument('log_file_name', type=str, help='Name of the log file')
    args = parser.parse_args()

    setup_logging(args.log_file_name, prefix_date=False)
    parameters = toml.load(args.config_file) 
    filelist = [int(n) for n in np.loadtxt(args.file_list, dtype=int, ndmin=1)]

    db.connect(parameters['Master']['sql_database'])

    filelist = db.query_obsid_list(filelist, return_list=True, return_dict=False)
    rank = args.rank

    process_files(filelist, parameters, rank)
    sys.exit()

if __name__ == "__main__":
    print('HELLO')
    main()  