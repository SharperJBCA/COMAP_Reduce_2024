# Pipeline.py 
#
# Description:
#  The pipeline module is responsible for executing and managing logs associated with
# the COMAP pipeline. 

import os 
import logging 
from modules.parameter_files.parameter_files import read_parameter_file
import importlib
from modules.SQLModule.SQLModule import db 
import multiprocessing as mp


def setup_logging(filename):
    """Setup the logging configuration for the pipeline"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Setup log file 
    logging.basicConfig(filename=filename, level=logging.INFO,format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

class Level2Pipeline: 

    def __init__(self, config_file : str = '', MIN_OBSID=7000) -> None:

        self.MIN_OBSID = MIN_OBSID

        self.config_file = config_file
        if os.path.exists(config_file):
            self.parameters = read_parameter_file(self.config_file) 
        else:
            self.parameters = {} 

    def split_filelist(self, filelist: list, nprocess: int) -> list: 
        """
        Split a filelist into nprocess chunks
        """
        return [filelist[i::nprocess] for i in range(nprocess)]
    
    def process_files(self, filelist: list, parameters: dict) -> None:
        """
        Process a list of files
        """


        for file in filelist:
            # Loop over the Master pipeline 
            logging.info(f"Processing file: {os.path.basename(file.level1_path)}")
            for module_info in parameters['Master']['_pipeline']:
                logging.info(f"Running module: {module_info['module']}")
                package = module_info['package']
                module = module_info['module']
                args = module_info['args']
                # Import the module
                module = getattr(importlib.import_module(package), module)
                module = module(**args)
                # Execute the module
                module.run(file)

    def execute_parallel(self, chunk_filelists: list, parameters: dict) -> None:
        # Create a pool of workers
        processes = [] 
        for filelist in chunk_filelists:
            p = mp.Process(target=self.process_files, args=(filelist, parameters))
            processes.append(p)
            p.start()
        
        # Wait for all processes to finish
        for p in processes:
            p.join()


    def run(self):
        
        # Read in the pipeline configuration file

        # Setup logging
        setup_logging(self.parameters['Master']['log_file'])

        # Setup the SQL database
        db.connect(self.parameters['Master']['sql_database'])

        # Get the source group and source, if None just get all files
        target_source_group = self.parameters['Master'].get('source_group', None)
        target_source = self.parameters['Master'].get('source', None)

        # Get the list of files to be processed 
        level1_filelist = db.get_unprocessed_files(source_group=target_source_group, source=target_source, min_obsid=self.MIN_OBSID, overwrite=True)

        # Loop over the filelist and process each file
        chunk_filelists = self.split_filelist(level1_filelist, self.parameters['Master']['nprocess'])

        # Run the pipeline in parallel
        self.execute_parallel(chunk_filelists, self.parameters)
        
        db.disconnect() 
