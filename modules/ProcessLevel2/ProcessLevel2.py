# Pipeline.py 
#
# Description:
#  The pipeline module is responsible for executing and managing logs associated with
# the COMAP pipeline. 

import os 
import sys
import logging 
from modules.parameter_files.parameter_files import read_parameter_file
import importlib
from modules.SQLModule.SQLModule import db 
import multiprocessing as mp
import time 
from pathlib import Path
import subprocess
import toml
from modules.pipeline_control.Pipeline import BadCOMAPFile
from modules.SQLModule.SQLModule import QualityFlag, FileFlag

def setup_logging(filename):
    """Setup the logging configuration for the pipeline"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Setup log file 
    logging.basicConfig(filename=filename, level=logging.INFO,format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

class Level2Pipeline: 

    def __init__(self, config_file : str = '', MIN_OBSID=7000, target_obsid=None) -> None:

        self.MIN_OBSID = MIN_OBSID
        self.target_obsid = target_obsid

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
    
    def process_files(self, filelist: list, parameters: dict, rank: int, base_delay: float) -> None:
        """
        Process a list of files
        """

        delay = rank * base_delay
        time.sleep(delay) 
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
                try:
                    module.run(file)
                except BadCOMAPFile as e:
                    logging.error(f"Error processing file: {file.level1_path}")
                    logging.error(e)
                    db.update_quality_flag_all(file.obsid, FileFlag.BAD_FILE)
                    break
                #except OSError as e:
                #    print(e)
                #    print(module_info)
                #    print(f"Error processing file:",file.level1_path)
                #    print(f'Error processing level 2 file', file.level2_path)
                #    sys.exit() 

    def execute_parallel_mp(self, chunk_filelists: list, parameters: dict, base_delay: float = 2) -> None:
        # Create a pool of workers
        processes = []
        for rank, filelist in enumerate(chunk_filelists):
            p = mp.Process(target=self.process_files, args=(filelist, parameters, rank, base_delay))
            processes.append(p)
            p.start()
        
        # Wait for all processes to finish
        for p in processes:
            p.join()

    def write_filelist(self, filelist: list, directory:str, rank: int) -> None: 
        """write file list to text file"""
        
        with open(f'{directory}/filelist_{rank:02d}.txt', 'w') as f:
            for file in filelist:
                f.write(str(file.obsid) + '\n')

    def write_parameters(self, parameters: dict, directory:str, rank: int) -> None:
        """write parameters dict to a toml file"""

        with open(f'{directory}/parameters_{rank:02d}.toml', 'w') as f:
            toml.dump(parameters, f) 

    def execute_parallel_subprocess(self, chunk_filelists: list, parameters: dict, base_delay: float = 2) -> None:
        # Create a pool of workers
        process_level2_dir = Path(__file__).parent.absolute()
        env = os.environ.copy()
        working_dir = os.getcwd()

        logger = logging.getLogger()
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                logging_file = handler.baseFilename
                print(logging_file)


        processes = []
        for rank, filelist in enumerate(chunk_filelists):
            # Write the filelist to a temporary file
            self.write_filelist(filelist, '/tmp', rank)
            # Write the parameters to a temporary file
            self.write_parameters(parameters, '/tmp', rank)
            # Run the pipeline in a subprocess

            command = ['python', f'{process_level2_dir}/run_level2_pipeline.py', f'/tmp/filelist_{rank:02d}.txt', f'/tmp/parameters_{rank:02d}.toml', str(rank), logging_file]
            #print(' '.join(command))
            #command = ['which','python']
            process = subprocess.Popen(command, shell=False, cwd=working_dir, env=env, 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                text=True)
            processes.append(process)
        for rank, process in enumerate(processes):
            print(f"Checking process {rank}, status: {'Running' if process.poll() is None else 'Finished'}")

            stdout, stderr = process.communicate()
            print(f'stdout, rank:{rank:02d}',stdout)
            print(f'stderr, rank:{rank:02d}',stderr)
            print(f"Checking process {rank}, status: {'Running' if process.poll() is None else 'Finished'}")

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
        if not self.target_obsid:
            level1_filelist = db.get_unprocessed_files(source_group=target_source_group, source=target_source, min_obsid=self.MIN_OBSID, overwrite=True)
        else: 
            obsid = [self.target_obsid]
            fileinfo = db.query_obsid_list(obsid, return_dict=False)[obsid[0]]
            level1_filelist = [fileinfo]

        # Loop over the filelist and process each file
        chunk_filelists = self.split_filelist(level1_filelist, self.parameters['Master']['nprocess'])

        # Run the pipeline in parallel
        self.execute_parallel_subprocess(chunk_filelists, self.parameters)
        
        db.disconnect() 
