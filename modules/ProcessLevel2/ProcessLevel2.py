# Pipeline.py 
#
# Description:
#  The pipeline module is responsible for executing and managing logs associated with
# the COMAP pipeline. 

import os 
import sys
import numpy as np
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
from modules.pipeline_control.Pipeline import RetryH5PY
from tqdm import tqdm

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

        self.target_obsid = self.parameters['Master'].get('target_obsid', None)

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
            print(' '.join(command))
            #command = ['which','python']
            process = subprocess.Popen(command, shell=False, cwd=working_dir, env=env, 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
                                text=True,bufsize=1)
            processes.append(process)

        completed = [False]*len(processes)
        while not all(completed):
            for i, process in enumerate(processes):
                if completed[i]:
                    continue
                    
                # Non-blocking poll to check if process has finished
                return_code = process.poll()
                if return_code is not None:
                    # Process has completed
                    completed[i] = True
                    
                    # # Collect output safely
                    # stdout, stderr = '', ''
                    # try:
                    #     stdout, stderr = process.communicate(timeout=1)
                    # except subprocess.TimeoutExpired:
                    #     # In case communicate somehow still blocks, we'll catch the timeout
                    #     process.kill()
                    #     stdout, stderr = process.communicate()
                    
                    # print(f"Process {i} completed with return code {return_code}")
                    # if stdout.strip():
                    #     print(f"STDOUT from rank {i}:\n{stdout.strip()}")
                    # if stderr.strip():
                    #     print(f"STDERR from rank {i}:\n{stderr.strip()}")
            
            time.sleep(0.1)

        # for rank, process in enumerate(processes):
        #     process.wait()
        #     print(f"Checking process {rank}, status: {'Running' if process.poll() is None else 'Finished'}")

        #     stdout, stderr = process.communicate()
        #     print(f'stdout, rank:{rank:02d}',stdout)
        #     print(f'stderr, rank:{rank:02d}',stderr)
        #     print(f"Checking process {rank}, status: {'Running' if process.poll() is None else 'Finished'}")

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
            print('hello')
            level1_filelist = db.get_unprocessed_files(source_group=target_source_group, source=target_source, min_obsid=self.MIN_OBSID, overwrite=True)
        else: 
            if isinstance(self.target_obsid,str):
                self.target_obsid = np.loadtxt(self.target_obsid, dtype=int).tolist()
            fileinfo = db.query_obsid_list(self.target_obsid, return_dict=False)#[obsid[0]]
            level1_filelist = [fileinfo[obsid] for obsid, v in fileinfo.items()] 

        print(len(level1_filelist), 'files to process')
        
        if self.parameters['Master']['force_run']:
            final_files = level1_filelist
        else:
            final_files = []
            for fileinfo in tqdm(level1_filelist, desc='Checking Level 2 Files for file list'):
                if fileinfo.level2_path is None: # Is the level 2 path in the database?
                    final_files.append(fileinfo)
                else:
                    if not os.path.exists(fileinfo.level2_path): # Does the level 2 file exist?
                        final_files.append(fileinfo)
                    else:
                        with RetryH5PY(fileinfo.level2_path, 'r') as f:  # If the file exists, check if it has been processed
                            if (not 'level2/binned_filtered_data' in f) or (self.parameters['modules']['ProcessLevel2']['GainFilterAndBin']['GainFilterAndBin']['GainFilterAndBin']['overwrite'] == True):
                                final_files.append(fileinfo)
                        
        print('TOTAL FILES', len(final_files), 'files to process')

        print('-----')
        print('Processing Level 2 Files for: {} {}'.format(target_source_group, target_source))
        print('Number of files to process:',len(final_files))
        print('-----')
        print('ObsIDs to process:',[f.obsid for f in final_files])
        # Loop over the filelist and process each file
        chunk_filelists = self.split_filelist(final_files, self.parameters['Master']['nprocess'])


        for rank, filelist in enumerate(chunk_filelists):
            print(f'Rank: {rank}, Number of files: {len(filelist)}')
        
         
        # Add master pid to parameters 
        # Clear lock files 
        lock_files_folder = f'/home/sharper/.COMAP_PIPELINE_LOCK_FILES/{os.getpid()}'
        self.parameters['Master']['lock_files_folder'] = lock_files_folder
        os.makedirs(lock_files_folder, exist_ok=True)
        for filename in os.listdir(lock_files_folder):
            os.remove(os.path.join(lock_files_folder, filename))

        # Run the pipeline in parallel
        self.execute_parallel_subprocess(chunk_filelists, self.parameters)
        
        db.disconnect() 
