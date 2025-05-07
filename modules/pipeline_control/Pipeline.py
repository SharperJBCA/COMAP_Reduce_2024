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
from datetime import datetime
import time 
import h5py 
import subprocess
import re
from modules.SQLModule.SQLModule import COMAPData

class BaseCOMAPModule: 

    def set_lock_file_path(self,lock_file_path : str = '.lock_file_path') -> None:
        os.makedirs(lock_file_path, exist_ok=True)
        self.lock_file_path = lock_file_path    
        
    def already_processed(self, file_info : COMAPData, ref_group : str, overwrite : bool = False) -> bool:
        """
        Check if the system temperature has already been processed
        """
        if overwrite:
            return False

        if not os.path.exists(file_info.level2_path):
            return False 

        with RetryH5PY(file_info.level2_path, 'r') as ds: 
            if ref_group in ds:
                return True 
        return False


def get_nfs_ops_per_second(mount_point="/scratch/nas_core2"):
    """
    Get the current operations per second for an NFS mount point.
    
    Args:
        mount_point: The NFS mount point to check
        
    Returns:
        ops_per_second: The current operations per second or None if not found
    """
    try:
        # Run nfsiostat to get NFS statistics
        result = subprocess.run(['nfsiostat', mount_point], 
                               capture_output=True, text=True, timeout=5)
        
        # Extract the op/s value using regex
        match = re.search(r'op/s\s+rpc bklog\s+(\d+\.\d+)', result.stdout)
        if match:
            return float(match.group(1))
        
        return None
    except (subprocess.SubprocessError, ValueError, IndexError) as e:
        print(f"Error getting NFS stats: {e}")
        return None


class BadCOMAPFile(Exception):
    """Exception raised when a COMAP data file cannot be processed due to missing or invalid data"""
    def __init__(self, obsid: int, message: str):
        self.obsid = obsid
        self.message = message
        super().__init__(f"ObsID {obsid}: {message}")



class RetryH5PY:
    """Context manager for retrying to open an HDF5 file that is locked.
        Will safely crash if the file cannot be opened after the maximum number of retries.
    """
    def __init__(self, filename, mode='r', max_retries=5, retry_delay=1.0, 
                 backoff_factor=2.0, **kwargs):
        self.filename = filename
        self.mode = mode
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
        self.kwargs = kwargs
        self.file = None
        
    @staticmethod
    def count_files_in_directory(directory):
        result = subprocess.run(' '.join(['ls',directory,'|', 'wc', '-l']), stdout=subprocess.PIPE, shell=True)
        return float(result.stdout.decode('utf-8')) 
    
    @staticmethod
    def read_dset(dset, sl, delay=5,lock_file_directory = '/home/sharper/.COMAP_PIPELINE_LOCK_FILES'):
        
        if lock_file_directory is None:
            return dset[*sl]

        pid = os.getpid()
        lock_filename = f'{lock_file_directory}/{pid}.lock'
        # check if there are lock files in directory 
        max_retries = 100
        retries = 0
        while retries < max_retries:

            lock_file_count = RetryH5PY.count_files_in_directory(f'{lock_file_directory}/')
            if lock_file_count >= 2:
                print(f"Lock file count: {lock_file_count}, waiting...")
                time.sleep(delay)
                retries += 1
            else:
                open(lock_filename, 'w').close()
                data =  dset[*sl]
                os.remove(lock_filename)
                break 
        else:
            print(f"Lock file count: {lock_file_count}, too many lock files, exiting...")
            raise BadCOMAPFile(None, "Too many lock files in directory")

        return data


    def __enter__(self):
        delay = self.retry_delay
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                self.file = h5py.File(self.filename, self.mode, **self.kwargs)
                return self.file
            except BlockingIOError as e:
                last_exception = e
                if attempt < self.max_retries:
                    print(f"HDF5 file locked. Retrying in {delay:.2f}s... (Attempt {attempt+1}/{self.max_retries})")
                    time.sleep(delay)
                    delay *= self.backoff_factor
                else:
                    break
        
        # If we get here, all retries have failed
        obsid = int(os.path.basename(self.filename).split('-')[1])
        raise BadCOMAPFile(obsid, f"Unable to open HDF5 file: {last_exception}")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file is not None:
            self.file.close()


def safe_open_h5(hdf5_dset, sl, max_ops_threshold=500, retry_delay=5, max_retries=100):
    """
    Open an HDF5 file only when NFS operations are below threshold.
    """
    import h5py
    
    mount_point = "/scratch/nas_core2"  # Customize as needed
    
    retries = 0
    while retries < max_retries:
        ops_per_second = get_nfs_ops_per_second(mount_point)
        
        if ops_per_second is None or ops_per_second < max_ops_threshold:
            try:
                return hdf5_dset[*sl]
            except (OSError, IOError) as e:
                # If operation actually fails despite low load, retry
                print(f"Error opening h5 file: {e}, retrying...")
                time.sleep(1)
                retries += 1
                continue
                
        print(f"NFS load too high ({ops_per_second} ops/s > {max_ops_threshold}), " 
              f"waiting {retry_delay}s... (attempt {retries+1}/{max_retries})")
        time.sleep(retry_delay)
        retries += 1



class CustomContextFilter(logging.Filter):
    """Custom filter that adds a custom variable to log records."""
    
    def __init__(self, name=''):
        super().__init__(name)
        self.custom_variable = " "
    
    def set_variable(self, value):
        """Update the custom variable value."""
        self.custom_variable = value
    
    def filter(self, record):
        record.obsid_variable = self.custom_variable
        return True

def setup_logging(filename, prefix_date=True):
    """Setup the logging configuration for the pipeline"""
    # Add date prefix to log file name
    basename = os.path.basename(filename)
    pathname = os.path.dirname(filename)
    if prefix_date:
        date_prefix = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename_full = os.path.join(pathname, date_prefix + '_' + basename)
    else:
        filename_full = filename
    os.makedirs(pathname, exist_ok=True)
    
    # Setup log file
    logging.basicConfig(filename=filename_full, level=logging.INFO,
                        format='%(asctime)s [%(obsid_variable)s] %(message)s', 
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    
    # Create and add our custom filter for the obsids
    context_filter = CustomContextFilter("obsid_filter")  
    logging.getLogger().addFilter(context_filter)

def update_log_variable(value):
    """Update the my_variable value in the logger filter."""
    root_logger = logging.getLogger()
    
    # Find our filter by name or by class
    for filter in root_logger.filters:
        if isinstance(filter, CustomContextFilter):  # Or check filter.name == "obsid_filter"
            filter.set_variable(value)
            break

def run_pipeline(config_file):
    
    # Read in the pipeline configuration file
    parameters = read_parameter_file(config_file) 

    # Setup logging
    setup_logging(parameters['Master']['log_file'])

    # Setup the SQL database
    db.connect(parameters['Master']['sql_database'])

    # Log the jobs in the pipeline
    logging.info('Pipeline jobs:')
    for module_info in parameters['Master']['_pipeline']:
        logging.info(f'{module_info["package"]}.{module_info["module"]}')
    logging.info('')

    # Loop over the Master pipeline 
    modules = []
    for module_info in parameters['Master']['_pipeline']:
        package = module_info['package']
        module = module_info['module']
        args = module_info['args']
        # Import the module
        module = getattr(importlib.import_module(package), module)
        module = module(**args)
        # Execute the module
        module.run()

        modules.append(module) 

    db.disconnect() 
    return modules