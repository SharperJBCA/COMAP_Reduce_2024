# Pipeline.py 
#
# Description:
#  The pipeline module is responsible for executing and managing logs associated with
# the COMAP pipeline. 

import os
import sys
import logging
import random
from modules.parameter_files.parameter_files import read_parameter_file
import importlib
from modules.SQLModule.SQLModule import db
from datetime import datetime
import time
import h5py
from modules.SQLModule.SQLModule import COMAPData

class BaseCOMAPModule:

    def __init__(self):
        pass

    def already_processed(self, file_info : COMAPData, ref_group : str, overwrite : bool = False) -> bool:
        """
        Check if a module has already written its output group to the Level-2 file.

        Args:
            file_info: COMAPData object with level2_path
            ref_group: HDF5 group path to check (e.g., 'level2/vane')
            overwrite: If True, always return False (force reprocessing)
        """
        if overwrite:
            return False

        if not os.path.exists(file_info.level2_path or ""):
            return False

        with RetryH5PY(file_info.level2_path, 'r') as ds:
            if ref_group in ds:
                return True
        return False


class BadCOMAPFile(Exception):
    """Exception raised when a COMAP data file cannot be processed due to missing or invalid data"""
    def __init__(self, obsid: int, message: str):
        self.obsid = obsid
        self.message = message
        super().__init__(f"ObsID {obsid}: {message}")



class RetryH5PY:
    """Context manager for retrying to open an HDF5 file that is locked or
    temporarily unavailable (e.g. transient NFS errors).

    Catches BlockingIOError, OSError, and IOError with exponential backoff
    and random jitter to avoid thundering-herd problems when multiple
    subprocess workers hit the same NFS mount.
    """
    def __init__(self, filename, mode='r', max_retries=8, retry_delay=2.0,
                 backoff_factor=2.0, **kwargs):
        self.filename = filename
        self.mode = mode
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
        self.kwargs = kwargs
        self.file = None

    @staticmethod
    def read_dset(dset, sl):
        """Read a slice from an HDF5 dataset."""
        return dset[tuple(sl)]

    def __enter__(self):
        delay = self.retry_delay
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                self.file = h5py.File(self.filename, self.mode, **self.kwargs)
                return self.file
            except (BlockingIOError, OSError, IOError) as e:
                last_exception = e
                if attempt < self.max_retries:
                    jittered = delay * (1 + random.uniform(-0.3, 0.3))
                    logging.warning(
                        "HDF5 open failed (%s). Retry %d/%d in %.1fs: %s",
                        type(e).__name__, attempt + 1, self.max_retries,
                        jittered, self.filename,
                    )
                    time.sleep(jittered)
                    delay *= self.backoff_factor
                else:
                    break

        # All retries exhausted
        try:
            obsid = int(os.path.basename(self.filename).split('-')[1])
        except (IndexError, ValueError):
            obsid = -1
        raise BadCOMAPFile(
            obsid,
            f"Unable to open HDF5 file after {self.max_retries} retries: {last_exception}",
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file is not None:
            self.file.close()



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

def load_module(module_info):
    """Dynamically import and instantiate a pipeline module from a config dict.

    Args:
        module_info: Dict with 'package', 'module', and 'args' keys
                     (as produced by read_parameter_file).

    Returns:
        Instantiated module object ready for .run()
    """
    package = module_info['package']
    module_name = module_info['module']
    args = module_info['args']
    logging.info(f"Loading module: {package}.{module_name}")
    module_cls = getattr(importlib.import_module(package), module_name)
    return module_cls(**args)


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
        module = load_module(module_info)
        module.run()
        modules.append(module)

    db.disconnect() 
    return modules
