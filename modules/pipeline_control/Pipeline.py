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

class BadCOMAPFile(Exception):
    """Exception raised when a COMAP data file cannot be processed due to missing or invalid data"""
    def __init__(self, obsid: int, message: str):
        self.obsid = obsid
        self.message = message
        super().__init__(f"ObsID {obsid}: {message}")

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

def setup_logging(filename):
    """Setup the logging configuration for the pipeline"""
    # Add date prefix to log file name
    date_prefix = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    basename = os.path.basename(filename)
    pathname = os.path.dirname(filename)
    filename_full = os.path.join(pathname, date_prefix + '_' + basename)
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