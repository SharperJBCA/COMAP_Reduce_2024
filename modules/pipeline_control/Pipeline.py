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

def setup_logging(filename):
    """Setup the logging configuration for the pipeline"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Setup log file 
    logging.basicConfig(filename=filename, level=logging.INFO,format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

def run_pipeline(config_file):
    
    # Read in the pipeline configuration file
    parameters = read_parameter_file(config_file) 

    # Setup logging
    setup_logging(parameters['Master']['log_file'])

    # Setup the SQL database
    db.connect(parameters['Master']['sql_database'])

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