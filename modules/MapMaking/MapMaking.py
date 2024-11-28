# MapMaking.py
#
#
# What features do I want? 
# 1) I want to be able to specify which files I want to make maps with 
# 2) I want to specify which feeds, bands, and output directory. 
# 3) I want to be able to specify the WCS of the maps 
# 4) I want to loop over many configurations and make many maps automatically
#
# So, probably I want this class to be more like the Processing Level 2 class, and 
#  run a little like a pipeline. 
# 

import logging
import h5py 
import glob 
import importlib
from tqdm import tqdm 
import os 
import sys 

from modules.SQLModule.SQLModule import db, COMAPData
from modules.parameter_files.parameter_files import read_parameter_file

class MapMaking: 

    def __init__(self, config_file='') -> None:
        logging.info("Initializing MapMaking")

        self.config_file = config_file
        if os.path.exists(config_file):
            self.parameters = read_parameter_file(self.config_file) 
        else:
            self.parameters = {} 

    def run(self) -> None:
        logging.info("Running MapMaking") 

        target_source_group = self.parameters['Master'].get('source_group', None)
        target_source       = self.parameters['Master'].get('source'      , None)
        min_obs_id          = self.parameters['Master'].get('min_obsid', 7000)
        max_obs_id          = self.parameters['Master'].get('max_obsid', 100000)

        # Get the data from the database
        query_source_group_list = db.query_source_group_list(target_source_group, min_obsid=min_obs_id, max_obsid=max_obs_id,return_dict=False)
        source_file_list = [f.level2_path for k,f in query_source_group_list.items()]
        for module_info in self.parameters['Master']['_pipeline']:
            package = module_info['package']
            module = module_info['module']
            args = module_info['args']
            # Import the module
            module = getattr(importlib.import_module(package), module)
            module = module(**args)
            # Execute the module
            module.run(source_file_list)