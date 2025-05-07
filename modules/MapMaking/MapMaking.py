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

def get_file_list(target_source_group=None, target_source=None, min_obs_id=7000, max_obs_id=100000, obsid_list=None):
    # Get the filelist from the database
    if obsid_list is not None:
        query_source_group_list = db.query_obsid_list(obsid_list, 
                                                            return_dict=False)
    else:
        query_source_group_list = db.query_source_group_list(target_source_group, 
                                                            source=target_source, 
                                                            min_obsid=min_obs_id, 
                                                            max_obsid=max_obs_id,
                                                            return_dict=False)
        
    print('Number of files:',len(query_source_group_list))
    source_file_list = [f.level2_path for k,f in query_source_group_list.items() if f.level2_path is not None]
    
    # Check that level2/binned_filtered_data in each file 
    final_files = [] 
    for f in source_file_list:
        with h5py.File(f, 'r') as h5:
            if ('level2/binned_filtered_data' in h5) and ('level2_noise_stats/binned_filtered_data/auto_rms' in h5):
                final_files.append(f) 
    
    return final_files

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

        min_obs_id          = self.parameters['Master'].get('min_obsid', 7000)
        max_obs_id          = self.parameters['Master'].get('max_obsid', 100000)
        obsid_list          = self.parameters['Master'].get('obsid_list', None)


        for module_info in self.parameters['Master']['_pipeline']:


            package = module_info['package']
            module = module_info['module']
            args = module_info['args']
            # Import the module
            module = getattr(importlib.import_module(package), module)
            module = module(**args)

            if isinstance(module.source, list):
                final_files = []
                for src in module.source:
                    final_files+=get_file_list(target_source_group=module.source_group,
                                                    target_source=src,
                                                    min_obs_id=min_obs_id, 
                                                    max_obs_id=max_obs_id, 
                                                    obsid_list=obsid_list)
            else:
                final_files = get_file_list(target_source_group=module.source_group, 
                                                target_source=module.source,
                                                min_obs_id=min_obs_id, 
                                                max_obs_id=max_obs_id, 
                                                obsid_list=obsid_list)
            # Execute the module
            module.run(final_files)


def get_line_file_list(target_source_group=None, target_source=None, min_obs_id=7000, max_obs_id=100000, obsid_list=None):
    # Get the filelist from the database
    if obsid_list is not None:
        query_source_group_list = db.query_obsid_list(obsid_list, 
                                                            return_dict=False)
    else:
        query_source_group_list = db.query_source_group_list(target_source_group, 
                                                            source=target_source, 
                                                            min_obsid=min_obs_id, 
                                                            max_obsid=max_obs_id,
                                                            return_dict=False)
        
    print('Number of files:',len(query_source_group_list))
    source_file_list = [[f.level1_path,f.level2_path] for k,f in query_source_group_list.items() if f.level2_path is not None]
    
    # Check that level2/binned_filtered_data in each file 
    final_files = [] 
    for (level1_file,level2_file) in source_file_list:
        with h5py.File(level2_file, 'r') as h5:
            if ('level2/binned_filtered_data' in h5) and ('level2_noise_stats/binned_filtered_data/auto_rms' in h5) and ('level2/vane/gain' in h5):
                final_files.append([level1_file,level2_file]) 
    
    return final_files

class LineMapMaking: 

    def __init__(self, config_file='') -> None:
        logging.info("Initializing MapMaking")

        self.config_file = config_file
        if os.path.exists(config_file):
            self.parameters = read_parameter_file(self.config_file) 
        else:
            self.parameters = {} 

    def run(self) -> None:
        logging.info("Running MapMaking") 

        min_obs_id          = self.parameters['Master'].get('min_obsid', 7000)
        max_obs_id          = self.parameters['Master'].get('max_obsid', 100000)
        obsid_list          = self.parameters['Master'].get('obsid_list', None)


        for module_info in self.parameters['Master']['_pipeline']:


            package = module_info['package']
            module = module_info['module']
            args = module_info['args']
            # Import the module
            module = getattr(importlib.import_module(package), module)
            module = module(**args)

            if isinstance(module.source, list):
                final_files = []
                for src in module.source:
                    final_files +=get_line_file_list(target_source_group=module.source_group, 
                                                    target_source=src,
                                                    min_obs_id=min_obs_id, 
                                                    max_obs_id=max_obs_id, 
                                                    obsid_list=obsid_list)
            else:
                final_files = get_line_file_list(target_source_group=module.source_group, 
                                                target_source=module.source,
                                                min_obs_id=min_obs_id, 
                                                max_obs_id=max_obs_id, 
                                                obsid_list=obsid_list)

            # Execute the module
            module.run(final_files)