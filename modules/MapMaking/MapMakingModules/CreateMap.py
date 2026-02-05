# CreateMap.py
#
# Description:
#  CreateMap class works with a ReadData class to create COMAP sky maps using 
#   destriping. 

import logging
from astropy.wcs import WCS 
from astropy.io.fits.header import Header
import toml
import sys 

from pathlib import Path
import subprocess
import os 
class CreateMap:

    def __init__(self, map_name : str = 'Unknown', band : int = 0, output_dir : str = 'outputs/', 
                 source : str = 'Unknown', source_group : str = 'Unknown',
                 wcs_def : str = None, wcs_def_file : str = None, offset_length : int = 100,
                 feeds : list = [i for i in range(1,20)],
                 tod_data_name : str = 'level2/binned_filtered_data',
                 database_file : str = 'databases/COMAP_manchester.db',
                 file_list_name : str = 'file_list.txt',
                 sigma_red_cutoff : float = 0.4,
                 calib_path : str = None,
                 planck_30_path : str = None, 
                 lambda_ridge : float = 1e-4, 
                 jackknife_odd_even : str = None,
                 n_processes : int = 1,**kwargs) -> None:
        logging.info("Initializing CreateMap")

        logger = logging.getLogger()
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                logging_file = handler.baseFilename
                print(logging_file)

        self.file_list_name = file_list_name
        self.map_name = map_name
        self.tod_data_name = tod_data_name
        self.band = band
        self.output_dir = output_dir
        self.source = source    
        self.source_group = source_group
        os.makedirs(self.output_dir, exist_ok=True) 
        self.config_file = f'{ os.getcwd()}/modules/MapMaking/map_making_configs/config.toml'
        self.parameters = {'band': band,
                      'wcs_def': wcs_def,
                      'source': source,
                        'source_group': source_group,
                      'offset_length': offset_length,
                      'feeds': feeds,
                      'map_name': map_name,
                      'tod_data_name': tod_data_name,
                      'log_file_name':logging_file,
                      'sigma_red_cutoff': sigma_red_cutoff,
                      'database':f'{ os.getcwd()}/{database_file}',
                      'file_list': f'{ os.getcwd()}/modules/MapMaking/map_making_configs/{file_list_name}',
                      'output_filename': f'{map_name}_band{band}_feed{"-".join([str(f) for f in feeds])}.fits',
                      'output_dir': output_dir,
                      'lambda_ridge': lambda_ridge,
                      'planck_30_path': planck_30_path,
                      'calib_path': calib_path,
                      'jackknife_odd_even':jackknife_odd_even
        }
        self.n_processes = n_processes

    def create_config_file(self, file_list : list, feeds : list = [i for i in range(1,20)], output_filename='output.fits') -> None: 
        """
        
        """
        # write dictionary to toml file
        self.parameters['feeds'] = feeds
        self.parameters['output_filename'] = output_filename
        with open(self.config_file, 'w') as f:
            toml.dump(self.parameters, f) 
        with open(self.parameters['file_list'], 'w') as f:
            for file in file_list:
                f.write(file + '\n')

        logging.info(f'Wrote config file to {self.config_file}')
        # log the config file
        for k, v in self.parameters.items():
            logging.info(f'{k}: {v}')

    def run(self, file_list : list) -> None:
        logging.info("Running CreateMap")
        mapmaking_dir = Path(__file__).parent.parent.absolute()
        working_dir = os.getcwd()
        feeds = self.parameters['feeds']
        feeds_str = '-'.join([f'{i:02d}' for i in feeds])
        self.create_config_file(file_list, feeds=feeds, output_filename=f'{self.map_name}_band{self.band:02d}_feed{feeds_str}.fits') 
        

        # Run mpirun from the MapMaking directory

        env = os.environ.copy()
        command = ['mpirun', '-n', str(self.n_processes), 'python', f'{mapmaking_dir}/run_map_making.py', f'{self.config_file}']
        print(' '.join(command))
        #subprocess.run(' '.join(command), shell=True, cwd=working_dir)
        file_list_str = 'CREATEMAP: MAPPING {n_files} for {self.source} band {self.band}'.format(n_files=len(file_list), self=self)
        print(file_list_str)
        logging.info(file_list_str)
        try:
            result = subprocess.run(command, shell=False, cwd=working_dir, capture_output=True, text=True, env=env)
            print("Return code:", result.returncode)
            print("Stdout:", result.stdout)
            print("Stderr:", result.stderr)
        except Exception as e:
            print("Exception:", str(e))