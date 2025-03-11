# CreateMap.py
#
# Description:
#  CreateMap class works with a ReadData class to create COMAP sky maps using 
#   destriping. 

import logging
from astropy.wcs import WCS 
from astropy.io.fits.header import Header
import toml
from modules.MapMaking.MapMakingModules.ReadData import Level2DataReader_Nov2024 
import sys 

from pathlib import Path
import subprocess
import os 

from modules.MapMaking.MapMakingModules.CreateMap import CreateMap

class CreateMapPerFeed(CreateMap):

    def run(self, file_list : list) -> None:
        logging.info("Running CreateMap")
        mapmaking_dir = Path(__file__).parent.parent.absolute()
        working_dir = Path(__file__).parent.parent.parent.parent.absolute() #os.getcwd()
        env = os.environ.copy()
        # Run mpirun from the MapMaking directory
        for feed in range(1,20):
            self.create_config_file(file_list, feeds=[feed], output_filename=f'{self.map_name}_band{self.band:02d}_feed{feed:02d}.fits') 
            command = ['mpirun', '--verbose', '-n', str(self.n_processes), 'python', f'{mapmaking_dir}/run_map_making.py', f'{self.config_file}']
            print(' '.join(command))
            try:
                result = subprocess.run(command, shell=False, cwd=working_dir, capture_output=True, text=True, env=env)
                print("Return code:", result.returncode)
                print("Stdout:", result.stdout)
                print("Stderr:", result.stderr)
            except Exception as e:
                print("Exception:", str(e))