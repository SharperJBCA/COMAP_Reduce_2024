# CreateMap.py
#
# Description:
#  CreateMap class works with a ReadData class to create COMAP sky maps using 
#   destriping. 

import logging
from astropy.wcs import WCS 
from astropy.io.fits.header import Header
import toml
from modules.MapMaking.MapMakingModules.ReadData import Level2Data_Nov2024 
import sys 

class CreateMap:

    def __init__(self, map_name : str = 'Unkwown', band : int = 0, output_dir : str = 'outputs/', wcs_def : str = None, wcs_def_file : str = None) -> None:
        logging.info("Initializing CreateMap")

        self.output_dir = output_dir
        if wcs_def is None:
            raise ValueError("wcs_def is required for CreateMap")
        if wcs_def_file is None:
            raise ValueError("wcs_def_file is required for CreateMap")
        self.wcs_def = wcs_def

        wcs_defs = toml.load(wcs_def_file) 

        self.wcs_dict = wcs_defs[wcs_def] 

        self.header = Header(self.wcs_dict) 

        self.data_object = Level2Data_Nov2024(band=band) 

        self.band = band
        self.map_name = map_name

    def run(self, file_list : list) -> None:
        logging.info("Running CreateMap")

        for file in file_list:
            print(file)
            sys.exit()
            self.data_object.read_data(file, band=self.band) 
            self.data_object.setup_wcs(self.header) 

        self.create_map()
        self.write_map() 

    def create_map(self) -> None:
        logging.info("Creating map")

        
