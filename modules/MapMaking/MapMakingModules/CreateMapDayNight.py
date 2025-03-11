# CreateMap.py
#
# Description:
#  CreateMapDayNight splits data into day and night lists based on maximum solar elevation. 

import logging
from astropy.wcs import WCS 
from astropy.io.fits.header import Header
import toml
from modules.MapMaking.MapMakingModules.ReadData import Level2DataReader_Nov2024 
import sys 

from modules.utils.Coordinates import comap_longitude, comap_latitude
from astropy.coordinates import solar_system_ephemeris, EarthLocation, AltAz, get_sun 
from astropy.time import Time, TimezoneInfo 
import astropy.units as u 
import h5py
from datetime import datetime

from pathlib import Path
import subprocess
import os 
class CreateMapDayNight:

    def __init__(self, map_name : str = 'Unknown', band : int = 0, output_dir : str = 'outputs/', 
                 wcs_def : str = None, wcs_def_file : str = None, offset_length : int = 100,
                 feeds : list = [i for i in range(1,20)],
                 sun_elevation_threshold : float = 0.0,
                 tod_data_name : str = 'level2/binned_filtered_data',
                 n_processes : int = 1) -> None:
        logging.info("Initializing CreateMap")

        self.sun_elevation_threshold = sun_elevation_threshold
        self.map_name = map_name
        self.tod_data_name = tod_data_name
        self.band = band
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True) 
        self.config_file = f'{ os.getcwd()}/modules/MapMaking/map_making_configs/config.toml'
        self.parameters = {'band': band,
                      'wcs_def': wcs_def,
                      'offset_length': offset_length,
                      'feeds': feeds,
                      'map_name': map_name,
                      'tod_data_name': tod_data_name,
                      'file_list': f'{ os.getcwd()}/modules/MapMaking/map_making_configs/file_list.txt',
                      'output_filename': f'{map_name}_band{band}_feed{"-".join([str(f) for f in feeds])}.fits',
                      'output_dir': output_dir}
        
        self.config_files = [
            f'{ os.getcwd()}/modules/MapMaking/map_making_configs/config_day.toml',
            f'{ os.getcwd()}/modules/MapMaking/map_making_configs/config_night.toml'
        ]
        self.parameters_day = self.parameters.copy()
        self.parameters_night = self.parameters.copy()
        self.parameters_day['file_list'] = f'{ os.getcwd()}/modules/MapMaking/map_making_configs/file_list_day.txt'
        self.parameters_day['output_filename'] = f'{map_name}_band{band}_feed{"-".join([str(f) for f in feeds])}_day.fits'
        self.parameters_night['file_list'] = f'{ os.getcwd()}/modules/MapMaking/map_making_configs/file_list_night.txt'
        self.parameters_night['output_filename'] = f'{map_name}_band{band}_feed{"-".join([str(f) for f in feeds])}_night.fits'

        self.n_processes = n_processes

    def split_day_night(self, file_list : list) -> tuple:
        """
        Split the file list into day and night lists based on maximum solar elevation. 
        """
        day_list = []
        night_list = [] 

        comap_loc = EarthLocation.from_geodetic(comap_longitude, comap_latitude, 1222.0) 
        for filename in file_list: 
            with h5py.File(filename, 'r') as h: 
                utc_start = h['comap'].attrs['utc_start'] 
                time = Time(datetime.strptime(utc_start, '%Y-%m-%d-%H:%M:%S')) 
                sun = get_sun(time)
                sun_altaz = sun.transform_to(AltAz(obstime=time, location=comap_loc))
                if sun_altaz.alt.deg > self.sun_elevation_threshold:
                    day_list.append(filename)
                else:
                    night_list.append(filename)

        return day_list, night_list

    def create_config_file(self, file_list : list, parameters : dict, config_file : str, feeds : list = [i for i in range(1,20)], output_filename='output.fits') -> None: 
        """
        
        """
        # write dictionary to toml file
        self.parameters['feeds'] = feeds
        self.parameters['output_filename'] = output_filename
        with open(config_file, 'w') as f:
            toml.dump(self.parameters, f) 
        with open(self.parameters['file_list'], 'w') as f:
            for file in file_list:
                f.write(file + '\n')

    def run(self, file_list : list, execute_job=True) -> None:
        logging.info("Running CreateMap")
        mapmaking_dir = Path(__file__).parent.parent.absolute()
        working_dir = os.getcwd()
        feeds = self.parameters['feeds']
        feeds_str = '-'.join([f'{i:02d}' for i in feeds])

        # Split data into day and night lists
        file_list_day, file_list_night = self.split_day_night(file_list)

        self.create_config_file(file_list_day, self.parameters_day, self.config_files[0], feeds=feeds, output_filename=self.parameters_day['output_filename']) 
        self.create_config_file(file_list_night, self.parameters_night, self.config_files[1], feeds=feeds, output_filename=self.parameters_night['output_filename'])
        

        # Run mpirun from the MapMaking directory

        env = os.environ.copy()
        for config_file in self.config_files:
            command = ['mpirun', '-n', str(self.n_processes), 'python', f'{mapmaking_dir}/run_map_making.py', f'{config_file}']
            print(' '.join(command))
            if execute_job:
                result = subprocess.run(command, shell=False, cwd=working_dir, capture_output=True, text=True, env=env)
