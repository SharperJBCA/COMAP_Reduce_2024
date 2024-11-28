# CreateLevel2File.py
#
# Description: 
#  Creates the level 2 file and stores the location in the SQL database. 
#   
import os 
import sys 
import numpy as np
import h5py 
from datetime import datetime
from modules.SQLModule.SQLModule import COMAPData, db

class CreateLevel2File: 

    def __init__(self, output_dir : str = '', save_dir_structure : str = './',
                  level1_datasets : list = [], level1_attributes : list = []) -> None:
        self.output_dir = output_dir 
        self.level1_datasets = level1_datasets # Format = [[dset_name, dset_scale], ...]
        self.level1_attributes = level1_attributes

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

    def already_processed(self, file_info : COMAPData) -> bool:
        """
        Check if the level 2 file has already been created 
        """
        if file_info.level2_path is not None:
            return True
        return False

    def run(self, file_info : COMAPData) -> None:
        """ """

        if self.already_processed(file_info):
            return

        # Create the level 2 file
        base_filename = os.path.basename(file_info.level1_path)

        # Format the output directory 
        date = datetime.strptime(file_info.utc_start, '%Y-%m-%d-%H:%M:%S')
        output_dir = os.path.join(self.output_dir, '{}/{}/level2'.format(date.strftime('%Y-%m'), file_info.source))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        level2_file = os.path.join(output_dir, f'Level2_{base_filename}')

        with h5py.File(file_info.level1_path, 'r') as file_level1:
            with h5py.File(level2_file, 'w') as file_level2:
                for dset in self.level1_datasets:
                    scale = np.array([float(dset[1])])
                    data = file_level1[dset[0]][...]
                    data_type = data.dtype
                    if scale != 1: 
                        data *= scale.astype(data_type)
                    file_level2.create_dataset(dset[0], data=data)
                for attr_path in self.level1_attributes:
                    if not attr_path in file_level2:
                        file_level2.create_group(attr_path)
                    for k,v in file_level1[attr_path].attrs.items():
                        file_level2.attrs[k] = v

        # Update the SQL database
        file_info.level2_path = level2_file
        db.insert_or_update_data_comapdata(file_info)

