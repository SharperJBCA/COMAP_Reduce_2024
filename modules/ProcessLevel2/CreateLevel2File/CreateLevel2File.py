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
import logging 
from modules.pipeline_control.Pipeline import BadCOMAPFile, RetryH5PY,BaseCOMAPModule


class CreateLevel2File(BaseCOMAPModule): 

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
        if (file_info.level2_path is not None): # is it in the database?
            if os.path.exists(file_info.level2_path): # does the file exist?
                with RetryH5PY(file_info.level2_path, 'r') as f: # has the data been copied over?
                    if 'spectrometer' in f:
                        return True
        return False

    def check_file_integrity(self, file_info: COMAPData, delete_if_corrupted: bool = False) -> bool:
        """
        Check of HDF5 file integrity
        """
        if isinstance(file_info.level2_path, type(None)):
            return False 
        if not os.path.exists(file_info.level2_path):
            return False
            
        corrupted = False
        datasets = []
        try:
            with RetryH5PY(file_info.level2_path, 'r') as f:
                def visit_handler(name, obj):
                    # Try to access attributes and data of each object
                    datasets.append(name)
                    if isinstance(obj, h5py.Dataset):
                        try:
                            # Check if we can read the data
                            _ = obj.shape
                            # Try to read a small slice of data
                            if obj.shape[0] > 0:
                                _ = obj[0]
                        except Exception as e:
                            print(f"Error accessing dataset {name}: {str(e)}")
                            nonlocal corrupted
                            corrupted = True
                            return True  # Stop visiting
                    return None
                
                # Visit all objects in the file
                f.visititems(visit_handler)

                
        except Exception as e:
            print(f"Error checking HDF5 file {file_info.level2_path}: {str(e)}")
            corrupted = True

        if not all([f in datasets for f in ['spectrometer','spectrometer/pixel_pointing/pixel_el']]):
            corrupted = True
            print(f"Dataset not found in {file_info.level1_path}")
            print(f"Dataset not found in {file_info.level2_path}: spectrometer/pixel_pointing/pixel_el")

        if corrupted and delete_if_corrupted:
            try:
                os.remove(file_info.level2_path)
                print(f"Deleted corrupted file: {file_info.level2_path}")
            except OSError as e:
                print(f"Failed to delete corrupted file {file_info.level2_path}: {str(e)}")
                
        return corrupted

    def run(self, file_info : COMAPData) -> None:
        """ """

        if self.check_file_integrity(file_info, delete_if_corrupted=True):
            logging.error(f"File was corrupted, deleting: {file_info.level2_path}")

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

        if not os.path.exists(file_info.level1_path):
            raise BadCOMAPFile(file_info.obsid, f"File {file_info.level1_path} not found") 
        
        try: # Test the file can be opened too 
            with RetryH5PY(file_info.level1_path, 'r') as f:
                pass 
        except Exception as e:
            raise BadCOMAPFile(file_info.obsid, f"Error opening {file_info.level1_path}: {str(e)}")

        with RetryH5PY(file_info.level1_path, 'r') as file_level1:
            with RetryH5PY(level2_file, 'w') as file_level2:
                for dset in self.level1_datasets:
                    scale = np.array([float(dset[1])])
                    try:
                        data = file_level1[dset[0]][...]
                    except KeyError:
                        raise BadCOMAPFile(file_info.obsid, f"Dataset {dset[0]} not found in {file_info.level1_path}")
                    data_type = data.dtype
                    if scale != 1: 
                        data *= scale.astype(data_type)
                    file_level2.create_dataset(dset[0], data=data)
                for attr_path in self.level1_attributes:
                    if not attr_path in file_level2:
                        file_level2.create_group(attr_path)
                    for k,v in file_level1[attr_path].attrs.items():
                        file_level2[attr_path].attrs[k] = v

        # Update the SQL database
        file_info.level2_path = level2_file
        db.insert_or_update_data(file_info)

