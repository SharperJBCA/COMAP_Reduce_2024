import os 
import sys 
import numpy as np
import h5py 
from datetime import datetime
from modules.SQLModule.SQLModule import COMAPData, db
import logging 
from modules.pipeline_control.Pipeline import BadCOMAPFile

class CheckLevel1FileExists:

    def __init__(self, **kwargs):
        pass 

    def run(self, file_info : COMAPData) -> None:
        print('HELLO!')
        print('Does the level 1 file exist?', os.path.exists(file_info.level1_path))
        if not os.path.exists(file_info.level1_path):
            raise BadCOMAPFile(file_info.obsid, f"File {file_info.level1_path} not found")
        
        print('Can the file be opened?')
        try: # Test the file can be opened too 
            with h5py.File(file_info.level1_path, 'r') as f:
                pass
            print('Yes')
        except Exception as e:
            print('No')
            raise BadCOMAPFile(file_info.obsid, f"Error opening {file_info.level1_path}: {str(e)}")
