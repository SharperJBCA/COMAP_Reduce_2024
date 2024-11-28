# FindScans.py
#
# Description:
#  Finds the scan edges between each repointing of the telescope. 


import numpy as np
import h5py 
from scipy.interpolate import interp1d
import sys 
from modules.SQLModule.SQLModule import COMAPData, db 

class FindScans:

    def __init__(self, plot : bool = False, plot_dir : str = 'outputs/FindScans', scan_status_code : int = 1) -> None:
        self.plot = plot
        self.plot_dir = plot_dir
        self.scan_status_code = scan_status_code

    def already_processed(self, file_info : COMAPData) -> bool:
        """
        Check if the scan edges have already been found 
        """
        with h5py.File(file_info.level2_path,'r') as ds: 
            if 'level2/scan_edges' in ds:
                return True
            return False

    def find_scans(self, file_info : COMAPData) -> None:
        """
        Find the scan edges between each repointing of the telescope
        """

        with h5py.File(file_info.level1_path,'r') as data: 
            scan_status = data['hk/antenna0/deTracker/lissajous_status'][...]
            scan_utc    = data['hk/antenna0/deTracker/utc'][...]
            scan_status_interp = interp1d(scan_utc,scan_status,kind='previous',bounds_error=False,
                                        fill_value='extrapolate')(data['spectrometer/MJD'][...])
            if np.sum(scan_status) == 0:
                # instead use the feature bits as we probably have 1 scan 
                select = np.where((data.features == 9))[0] 
                scan_edges = np.array([select[0],select[-1]]).reshape(1,2)
            else:
                scans = np.where((scan_status_interp == self.scan_status_code))[0]
                diff_scans = np.diff(scans)
                edges = scans[np.concatenate(([0],np.where((diff_scans > 1))[0], [scans.size-1]))]
                scan_edges = np.array([edges[:-1],edges[1:]]).T

        return scan_edges
    
    def save_scans(self, file_info : COMAPData, scan_edges : np.ndarray) -> None:
        """
        Save the scan edges to the database 
        """

        with h5py.File(file_info.level2_path,'a') as ds: 
            if not 'level2' in ds:
                ds.create_group('level2')
            grp = ds['level2']
            if 'scan_edges' in grp:
                del grp['scan_edges']
            grp.create_dataset('scan_edges',data=scan_edges)

    def run(self, file_info: COMAPData) -> None:
        """
        Find the scan edges between each repointing of the telescope
        """

        if self.already_processed(file_info):
            return

        scan_edges = self.find_scans(file_info) 
        self.save_scans(file_info, scan_edges)
