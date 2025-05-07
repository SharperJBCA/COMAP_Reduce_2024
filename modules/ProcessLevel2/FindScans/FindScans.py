# FindScans.py
#
# Description:
#  Finds the scan edges between each repointing of the telescope. 


import numpy as np
import h5py 
from scipy.interpolate import interp1d
import sys 
from modules.SQLModule.SQLModule import COMAPData 
from modules.ProcessLevel2.SystemTemperature.SystemTemperature import SystemTemperature
from modules.utils.data_handling import read_2bit 

from modules.pipeline_control.Pipeline import BadCOMAPFile, RetryH5PY,BaseCOMAPModule

class FindScans(BaseCOMAPModule):

    def __init__(self, plot : bool = False, plot_dir : str = 'outputs/FindScans', scan_status_code : int = 1, overwrite : bool = False) -> None:
        self.plot = plot
        self.plot_dir = plot_dir
        self.scan_status_code = scan_status_code
        self.overwrite = overwrite

    def already_processed(self, file_info : COMAPData, overwrite : bool = False) -> bool:
        """
        Check if the scan edges have already been found 
        """
        if overwrite:
            return False

        with RetryH5PY(file_info.level2_path,'r') as ds: 
            if 'level2/scan_edges' in ds:
                return True
            return False

    def find_scans(self, file_info : COMAPData) -> None:
        """
        Find the scan edges between each repointing of the telescope
        """

        with RetryH5PY(file_info.level1_path,'r') as data: 
            scan_status = data['hk/antenna0/deTracker/lissajous_status'][...]
            scan_utc    = data['hk/antenna0/deTracker/utc'][...]
            scan_status_interp = interp1d(scan_utc,scan_status,kind='previous',bounds_error=False,
                                        fill_value='extrapolate')(data['spectrometer/MJD'][...])
            
            if file_info.source_group == 'SkyDip':
                features = read_2bit(data['/hk/array/frame/features'][...])  
                features_mjd = data['/hk/array/frame/utc'][...] 
                mjd = data['spectrometer/MJD'][...] 
                features = np.interp(mjd, features_mjd, features)
                features[features < 8] = -1
                scan_edges = SystemTemperature.find_contiguous_blocks(features, 8)
                return scan_edges
            
            if np.sum(scan_status) == 0:
                # For none lissajous scans we use the features to find the scan edges. 
                features = read_2bit(data['spectrometer/features'][...])
                if len(np.unique(features)) <= 1:
                    raise BadCOMAPFile(file_info.obsid, "No valid features found in data file")
                vane_edges = SystemTemperature.find_contiguous_blocks(features, 13)

                # Don't want data before or after the Vane. 
                features[:vane_edges[0][0]] = -1
                if len(vane_edges) == 2:
                    features[vane_edges[1][1]:] = -1

                if 8 in features:
                    scan_edges = SystemTemperature.find_contiguous_blocks(features, 8)
                elif 9 in features:
                    scan_edges = SystemTemperature.find_contiguous_blocks(features, 9)
                elif 10 in features:
                    scan_edges = SystemTemperature.find_contiguous_blocks(features, 10)
                else:
                    scan_edges = None
            elif not any([s == 1 for s in scan_status]):
                # If the lissajous scan never scans (i.e., no status = 1) then skip this file. 
                raise BadCOMAPFile(file_info.obsid, "No valid features found in data file")
            else:
                # We use the lissajous_status to find scan edges for lissajous obs. 
                scans = np.where((scan_status_interp == self.scan_status_code))[0]
                diff_scans = np.diff(scans)
                edges = scans[np.concatenate(([0],np.where((diff_scans > 1))[0], [scans.size-1]))]
                scan_edges_raw = np.array([edges[:-1],edges[1:]]).T
                scan_edges = []
                # Remove first 1000 and last 100 samples to avoid edge effects.
                for start, end in scan_edges_raw: 
                    start += 1000 
                    end -= 100 
                    if (end - start) < 100:
                        continue
                    else:
                        scan_edges.append([start, end])
                scan_edges = np.array(scan_edges)

        if isinstance(scan_edges, type(None)):
            print(file_info.obsid,np.unique(features))
            raise BadCOMAPFile(file_info.obsid, "No valid features found in data file")

        return scan_edges
    
    def save_scans(self, file_info : COMAPData, scan_edges : np.ndarray) -> None:
        """
        Save the scan edges to the database 
        """

        with RetryH5PY(file_info.level2_path,'a') as ds: 
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

        if self.already_processed(file_info, self.overwrite):
            return

        scan_edges = self.find_scans(file_info) 
        self.save_scans(file_info, scan_edges)
