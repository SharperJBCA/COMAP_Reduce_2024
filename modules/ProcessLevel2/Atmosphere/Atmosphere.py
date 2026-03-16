# Atmosphere.py 
#
# Description: 
#
#  Classes for fitting the atmosphere both per observation and per sky dip. 
#

import numpy as np
from scipy.linalg import solve 
from tqdm import tqdm
from itertools import product
from modules.SQLModule.SQLModule import COMAPData
from modules.utils.data_handling import read_2bit
import h5py 
from matplotlib import pyplot 
import os 
import logging
import time
from modules.pipeline_control.Pipeline import RetryH5PY, BaseCOMAPModule

class AtmosphereInObservation(BaseCOMAPModule):
    """
    Class for fitting atmosphere 
    """

    def __init__(self, plot=False, plot_dir='outputs/AtmosphereInObservation', overwrite=False) -> None:
        super().__init__()
        self.NFEEDS = 19
        self.NBANDS = 4
        self.NCHANNELS = 1024

        self.plot = plot
        self.plot_dir = plot_dir 
        self.overwrite = overwrite

        self.target_tod_dataset = 'spectrometer/tod'
    def already_processed(self, file_info : COMAPData, overwrite=False) -> bool:
        """
        Check if the atmosphere has already been fit for this observation 
        """

        if overwrite:
            return False
            
        with RetryH5PY(file_info.level2_path, 'r') as lvl2: 
            if 'level2/atmosphere' in lvl2:
                return True
            return False

    def fit_atmosphere(self, file_info : COMAPData) -> tuple:
        """
        Fit the atmosphere for a single observation 
        """

        with RetryH5PY(file_info.level2_path,'r') as lvl2:
            with RetryH5PY(file_info.level1_path,'r') as ds: 

                feeds = ds['spectrometer/feeds'][:] 
                scan_edges = lvl2['level2/scan_edges'][...]
                gains = lvl2['level2/vane/gain'][0,...]
                print('GAINS SHAPE',gains.shape)

                n_scans = scan_edges.shape[0] 
                atmos_offsets = np.zeros((self.NFEEDS, self.NBANDS, self.NCHANNELS, n_scans))
                atmos_tau     = np.zeros((self.NFEEDS, self.NBANDS, self.NCHANNELS, n_scans))
                print('NUMBER OF SCANS',n_scans)
                for ifeed, feed in enumerate(tqdm(feeds,desc='Fit Atmosphere')): 
                    if feed == 20:
                        continue 

                    
                    feed_data = RetryH5PY.read_dset(ds['spectrometer/tod'], [slice(ifeed, ifeed+1), slice(None), slice(None),slice(None)],lock_file_directory=self.lock_file_path)[0,...]
                    #feed_data = ds[self.target_tod_dataset][ifeed, ...]
                    #elevation = ds['spectrometer/pixel_pointing/pixel_el'][ifeed,...]
                    elevation = RetryH5PY.read_dset(ds['spectrometer/pixel_pointing/pixel_el'], [slice(ifeed, ifeed+1), slice(None)], delay=1,lock_file_directory=self.lock_file_path).flatten()

                    gain = gains[ifeed]#lvl2['level2/vane/gain'][0,ifeed,...]


                    # Reshape arrays to handle all bands and channels at once
                    data = feed_data / gain[...,np.newaxis]

                    indices = product(
                        range(self.NBANDS),
                        range(self.NCHANNELS),
                        enumerate(scan_edges)
                    )
                
                    for iband, ichannel, (iscan, (scan_start, scan_end)) in tqdm(indices, total=self.NBANDS*self.NCHANNELS*n_scans, desc='Fit Atmosphere'):

                        data = feed_data[iband, ichannel, scan_start:scan_end]/gain[iband, ichannel]
                        el   = np.radians(elevation[scan_start:scan_end])
                        # Need to check if the elevation is changing too
                        el_change = np.abs(np.max(el) - np.min(el))

                        if all(np.isnan(data)):
                            continue
                        mask = np.isfinite(data) & np.isfinite(el) 
                        data = data[mask]
                        el = el[mask]
                        if data.size == 0:
                            continue

                        if el_change < np.radians(0.2):
                            atmos_offsets[feed-1,iband,ichannel,iscan] = np.nanmedian(data) # just store offset if no variations. 
                            continue

                        # Fit the atmosphere 
                        A = np.ones((el.size,2))
                        A[:,1] = 1./np.sin(el)
                        try:
                            atmos_offsets[feed-1,iband,ichannel,iscan], atmos_tau[feed-1,iband,ichannel,iscan] = solve(A.T@A, A.T@data)  
                        except np.linalg.LinAlgError:
                            logging.error(f'Failed to fit atmosphere for feed {feed}, band {iband}, channel {ichannel}, scan {iscan}')
                            continue
                    
        return atmos_offsets, atmos_tau

    def plot_atmosphere(self, file_info : COMAPData, atmos_offsets : np.ndarray, atmos_tau : np.ndarray) -> None:
        """ Plot the system temperature measurements """

        os.makedirs(self.plot_dir, exist_ok=True)

        with RetryH5PY(file_info.level1_path, 'r') as ds: 
            feeds = ds['/spectrometer/feeds'][...] 
            frequencies = ds['/spectrometer/frequency'][...].flatten() 
            freq_idx = np.argsort(frequencies)
            frequencies = frequencies[freq_idx]

        pyplot.figure(figsize=(36,16))
        ptops = []
        pbots = []
        atmos_offsets = np.mean(atmos_offsets, axis=3)
        atmos_tau = np.mean(atmos_tau, axis=3)
        for ifeed, feed in enumerate(feeds):
            if feed == 20:
                continue
            plt = pyplot.plot(frequencies, atmos_offsets[ifeed].flatten()[freq_idx], label=f'{feed:02d}',lw=2)

            ptops.append(np.nanpercentile(atmos_offsets[ifeed].flatten(), 95))
            pbots.append(np.nanpercentile(atmos_offsets[ifeed].flatten(), 5))
        ptop = np.nanmax(ptops)
        pbot = np.nanmin(pbots)
        if np.isnan(ptop) or np.isnan(pbot):
            ptop = 100
            pbot = 0
        pyplot.ylim(10, 70)
        pyplot.xlabel('Frequency (GHz)')
        pyplot.ylabel('System Temperature (K)')
        pyplot.xlim(26, 34)
        pyplot.legend(title='Feed',ncol=2) 
        pyplot.grid()
        pyplot.savefig(os.path.join(self.plot_dir, f'obs{file_info.obsid:08d}_atmos_offset.png'))
        pyplot.close()

        pyplot.figure(figsize=(36,16))
        ptops = []
        pbots = []
        for ifeed, feed in enumerate(feeds):
            if feed == 20:
                continue
            plt = pyplot.plot(frequencies, atmos_tau[ifeed].flatten()[freq_idx], label=f'{feed:02d}',lw=2)

            ptops.append(np.nanpercentile(atmos_tau[ifeed].flatten(), 95))
            pbots.append(np.nanpercentile(atmos_tau[ifeed].flatten(), 5))
        ptop = np.nanmax(ptops)
        pbot = np.nanmin(pbots)
        if np.isnan(ptop) or np.isnan(pbot):
            ptop = 100
            pbot = 0
        pyplot.ylim(0,30)
        pyplot.xlabel('Frequency (GHz)')
        pyplot.ylabel(r'Optical Depth * T$_\mathrm{atm}$')
        pyplot.xlim(26, 34)
        pyplot.legend(title='Feed',ncol=2) 
        pyplot.grid()
        pyplot.savefig(os.path.join(self.plot_dir, f'obs{file_info.obsid:08d}_atmos_tau.png'))
        pyplot.close()

    def save_atmosphere(self, file_info : COMAPData, atmos_offsets : np.ndarray, atmos_tau : np.ndarray) -> None:
        """ Save atmosphere measurements to level 2 file """

        with RetryH5PY(file_info.level2_path, 'a') as lvl2: 
            if not 'level2' in lvl2:
                lvl2.create_group('level2')
            grp = lvl2['level2']
            if not 'atmosphere' in grp:
                grp2 = grp.create_group('atmosphere')
            grp2 = grp['atmosphere']
            if 'offsets' in grp2:
                del grp2['offsets']
            ds = grp2.create_dataset('offsets', data=atmos_offsets)
            ds.attrs['unit'] = 'K'
            if 'tau' in grp2:
                del grp2['tau']
            ds = grp2.create_dataset('tau', data=atmos_tau)

            
    def get_nchannels(self, file_info : COMAPData) -> int:
        """
        Get the number of channels in the data 
        """
        with RetryH5PY(file_info.level1_path, 'r') as f:
            return f[self.target_tod_dataset].shape[2] 

    def run(self, file_info : COMAPData) -> None:
        """
        Fit the atmosphere for a single observation 
        """

        if self.already_processed(file_info, self.overwrite):
            return

        logging.info(f'LEVEL1  {file_info.level1_path}')
        logging.info(f'LEVEL2 {file_info.level2_path}')
        self.NCHANNELS = self.get_nchannels(file_info)

        atmos_offsets, atmos_tau = self.fit_atmosphere(file_info) 

        self.save_atmosphere(file_info, atmos_offsets, atmos_tau)
        if self.plot:
            self.plot_atmosphere(file_info, atmos_offsets, atmos_tau)


class AtmosphereFromVane:
    """
    Class for fitting atmosphere 
    """

    def __init__(self, plot=False, plot_dir='outputs/AtmosphereInObservation') -> None:
        self.NFEEDS = 19
        self.NBANDS = 4
        self.NCHANNELS = 1024

        self.plot = plot
        self.plot_dir = plot_dir 

        self.target_tod_dataset = 'spectrometer/tod'

    def already_processed(self, file_info : COMAPData) -> bool:
        """
        Check if the atmosphere has already been fit for this observation 
        """
            
        with RetryH5PY(file_info.level2_path, 'r') as lvl2: 
            if 'level2/atmosphere_vane' in lvl2:
                return True
            return False

    def fit_atmosphere(self, file_info : COMAPData) -> tuple:
        """
        Fit the atmosphere for a single observation 
        """

        with RetryH5PY(file_info.level2_path,'r') as lvl2:
            with RetryH5PY(file_info.level1_path,'r') as ds: 

                feeds = ds['spectrometer/feeds'][:] 
                frequencies = ds['spectrometer/frequency'][:]
                features  = read_2bit(ds['spectrometer/features'][...])
                vane_edges = self.vane_idx_edges(features, 13)

                data_indices = np.where((features != -1) & (features != 13))[0] 
                start_index  = data_indices[0]
                end_index    = data_indices[-1]
                scan_edges = lvl2['level2/scan_edges'][...]
                n_scans = scan_edges.shape[0] 
                atmos_offsets = np.zeros((self.NFEEDS, self.NBANDS, self.NCHANNELS, n_scans))
                atmos_tau     = np.zeros((self.NFEEDS, self.NBANDS, self.NCHANNELS, n_scans))

                for ifeed, feed in enumerate(tqdm(feeds,desc='Fit Atmosphere')): 
                    if feed == 20:
                        continue 

                    feed_data = ds[self.target_tod_dataset][ifeed, ...]
                    elevation = ds['spectrometer/pixel_pointing/pixel_el'][ifeed,...]
                    gain = lvl2['level2/vane/gain'][0,ifeed,...]

                    # Reshape arrays to handle all bands and channels at once
                    data = feed_data / gain[...,np.newaxis]

                    indices = product(
                        range(self.NBANDS),
                        range(self.NCHANNELS),
                        enumerate(scan_edges)
                    )
                
                    for iband, ichannel, (iscan, (scan_start, scan_end)) in tqdm(indices, total=self.NBANDS*self.NCHANNELS*n_scans, desc='Fit Atmosphere'):
                        data = feed_data[iband, ichannel, scan_start:scan_end]/gain[iband, ichannel]
                        el   = np.radians(elevation[scan_start:scan_end])
                        if all(np.isnan(data)):
                            continue
                        mask = ~np.isnan(data)
                        data = data[mask]
                        el = el[mask]
                        # Fit the atmosphere 
                        A = np.ones((el.size,2))
                        A[:,1] = 1./np.sin(el)
                        atmos_offsets[feed-1,iband,ichannel,iscan], atmos_tau[feed-1,iband,ichannel,iscan] = solve(A.T@A, A.T@data)  
                    
        return atmos_offsets, atmos_tau

    def plot_atmosphere(self, file_info : COMAPData, atmos_offsets : np.ndarray, atmos_tau : np.ndarray) -> None:
        """ Plot the system temperature measurements """

        os.makedirs(self.plot_dir, exist_ok=True)

        with RetryH5PY(file_info.level1_path, 'r') as ds: 
            feeds = ds['/spectrometer/feeds'][...] 
            frequencies = ds['/spectrometer/frequency'][...].flatten() 
            freq_idx = np.argsort(frequencies)
            frequencies = frequencies[freq_idx]

        pyplot.figure(figsize=(36,16))
        ptops = []
        pbots = []
        atmos_offsets = np.mean(atmos_offsets, axis=3)
        atmos_tau = np.mean(atmos_tau, axis=3)
        for ifeed, feed in enumerate(feeds):
            if feed == 20:
                continue
            plt = pyplot.plot(frequencies, atmos_offsets[ifeed].flatten()[freq_idx], label=f'{feed:02d}',lw=2)

            ptops.append(np.nanpercentile(atmos_offsets[ifeed].flatten(), 95))
            pbots.append(np.nanpercentile(atmos_offsets[ifeed].flatten(), 5))
        ptop = np.nanmax(ptops)
        pbot = np.nanmin(pbots)
        if np.isnan(ptop) or np.isnan(pbot):
            ptop = 100
            pbot = 0
        pyplot.ylim(10, 70)
        pyplot.xlabel('Frequency (GHz)')
        pyplot.ylabel('System Temperature (K)')
        pyplot.xlim(26, 34)
        pyplot.legend(title='Feed',ncol=2) 
        pyplot.grid()
        pyplot.savefig(os.path.join(self.plot_dir, f'obs{file_info.obsid:08d}_atmos_offset.png'))
        pyplot.close()

        pyplot.figure(figsize=(36,16))
        ptops = []
        pbots = []
        for ifeed, feed in enumerate(feeds):
            if feed == 20:
                continue
            plt = pyplot.plot(frequencies, atmos_tau[ifeed].flatten()[freq_idx], label=f'{feed:02d}',lw=2)

            ptops.append(np.nanpercentile(atmos_tau[ifeed].flatten(), 95))
            pbots.append(np.nanpercentile(atmos_tau[ifeed].flatten(), 5))
        ptop = np.nanmax(ptops)
        pbot = np.nanmin(pbots)
        if np.isnan(ptop) or np.isnan(pbot):
            ptop = 100
            pbot = 0
        pyplot.ylim(0,30)
        pyplot.xlabel('Frequency (GHz)')
        pyplot.ylabel(r'Optical Depth * T$_\mathrm{atm}$')
        pyplot.xlim(26, 34)
        pyplot.legend(title='Feed',ncol=2) 
        pyplot.grid()
        pyplot.savefig(os.path.join(self.plot_dir, f'obs{file_info.obsid:08d}_atmos_tau.png'))
        pyplot.close()

    def save_atmosphere(self, file_info : COMAPData, atmos_offsets : np.ndarray, atmos_tau : np.ndarray) -> None:
        """ Save atmosphere measurements to level 2 file """

        with RetryH5PY(file_info.level2_path, 'a') as lvl2: 
            if not 'level2' in lvl2:
                lvl2.create_group('level2')
            grp = lvl2['level2']
            if not 'atmosphere_vane' in grp:
                grp2 = grp.create_group('atmosphere_vane')
            grp2 = grp['atmosphere']
            if 'offsets' in grp2:
                del grp2['offsets']
            ds = grp2.create_dataset('offsets', data=atmos_offsets)
            ds.attrs['unit'] = 'K'
            if 'tau' in grp2:
                del grp2['tau']
            ds = grp2.create_dataset('tau', data=atmos_tau)

    def get_nchannels(self, file_info : COMAPData) -> int:
        """
        Get the number of channels in the data 
        """
        with RetryH5PY(file_info.level1_path, 'r') as f:
            return f[self.target_tod_dataset].shape[2] 

    def run(self, file_info : COMAPData) -> None:
        """
        Fit the atmosphere for a single observation 
        """

        if self.already_processed(file_info):
            return

        self.NCHANNELS = self.get_nchannels(file_info)

        atmos_offsets, atmos_tau = self.fit_atmosphere(file_info) 

        self.save_atmosphere(file_info, atmos_offsets, atmos_tau)
        if self.plot:
            self.plot_atmosphere(file_info, atmos_offsets, atmos_tau)

    