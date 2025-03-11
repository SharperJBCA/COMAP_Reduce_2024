# SystemTemperature.py
#
# Description:
#  This module measures the COMAP cal vane events and calculates
#   the system temperature and gain. 
#
#  Uses the hot/cold load equation. 
#       V_hot = G(T_vane + T_sys) 
#       V_cold = G(T_cmb e^-tau + T_atm ( 1 - e^-tau)  + T_sys)
#       Assume that: T_vane = T_atm = 273K 
#                    T_cmb = 2.7K
#       V_hot - V_cold = G(T_atm - T_cmb ) e^-tau 
#       Assume tau ~ 0 so:
#       G = (V_hot - V_cold) / (T_atm - T_cmb) 
#
#       Use Y-factor of Y = V_hot / V_cold = (T_vane + T_sys) / (T_cmb e^-tau + T_atm ( 1 - e^-tau)  + T_sys) 
#       Y = (T_h + T_r )/ (T_c + T_r) 
#       T_r = (T_h - Y T_c )/ (Y - 1)

import numpy as np 
import sys 
from tqdm import tqdm 
import h5py
from modules.utils.data_handling import read_2bit 
from modules.SQLModule.SQLModule import COMAPData

from scipy.interpolate import interp1d
from matplotlib import pyplot 
import os 
from matplotlib.lines import Line2D
import logging

from modules.pipeline_control.Pipeline import BadCOMAPFile

class SystemTemperature: 

    def __init__(self, plot_vane : bool = False, plot_dir : str = '', plot_tsys : bool = True, overwrite=False) -> None:
        self.plot_vane = plot_vane
        self.plot_dir = plot_dir
        self.plot_tsys = plot_tsys

        self.VANE_FEATURE = 13
        self.HK_SPEC_TIME_OFFSET = 0.6

        self.VANE_ANGLE_HIGH_CUTOFF = 70.0 # degrees
        self.VANE_ANGLE_LOW_CUTOFF = 180.0 # degrees

        self.T_c = 2.73 # K 

        self.overwrite = overwrite

    def already_processed(self, file_info : COMAPData, overwrite : bool = False) -> bool:
        """
        Check if the system temperature has already been processed
        """
        if overwrite:
            return False

        if not os.path.exists(file_info.level2_path):
            return False 

        with h5py.File(file_info.level2_path, 'r') as ds: 
            if 'level2/vane' in ds:
                return True 
        return False

    
    def plot_vane_events(self, file_info : COMAPData) -> None:
        """
        Plot the vane data 
        """

        os.makedirs(self.plot_dir, exist_ok=True)

        ds = h5py.File(file_info.level1_path,'r')

        antenna0_mjd = ds['/hk/antenna0/vane/utc'][...] - self.HK_SPEC_TIME_OFFSET/(24*3600.) # MJD 
        vane_angle = ds['/hk/antenna0/vane/angle'][...]/100. # degrees
        features = read_2bit(ds['/hk/array/frame/features'][...])  
        unique_features = np.unique(features)
        cmap = pyplot.get_cmap('tab20')
        color_map = {k: cmap(i/20) for i,k in enumerate(unique_features)}

        spec_mjd = ds['/spectrometer/MJD'][:] 
        spec_tod = ds['/spectrometer/tod'][:,0,100,:] 
        spec_features = interp1d(antenna0_mjd, features, kind='nearest', bounds_error=False, fill_value=-1)(spec_mjd)  
        spec_angle = interp1d(antenna0_mjd, vane_angle, kind='nearest', bounds_error=False, fill_value=-1)(spec_mjd)
        colors = np.array([color_map[x] for x in spec_features])
        time_seconds = (spec_mjd-spec_mjd.min())*24*3600
        vane_edges = self.find_contiguous_blocks(spec_features, self.VANE_FEATURE)

        for i, (idx_start,idx_stop) in enumerate(vane_edges):
            # Plot the vane events
            pyplot.figure(figsize=(12,8))
            pyplot.scatter(time_seconds[idx_start:idx_stop], 
                        spec_tod[0,idx_start:idx_stop], 
                        c=colors[idx_start:idx_stop], cmap='tab20', s=1)
            pyplot.xlabel('Time (seconds)')
            pyplot.ylabel('Antenna Temperature (K)')
            legend_handles = [Line2D([0],[0],marker='o', color=cmap(i/20), label=f'{k}') for i,k in enumerate(unique_features)]
            pyplot.legend(legend_handles, [k for k in unique_features], title='Feature')

            pyplot.savefig(os.path.join(self.plot_dir, f'obs{file_info.obsid:08d}_vane_event{i:02d}.png'))
            pyplot.close()

        # with angle
        feeds = ds['/spectrometer/feeds'][...]  
        for i, (idx_start,idx_stop) in enumerate(vane_edges):
            # Plot the vane events
            pyplot.figure(figsize=(12,8))
            for ifeed, feed in enumerate(feeds):
                if feed == 20:
                    continue
                tod = spec_tod[ifeed, idx_start:idx_stop] 
                tod = (tod - np.min(tod))/(np.max(tod)-np.min(tod))
                spec_angle_tod = spec_angle[idx_start:idx_stop]
                angle_edges = np.linspace(60, spec_angle_tod.max(), 15)
                angle_mids = (angle_edges[1:] + angle_edges[:-1])/2
                angle_bins = np.histogram(spec_angle_tod, bins=angle_edges, weights=tod)[0]/np.histogram(spec_angle_tod, bins=angle_edges)[0]
                angle_sort = np.argsort(spec_angle_tod)
                pyplot.scatter(spec_angle_tod[angle_sort], tod[angle_sort], s=1, c=cmap(ifeed/20),alpha=0.25)
                pyplot.plot(angle_mids, 
                             angle_bins ,label=f'{feed:02d}',lw=3, color=cmap(ifeed/20))
            pyplot.xlabel('Vane Angle (degrees)')
            pyplot.ylabel('Normalised Antenna Temperature')
            pyplot.legend(title='Feed', ncol=2)
            pyplot.axvline(self.VANE_ANGLE_HIGH_CUTOFF, color='k', linestyle='--')
            pyplot.axvline(self.VANE_ANGLE_LOW_CUTOFF, color='k', linestyle='--')
            pyplot.savefig(os.path.join(self.plot_dir, f'obs{file_info.obsid:08d}_angle_vane_event{i:02d}.png'))
            pyplot.close()

        # with angle
        feeds = ds['/spectrometer/feeds'][...]  
        for i, (idx_start,idx_stop) in enumerate(vane_edges):
            # Plot the vane events
            pyplot.figure(figsize=(12,8))
            for ifeed, feed in enumerate(feeds):
                if feed == 20:
                    continue
                tod = spec_tod[ifeed, idx_start:idx_stop] 
                tod = (tod - np.min(tod))/(np.max(tod)-np.min(tod))
                spec_angle_tod = spec_angle[idx_start:idx_stop]
                angle_edges = np.linspace(60, spec_angle_tod.max(), 15)
                angle_mids = (angle_edges[1:] + angle_edges[:-1])/2
                angle_bins = np.histogram(spec_angle_tod, bins=angle_edges, weights=tod)[0]/np.histogram(spec_angle_tod, bins=angle_edges)[0]
                angle_sort = np.argsort(spec_angle_tod)
                pyplot.scatter(spec_angle_tod[angle_sort], tod[angle_sort], s=1, c=cmap(ifeed/20),alpha=0.25)
                pyplot.plot(angle_mids, 
                             angle_bins ,label=f'{feed:02d}',lw=3, color=cmap(ifeed/20))
            pyplot.xlabel('Vane Angle (degrees)')
            pyplot.ylabel('Normalised Antenna Temperature')
            pyplot.legend(title='Feed', ncol=2)
            pyplot.xlim(60, 75)
            pyplot.ylim(0.9,1.1)
            pyplot.axvline(self.VANE_ANGLE_HIGH_CUTOFF, color='k', linestyle='--')
            pyplot.savefig(os.path.join(self.plot_dir, f'obs{file_info.obsid:08d}_angle_zoom_vane_event{i:02d}.png'))
            pyplot.close()

        
            # with angle
        feeds = ds['/spectrometer/feeds'][...]  
        for i, (idx_start,idx_stop) in enumerate(vane_edges):
            # Plot the vane events
            pyplot.figure(figsize=(12,8))
            for ifeed, feed in enumerate(feeds):
                if feed == 20:
                    continue
                tod = spec_tod[ifeed, idx_start:idx_stop] 
                tod = (tod - np.min(tod))/(np.max(tod)-np.min(tod))
                spec_angle_tod = spec_angle[idx_start:idx_stop]
                angle_edges = np.linspace(60, spec_angle_tod.max(), 15)
                angle_mids = (angle_edges[1:] + angle_edges[:-1])/2
                angle_bins = np.histogram(spec_angle_tod, bins=angle_edges, weights=tod)[0]/np.histogram(spec_angle_tod, bins=angle_edges)[0]
                angle_sort = np.argsort(spec_angle_tod)
                pyplot.scatter(spec_angle_tod[angle_sort], tod[angle_sort], s=1, c=cmap(ifeed/20),alpha=0.25)
                pyplot.plot(angle_mids, 
                             angle_bins ,label=f'{feed:02d}',lw=3, color=cmap(ifeed/20))
            pyplot.xlabel('Vane Angle (degrees)')
            pyplot.ylabel('Normalised Antenna Temperature')
            pyplot.legend(title='Feed', ncol=4, loc='lower right')
            pyplot.xlim(160)
            pyplot.ylim(-0.1,0.1)
            pyplot.axvline(self.VANE_ANGLE_LOW_CUTOFF, color='k', linestyle='--')
            pyplot.savefig(os.path.join(self.plot_dir, f'obs{file_info.obsid:08d}_angle_zoom_low_vane_event{i:02d}.png'))
            pyplot.close()

    @staticmethod
    def find_contiguous_blocks(arr, value):
        # Create boolean mask
        mask = (arr == value)
        
        # Add padding to handle edge cases
        padded = np.concatenate(([False], mask, [False]))
        
        # Find edges using diff
        edges = np.flatnonzero(np.diff(padded.astype(int)))
        
        # Pair up start and end indices
        return list(zip(edges[::2], edges[1::2] - 1))
    
    def get_vane_data(self, ds : h5py.File) -> np.ndarray:
        """
        Get the vane data 
        """
        if not '/hk/antenna0/vane/Tshroud' in ds:
            raise BadCOMAPFile(int(ds['comap'].attrs['obsid']), "No vane data found in data file")

        T_h = np.mean(ds['/hk/antenna0/vane/Tshroud'][:])*0.01 + 273.15 # K

        antenna0_mjd = ds['/hk/antenna0/vane/utc'][...] - self.HK_SPEC_TIME_OFFSET/(24*3600.) # MJD 
        vane_angle = ds['/hk/antenna0/vane/angle'][...]/100. # degrees
        features = read_2bit(ds['/hk/array/frame/features'][...])  

        spec_mjd = ds['/spectrometer/MJD'][:] 
        spec_tod = ds['/spectrometer/tod']
        spec_features = interp1d(antenna0_mjd, features, kind='nearest', bounds_error=False, fill_value=-1)(spec_mjd)  
        spec_angle = interp1d(antenna0_mjd, vane_angle, kind='nearest', bounds_error=False, fill_value=-1)(spec_mjd)

        # Get start and stop times of the vane events 
        vane_edges = self.find_contiguous_blocks(spec_features, self.VANE_FEATURE)
        n_vane_events = len(vane_edges)
        if n_vane_events < 1:
            print('No vane events found')
            obsid = int(ds['comap'].attrs['obsid'])
            raise BadCOMAPFile(obsid, "No valid vane events found in data file")

        vane_angles = [] 
        vane_hots   = [] 
        vane_colds  = []
        print('READING VANE DATA')
        for i, (idx_start,idx_stop) in enumerate(tqdm(vane_edges,desc='Vane Events')):
            vane_angles.append(spec_angle[idx_start:idx_stop])
            min_vane = np.nanmin(spec_angle[idx_start:idx_stop])
            tod = spec_tod[...,idx_start:idx_stop] 
            vane_hot_idx = np.where((vane_angles[-1] < (min_vane + 4)))[0]
            vane_hots.append(np.nanmean(tod[...,vane_hot_idx], axis=-1))
            vane_cold_idx = np.where((vane_angles[-1] > self.VANE_ANGLE_LOW_CUTOFF))[0]
            vane_colds.append(np.nanmean(tod[...,vane_cold_idx], axis=-1))
        print('DONE READING VANE DATA')
        return T_h, np.array(vane_hots), np.array(vane_colds)
    
    def calculate_system_temperature(self, T_h : float, vane_hots  : np.ndarray, vane_colds : np.ndarray) -> np.ndarray:
        """ Calculate System Temperature using 
        
        T_r = (T_h - Y T_c) / (Y - 1) 
        Y = V_h/V_c 
        """
        Y = vane_hots / vane_colds
        T_sys = (T_h - Y * self.T_c) / (Y - 1) 
        return T_sys
    
    def calculate_gain(self, T_h : float, vane_hots  : np.ndarray, vane_colds : np.ndarray) -> np.ndarray:
        """ Calculate the gain of the system """
        G = (vane_hots - vane_colds) / (T_h - self.T_c) 
        return G
    
    def save_system_temperature(self, file_info : COMAPData, tsys : np.ndarray, gain : np.ndarray) -> None:
        """ Save the system temperature and gain to the level2 file """

        with h5py.File(file_info.level2_path, 'a') as ds: 
            if not 'level2/vane' in ds:
                ds.create_group('level2/vane')
            grp = ds['level2/vane']
            if 'system_temperature' in grp:
                del grp['system_temperature']
            grp.create_dataset('system_temperature', data=tsys)

            if 'gain' in grp:
                del grp['gain']
            grp.create_dataset('gain', data=gain)

    def plot_tsys_measurements(self, file_info : COMAPData, tsys : np.ndarray, gain : np.ndarray) -> None:
        """ Plot the system temperature measurements """

        os.makedirs(self.plot_dir, exist_ok=True)

        with h5py.File(file_info.level1_path, 'r') as ds: 
            feeds = ds['/spectrometer/feeds'][...] 
            frequencies = ds['/spectrometer/frequency'][...].flatten() 
            freq_idx = np.argsort(frequencies)
            frequencies = frequencies[freq_idx]

        pyplot.figure(figsize=(36,16))
        ptops = []
        pbots = []
        for ifeed, feed in enumerate(feeds):
            if feed == 20:
                continue
            for ievent in range(tsys.shape[0]):
                if ievent == 0:
                    plt = pyplot.plot(frequencies, tsys[ievent,ifeed].flatten()[freq_idx], label=f'{feed:02d}',lw=2)
                else:
                    pyplot.plot(frequencies, tsys[ievent,ifeed].flatten()[freq_idx], color=plt[0].get_color(), alpha=0.5,lw=2)

                ptops.append(np.nanpercentile(tsys[ievent,ifeed].flatten(), 95))
                pbots.append(np.nanpercentile(tsys[ievent,ifeed].flatten(), 5))
        ptop = np.nanmax(ptops)
        pbot = np.nanmin(pbots)
        if np.isnan(ptop) or np.isnan(pbot):
            ptop = 100
            pbot = 0
        pyplot.ylim(pbot, ptop)
        pyplot.xlabel('Frequency (GHz)')
        pyplot.ylabel('Temperature (K)')
        pyplot.xlim(26, 34)
        pyplot.legend(title='Feed',ncol=2) 
        pyplot.grid()
        pyplot.savefig(os.path.join(self.plot_dir, f'obs{file_info.obsid:08d}_tsys.png'))
        pyplot.close()

        pyplot.figure(figsize=(36,16))
        for ifeed, feed in enumerate(feeds):
            if feed == 20:
                continue
            for ievent in range(tsys.shape[0]):
                if ievent == 0:
                    plt = pyplot.plot(frequencies, gain[ievent,ifeed].flatten()[freq_idx], label=f'{feed:02d}',lw=2)
                else:
                    pyplot.plot(frequencies, gain[ievent,ifeed].flatten()[freq_idx], color=plt[0].get_color(), alpha=0.5,lw=2)
        pyplot.xlabel('Frequency (GHz)')
        pyplot.ylabel('Gain')
        pyplot.xlim(26, 34)
        pyplot.legend(title='Feed',ncol=2)
        pyplot.savefig(os.path.join(self.plot_dir, f'obs{file_info.obsid:08d}_gain.png'))
        pyplot.close()

    def measure_system_temperature(self, file_info : COMAPData) -> None:
        """
        Measure the system temperature 
        """

        with h5py.File(file_info.level1_path, 'r') as ds: 
            
            T_h, vane_hots, vane_colds = self.get_vane_data(ds)

            if isinstance(vane_hots, type(None)): # No vane events means we can't calibrate, skip observation. 
                return 
            tsys = self.calculate_system_temperature(T_h, vane_hots, vane_colds) 
            gain = self.calculate_gain(T_h, vane_hots, vane_colds) 

        self.save_system_temperature(file_info, tsys, gain)  

        if self.plot_tsys:
            self.plot_tsys_measurements(file_info, tsys, gain) 


    def run(self, file_info : COMAPData) -> None: 


        if self.already_processed(file_info, self.overwrite):
            return 

        print('PROCESSING SYSTEM TEMPERATURE FOR:', file_info.level1_path) 
        logging.info(f'Processing System Temperature for {file_info.level1_path}')
        self.measure_system_temperature(file_info) 

        if self.plot_vane:
            self.plot_vane_events(file_info) 
