# GainFilter.py
#
# Description:
#  Implements the continuous gain filter 
#   Model: y(t, nu) = dg(t) + dT(t) / Tsys(nu) + alpha(t) / Tsys(nu) (nu - nu_0) / nu_0, nu_0 = 30 GHz

import numpy as np
import numpy.ma as ma
from tqdm import tqdm
import h5py 
from itertools import product 
import sys 
from modules.utils.data_handling import read_2bit
from modules.SQLModule.SQLModule import COMAPData
from modules.utils.linear_solver import AMatrixGainFilter
from matplotlib import pyplot 

from modules.ProcessLevel2.GainFilterAndBin.GainFilters import GainFilterBase, GainFilterWithPrior
from modules.pipeline_control.Pipeline import RetryH5PY,BadCOMAPFile,BaseCOMAPModule

class GainFilterAndBin(BaseCOMAPModule): 

    def __init__(self, end_cut=50, n_freq_bin=2, filtered_binned_data_name='binned_filtered_data', 
                 gain_filter_name='GainFilterBase', overwrite=False, gain_filter_kwargs={}) -> None:
        super().__init__()
        # self.NCHANNELS = 1024
        self.NBANDS = 4
        self.NFEEDS = 19
        self.end_cut = end_cut 
        self.n_freq_bin = n_freq_bin
        self.system_temperature_cutoff = 10
        self.filtered_binned_data_name = filtered_binned_data_name
        self.overwrite = overwrite

        filter_classes = {'GainFilterBase': GainFilterBase,
                          'GainFilterWithPrior': GainFilterWithPrior}
        self.gain_filter = filter_classes.get(gain_filter_name, GainFilterBase)(**gain_filter_kwargs)

    def already_processed(self, file_info : COMAPData, overwrite : bool = False) -> bool:
        """
        Check if the gain filter and bin has already been applied 
        """
        if overwrite:
            return False
    
        with RetryH5PY(file_info.level2_path, 'r') as ds: 
            if f'level2/{self.filtered_binned_data_name}' in ds:
                return True
            return False

    def bin_frequencies(self, data, freqs, weights, n_freq_bin, mask=None):
        """
        Perform weighted averaging of frequencies within each band, accounting for masks
        and calculating frequency statistics.
        
        Parameters:
        -----------
        data : ndarray
            Input data with shape (n_bands, n_freq, n_time)
        freqs : ndarray
            Frequency values with shape (n_bands, n_freq)
        weights : ndarray
            Weights for each channel, shape (n_bands, n_freq) or (n_bands, n_freq, n_time)
        n_freq_bin : int
            Desired number of frequency bins (must be smaller than original n_freq)
        mask : ndarray, optional
            Boolean mask with same shape as data. True indicates masked (invalid) values.
            Can be either (n_bands, n_freq) or (n_bands, n_freq, n_time)
            
        Returns:
        --------
        ndarray
            Binned data with shape (n_bands, n_freq_bin, n_time)
        ndarray
            Boolean mask for binned data
        ndarray
            Central frequencies for each bin (n_bands, n_freq_bin)
        ndarray
            Bandwidth of each bin (n_bands, n_freq_bin)
        """
        n_bands, n_freq, n_time = data.shape
        
        # Check if n_freq is divisible by n_freq_bin
        if n_freq % n_freq_bin != 0:
            raise ValueError(f"n_freq ({n_freq}) must be divisible by n_freq_bin ({n_freq_bin})")
        
        # Calculate bin size
        bin_size = n_freq // n_freq_bin
        
        # Broadcast weights if necessary
        if weights.ndim == 2:
            weights = weights[..., np.newaxis]
        
        # Handle mask
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[..., np.newaxis]
            
            # Apply mask to weights
            weights = ma.array(weights, mask=mask)
        
        # Create masked arrays
        masked_data = ma.array(data, mask=mask if mask is not None else False)
        
        # Reshape data and weights
        reshaped_data = masked_data.reshape(n_bands, n_freq_bin, bin_size, n_time)
        reshaped_weights = weights.reshape(n_bands, n_freq_bin, bin_size, 1)
        
        # Calculate weighted average
        weighted_sum = ma.sum(reshaped_data * reshaped_weights, axis=2)
        weight_sum = ma.sum(reshaped_weights, axis=2)
        binned_data = weighted_sum / weight_sum
        
        # Calculate mask for output (bin is masked if sum of weights is zero or too many input frequencies were masked)
        if mask is not None:
            mask_reshaped = mask.reshape(n_bands, n_freq_bin, bin_size, n_time)
            binned_mask = (np.sum(mask_reshaped, axis=2) > bin_size/2) | (weight_sum == 0)
        else:
            binned_mask = weight_sum == 0
        
        # Calculate frequency statistics for each bin
        freqs_reshaped = freqs.reshape(n_bands, n_freq_bin, bin_size)
        weights_2d = np.mean(weights, axis=-1)  # Average weights over time if time-dependent
        weights_reshaped = weights_2d.reshape(n_bands, n_freq_bin, bin_size)
        
        # Calculate central frequency (weighted average of frequencies in bin)
        central_freqs = np.sum(freqs_reshaped * weights_reshaped, axis=2) / np.sum(weights_reshaped, axis=2)
        
        # Calculate bandwidth (weighted standard deviation of frequencies in bin)
        freq_diff_sq = (freqs_reshaped - central_freqs[..., np.newaxis])**2
        bandwidths = np.sqrt(np.sum(freq_diff_sq * weights_reshaped, axis=2) / np.sum(weights_reshaped, axis=2))
        
        # Fill masked values with NaN
        binned_data = binned_data.filled(np.nan)
        
        return binned_data, binned_mask, central_freqs, bandwidths
    
    def system_temperature_mask(self, system_temperature : np.ndarray) -> np.ndarray:
        """
        Mask system temperature values that are below 10 K
        """
        median_system_temperature = np.nanmedian(system_temperature)
        mask = system_temperature > (median_system_temperature + self.system_temperature_cutoff)

    def process_data(self, data : np.ndarray, 
                     frequencies : np.ndarray, 
                     system_temperature : np.ndarray, 
                     atmos_offsets : np.ndarray, 
                     atmos_tau : np.ndarray, 
                     elevation : np.ndarray,
                    scan_edges : np.ndarray,
                    auto_rms : np.ndarray,
                    feed : int,
                    sigma_red, alpha) -> tuple:
        """
        Expects vane calibrated data and auto_rms values 

        Data should be for a single feed only.

        data - (band, channel, time)
        frequencies - (band, channel)
        system_temperature - (band, channel)
        atmos_offsets - (band, channel, scan)
        atmos_tau - (band, channel, scan)
        elevation - (time)
        scan_edges - [(scan_start, scan_end), scan]
        auto_rms - (band, channel)
        
        """
        median_offsets = []
        # Subtract the best fit atmosphere model 
        for iscan, (scan_start, scan_end) in enumerate(scan_edges):
            median_offsets.append(np.nanmedian(data[...,scan_start:scan_end],axis=-1))
            mdl_atmos = atmos_offsets[...,iscan,np.newaxis] +\
                atmos_tau[...,iscan,np.newaxis]/np.sin(elevation[np.newaxis,np.newaxis,scan_start:scan_end])
            data[...,scan_start:scan_end] -= (atmos_offsets[...,iscan,np.newaxis] +\
                                                atmos_tau[...,iscan,np.newaxis]/np.sin(elevation[np.newaxis,np.newaxis,scan_start:scan_end]))
        
        # Apply the gain filter 
        data_filtered = np.zeros_like(data)
        weights = 1./auto_rms**2

        for iscan,(scan_start, scan_end) in enumerate(scan_edges): 
            data_filtered[...,scan_start:scan_end], mask, gain_temp = self.gain_filter(data[...,scan_start:scan_end], 
                                                                        system_temperature, 
                                                                        median_offsets[iscan],
                                                                        feed,
                                                                        sigma_red=sigma_red,
                                                                        alpha=alpha)
            weights[~mask] = 0.0 
        # Average the data in frequency 
        # n_bands, n_channels, n_tod = data.shape
        # ones = np.ones((n_bands, self.n_freq_bin, n_channels//self.n_freq_bin, 1))
        # binned_data[feed-1] = np.sum(data.reshape(n_bands, self.n_freq_bin, n_channels//self.n_freq_bin,  n_tod) * weights[:,np.newaxis].reshape(n_bands, self.n_freq_bin, n_channels//self.n_freq_bin,  1), axis=2) /\
        #       np.sum(weights[...,np.newaxis].reshape(n_bands, self.n_freq_bin, n_channels//self.n_freq_bin,  1)*ones, axis=2)
        # binned_filtered_data[feed-1] = np.sum(data_filtered.reshape(n_bands, self.n_freq_bin, n_channels//self.n_freq_bin,  n_tod) * weights[:,np.newaxis].reshape(n_bands, self.n_freq_bin, n_channels//self.n_freq_bin,  1), axis=2) /\
        #     np.sum(weights[...,np.newaxis].reshape(n_bands, self.n_freq_bin, n_channels//self.n_freq_bin,  1)*ones, axis=2)

        binned_data, binned_mask, central_freqs, bandwidths = self.bin_frequencies(data, frequencies, weights, self.n_freq_bin) 
        binned_filtered_data, _, _, _ = self.bin_frequencies(data_filtered, frequencies, weights, self.n_freq_bin) 

        return binned_data, binned_filtered_data, central_freqs, bandwidths
    
    def gain_filter_and_bin(self, file_info : COMAPData, skip_feeds = [20]) -> np.ndarray:
        with RetryH5PY(file_info.level2_path,'r') as lvl2:
            with RetryH5PY(file_info.level1_path,'r') as ds: 
                feeds = ds['spectrometer/feeds'][:] 
                frequencies = ds['spectrometer/frequency'][:]
                features  = read_2bit(ds['spectrometer/features'][...])
                scan_edges = lvl2['level2/scan_edges'][...]
                scan_edge_mask = np.zeros(features.shape, dtype=bool)
                for scan_start, scan_end in scan_edges:
                    scan_edge_mask[scan_start:scan_end] = True
                
                n_tod = ds['spectrometer/tod'].shape[-1] 

                binned_data = np.zeros((self.NFEEDS, self.NBANDS, self.n_freq_bin, n_tod))
                binned_filtered_data = np.zeros((self.NFEEDS, self.NBANDS, self.n_freq_bin, n_tod)) 
                central_freqs = np.zeros((self.NFEEDS, self.NBANDS, self.n_freq_bin))
                bandwidths = np.zeros((self.NFEEDS, self.NBANDS, self.n_freq_bin))

                for ifeed, feed in enumerate(tqdm(feeds,desc='Gain Filter and Bin')): 
                    if feed in skip_feeds:
                        continue
                    # First, read in the data we need
                    data = RetryH5PY.read_dset(ds['spectrometer/tod'], [slice(ifeed, ifeed+1), slice(None), slice(None), slice(None)],lock_file_directory=self.lock_file_path)[0,...]
                    system_temperature = lvl2['level2/vane/system_temperature'][0,ifeed,...] 
                    gain = lvl2['level2/vane/gain'][0,ifeed,...]
                    atmos_offsets = lvl2['level2/atmosphere/offsets'][ifeed,...]
                    atmos_tau = lvl2['level2/atmosphere/tau'][ifeed,...] 
                    # Calibrate the data to the vane 
                    auto_rms = lvl2['level1_noise_stats/auto_rms'][feed-1,...]/gain 
                    data = data/gain[...,np.newaxis]
                    if 'level2_noise_stats/binned_data' in lvl2:
                        sigma_red = lvl2['level2_noise_stats/binned_data/sigma_red'][feed-1,...]
                        alpha = lvl2['level2_noise_stats/binned_data/alpha'][feed-1,...]
                    else:
                        sigma_red = None 
                        alpha = None

                    try:
                        elevation = np.radians(lvl2['spectrometer/pixel_pointing/pixel_el'][ifeed,...])
                    except KeyError:
                        raise BadCOMAPFile(file_info.obsid, 'Elevation not found in level 2 file')
                    
                    # Then bin the data 
                    binned_data[feed-1],binned_filtered_data[feed-1],central_freqs[feed-1], bandwidths[feed-1] = self.process_data(data, 
                                                                                                        frequencies, 
                                                                                                        system_temperature, 
                                                                                                        atmos_offsets, 
                                                                                                        atmos_tau, 
                                                                                                        elevation,
                                                                                                        scan_edges,
                                                                                                        auto_rms,
                                                                                                        feed,
                                                                                                        sigma_red,
                                                                                                        alpha)


                     
        return binned_data, binned_filtered_data, central_freqs, bandwidths

    def save_data(self, file_info : COMAPData, binned_data : np.ndarray, binned_filtered_data : np.ndarray, central_freqs : np.ndarray, bandwidths : np.ndarray) -> None:
        with RetryH5PY(file_info.level2_path, 'a') as ds:
            if not 'level2' in ds:
                ds.create_group('level2')
            grp = ds['level2']
            if 'binned_data' in grp:
                del grp['binned_data']
            if self.filtered_binned_data_name in grp:
                del grp[self.filtered_binned_data_name]
            if 'central_freqs' in grp:
                del grp['central_freqs']
            if 'bandwidths' in grp:
                del grp['bandwidths']
            grp.create_dataset('binned_data', data=binned_data)
            grp.create_dataset(self.filtered_binned_data_name, data=binned_filtered_data)
            grp.create_dataset('central_freqs', data=central_freqs)
            grp.create_dataset('bandwidths', data=bandwidths)

    def run(self, file_info : COMAPData) -> None:
        """ """

        if self.already_processed(file_info, self.overwrite):
            return
        
        binned_data, binned_filtered_data, central_freqs, bandwidths = self.gain_filter_and_bin(file_info)
        self.save_data(file_info, binned_data, binned_filtered_data, central_freqs, bandwidths) 

