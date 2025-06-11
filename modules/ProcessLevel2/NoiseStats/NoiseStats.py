# NoiseStats.py
#
# Description:
#  Measure 1/f noise and white noise in the data 
#   Works on both level 1 and level 2 data 
#   All the statistics are stored in the level 2 data files  

import h5py 

import itertools
import logging
from tqdm import tqdm 
import numpy as np 
from matplotlib import pyplot 
from scipy.optimize import minimize
import os 
from modules.SQLModule.SQLModule import COMAPData, db
from modules.utils.data_handling import read_2bit
from modules.pipeline_control.Pipeline import RetryH5PY,BaseCOMAPModule, BadCOMAPFile
from modules.utils.median_filter import medfilt

class NoiseStatsLevel1(BaseCOMAPModule):

    def __init__(self, plot=False, output_dir='outputs/NoiseStats') -> None:
        self.NCHANNELS = 1024
        self.NBANDS = 4
        self.NFEEDS = 19
        self.plot = plot
        self.output_dir = output_dir
        self.target_tod_dataset = 'spectrometer/tod'
        os.makedirs(self.output_dir, exist_ok=True)

    def already_processed(self, file_info : COMAPData) -> bool:
        """
        Check if the noise statistics have already been calculated for this file 
        """
        with RetryH5PY(file_info.level2_path, 'r') as f:
            if 'level1_noise_stats' in f:
                return True
            
    def get_nchannels(self, file_info : COMAPData) -> int:
        """
        Get the number of channels in the data 
        """
        with RetryH5PY(file_info.level1_path, 'r') as f:
            return f[self.target_tod_dataset].shape[2] 

    def run(self, file_info : COMAPData) -> None:
        """ """

        if self.already_processed(file_info):
            return 
        


        self.NCHANNELS = self.get_nchannels(file_info)
        statistics = self.fit_statistics(file_info) 
        self.save_statistics(file_info, statistics) 

        if self.plot: 
            self.plot_statistics(file_info, statistics)

    def plot_statistics(self, file_info : COMAPData, statistics : dict) -> None:
        """
        Plot the four statistics as 1D plots vs. frequency for each feed.
        Each stat gets its own panel.  
        """

        with RetryH5PY(file_info.level1_path, 'r') as f:
            frequency = f['spectrometer/frequency'][:].flatten()
            sort_freq = np.argsort(frequency)
            feeds = f['spectrometer/feeds'][:] 

        fig = pyplot.figure(figsize=(36,36))
        axes = fig.subplots(2,2, sharex=True)
        for (ifeed, feed) in enumerate(feeds):
            if feed == 20:  
                continue
            axes[0,0].plot(frequency[sort_freq], statistics['sigma_white'][feed-1, :, :].flatten()[sort_freq], label=f'Feed {feed:02d}')
            axes[0,1].plot(frequency[sort_freq], statistics['sigma_red'][feed-1, :, :].flatten()[sort_freq], label=f'Feed {feed:02d}')
            axes[1,0].plot(frequency[sort_freq], statistics['alpha'][feed-1, :, :].flatten()[sort_freq], label=f'Feed {feed:02d}')
            axes[1,1].plot(frequency[sort_freq], statistics['auto_rms'][feed-1, :, :].flatten()[sort_freq], label=f'Feed {feed:02d}')
        axes[0,0].set_ylabel('White Noise')
        axes[0,1].set_ylabel('Red Noise')
        axes[1,0].set_ylabel('Alpha')
        axes[1,1].set_ylabel('Auto RMS')
        axes[1,0].set_xlabel('Frequency [GHz]')
        axes[1,1].set_xlabel('Frequency [GHz]')
        axes[0,0].legend()
        fig.savefig(os.path.join(self.output_dir, f'{file_info.obsid}_noise_stats.png'))
        pyplot.close(fig)

    def save_statistics(self, file_info : COMAPData, statistics : dict) -> None:
        """ """

        with RetryH5PY(file_info.level2_path, 'a') as f:
            for key in statistics:
                if 'level1_noise_stats' not in f:
                    f.create_group('level1_noise_stats')
                grp = f['level1_noise_stats']
                if key in grp:
                    del grp[key]
                grp.create_dataset(f'{key}', data=statistics[key])

    def auto_rms(self, data : np.ndarray) -> float:
        """ """
        N = len(data)//2 * 2
        rms = np.nanstd(data[:N:2]-data[1:N:2])/np.sqrt(2)
        return rms
    
    def power_spectrum(self, data : np.ndarray) -> tuple:
        """ """
        N = len(data)
        data_norm = data - np.nanmean(data)
        data_norm = data * np.hanning(N)
        Pk = np.fft.fft(data_norm)
        Pk = np.abs(Pk[:N//2])**2
        Pk = Pk / N

        k = np.fft.fftfreq(N, d=1)
        k = k[:N//2] 

        return k[1:], Pk[1:]
    
    def fnoise_fit(self, data : np.ndarray, auto_rms : float, k_ref : float = 0.1) -> tuple:
        """
        Transform data to fourier spectrum and fit model:
        P(k) = sigma_white^2 + sigma_red^2 * (k/k_ref)^alpha
        """

        k, Pk = self.power_spectrum(data)

        def model(k, sigma_white, sigma_red, alpha):
            return (sigma_red/2.)**2 * np.abs(k/k_ref)**alpha + (sigma_white/2.)**2

        def chi2(params):
            sigma_white, sigma_red, alpha = params
            return np.sum((np.log(Pk)- np.log(model(k, sigma_white, sigma_red, alpha)))**2)

        res = minimize(chi2, [auto_rms, 1e-3, -1.0], bounds=[(0, None), (0, None), (-4, 4)])

        return res.x

    def fit_statistics(self, file_info : COMAPData) -> dict:
        """ """

        statistics = {'sigma_white': np.zeros((self.NFEEDS, self.NBANDS, self.NCHANNELS)),
                        'sigma_red': np.zeros((self.NFEEDS, self.NBANDS, self.NCHANNELS)),
                        'alpha': np.zeros((self.NFEEDS, self.NBANDS, self.NCHANNELS)),
                        'auto_rms': np.zeros((self.NFEEDS, self.NBANDS, self.NCHANNELS))}
                                        

        with RetryH5PY(file_info.level1_path,'r') as f:
            feeds = f['spectrometer/feeds'][:]
            features  = read_2bit(f['spectrometer/features'][...])
            data_indices = np.where((features != -1) & (features != 13))[0] 
            if len(data_indices) == 0:
                logging.info(f'No data found in {os.path.basename(file_info.level1_path)} for source {file_info.source}')
                return statistics
            start_index  = data_indices[0]
            end_index    = data_indices[-1]

            for (ifeed, feed) in enumerate(feeds):
                if feed == 20:
                    continue 

                #feed_data = f[self.target_tod_dataset][ifeed, ...]
                feed_data = RetryH5PY.read_dset(f[self.target_tod_dataset], [slice(ifeed, ifeed+1), slice(None), slice(None), slice(None)],lock_file_directory=self.lock_file_path)[0,...]

                indices = itertools.product(
                    range(self.NBANDS),
                    range(self.NCHANNELS)
                )
            
                print(f'Stats for feed {feed}...')
                for iband, ichannel in indices:#, total=self.NBANDS*self.NCHANNELS, desc=f'Feed {feed:02d}'):
                    data = feed_data[iband, ichannel, start_index:end_index]
                    if all(np.isnan(data)):
                        print(f'SKIPPING FEED {feed} BAND {iband} CHANNEL {ichannel}', np.sum(data))
                        continue
                    auto_rms = self.auto_rms(data)
                    sigma_white, sigma_red, alpha = 0,0,0#self.fnoise_fit(data, auto_rms) 

                    statistics['sigma_white'][feed-1, iband, ichannel] = sigma_white
                    statistics['sigma_red'][feed-1, iband, ichannel] = sigma_red
                    statistics['alpha'][feed-1, iband, ichannel] = alpha
                    statistics['auto_rms'][feed-1, iband, ichannel] = auto_rms

        return statistics

class NoiseStatsLevel2(NoiseStatsLevel1):
    """
    
    ###Note to future me###
    The white noise (sigma_white_noise) will be 0.0 if the red noise is 
    so high that the knee frequency is > sample_frequency/2. 
    So, just remember this when assigning flags, 0.0 noise is not a bad fit but it might imply
    bad data!

    """

    def __init__(self, plot=False, output_dir='outputs/NoiseStatsLevel2', 
                 n_channels=2, output_group_name='level2_noise_stats', 
                 target_tod_datasets=['level2/binned_filtered_data'],
                 overwrite=False) -> None:
        self.n_channels = n_channels
        self.NFEEDS = 19
        self.NBANDS = 4 
        self.output_group_name = output_group_name  
        self.target_tod_datasets = target_tod_datasets
        self.plot = plot
        self.output_dir = output_dir
        self.overwrite = overwrite
        os.makedirs(self.output_dir, exist_ok=True)

    def already_processed(self, file_info : COMAPData) -> bool:
        """
        Check if the noise statistics have already been calculated for this file 
        """
        with RetryH5PY(file_info.level2_path, 'r') as f:
            if self.overwrite:
                return False
            if self.output_group_name in f:
                return True
            return False

    def get_nchannels(self, file_info : COMAPData) -> int:
        """
        Get the number of channels in the data 
        """
        with RetryH5PY(file_info.level2_path, 'r') as f:
            return f[self.target_tod_datasets[0]].shape[2] 

    def good_file(self, file_info : COMAPData) -> None:
        """
        Check if the file contains the target datasets 
        """
        with RetryH5PY(file_info.level2_path,'r') as f:
            for dset in self.target_tod_datasets:
                if dset not in f:
                    raise BadCOMAPFile(file_info.obsid, f'{dset} not found in {file_info.level2_path}')

    def run(self, file_info : COMAPData) -> None:
        """ """

        if self.already_processed(file_info) and not self.overwrite:
            return
        
        # Check if the file is processed
        self.good_file(file_info)
        
        self.nchannels = self.get_nchannels(file_info)

        statistics = self.fit_statistics(file_info)
        self.save_statistics(file_info, statistics) 
        #if self.plot: 
        #    self.plot_statistics(file_info)

    def plot_statistics(self, file_info : COMAPData) -> None:
        """
        Plot the four statistics as 1D plots vs. frequency for each feed.
        Each stat gets its own panel.  
        """
        with RetryH5PY(file_info.level2_path, 'r') as f:
            statistics = {key: f[f'{self.output_group_name}/{key}'][...] for key in f[f'{self.output_group_name}'].keys()}
            frequency = f['level2/central_freqs'][0,...].flatten()
            sort_freq = np.argsort(frequency)
            feeds = f['spectrometer/feeds'][:]

        fig = pyplot.figure(figsize=(36,36))
        axes = fig.subplots(2,2, sharex=True)
        for (ifeed, feed) in enumerate(feeds):
            if feed == 20:  
                continue
            if feed > 10:
                ls = '--'
            else:
                ls = '-'
            lw = 3
            axes[0,0].plot(frequency[sort_freq], np.mean(statistics['sigma_white'][feed-1, :, :],axis=-1).flatten()[sort_freq], 
                           label=f'Feed {feed:02d}', ls=ls, lw=lw)
            axes[0,1].plot(frequency[sort_freq], np.mean(statistics['sigma_red'][feed-1, :, :],axis=-1).flatten()[sort_freq], 
                           label=f'Feed {feed:02d}', ls=ls, lw=lw)
            axes[1,0].plot(frequency[sort_freq], np.mean(statistics['alpha'][feed-1, :, :],axis=-1).flatten()[sort_freq], 
                           label=f'Feed {feed:02d}', ls=ls, lw=lw)
            axes[1,1].plot(frequency[sort_freq], np.mean(statistics['auto_rms'][feed-1, :, :],axis=-1).flatten()[sort_freq], 
                           label=f'Feed {feed:02d}', ls=ls, lw=lw)
        axes[0,0].set_ylabel('White Noise')
        axes[0,1].set_ylabel('Red Noise')
        axes[1,0].set_ylabel('Alpha')
        axes[1,1].set_ylabel('Auto RMS')
        axes[1,0].set_xlabel('Frequency [GHz]')
        axes[1,1].set_xlabel('Frequency [GHz]')
        axes[0,0].legend()
        fig.savefig(os.path.join(self.output_dir, f'{file_info.obsid}_noise_stats.png'))
        pyplot.close(fig)

    def save_statistics(self, file_info : COMAPData, all_statistics : dict) -> None:
        """ """

        # Store them in the level 2 file
        print('======')
        with RetryH5PY(file_info.level2_path, 'a') as f:
            for dataset_name, statistics in all_statistics.items():
                dataset_name = dataset_name.split('/')[-1]
                for key in statistics:
                    output_group_name = f'{self.output_group_name}/{dataset_name}'
                    if output_group_name not in f:
                        f.create_group(output_group_name)
                    grp = f[output_group_name]
                    if key in grp:
                        del grp[key]
                    print('saving',output_group_name,key,statistics[key].shape,statistics[key][0,0,0,0])
                    grp.create_dataset(f'{key}', data=statistics[key])

        # BUT WAIT! We also store them in the database 
        key2sql_map = {('binned_data', 'sigma_white'): 'unfiltered_white_noise',
                       ('binned_data', 'sigma_red'): 'unfiltered_red_noise',
                       ('binned_data', 'alpha'): 'unfiltered_noise_index',
                       ('binned_data', 'auto_rms'): 'unfiltered_auto_rms',
                       ('binned_filtered_data', 'sigma_white'): 'filtered_white_noise',
                       ('binned_filtered_data', 'sigma_red'): 'filtered_red_noise',
                       ('binned_filtered_data', 'alpha'): 'filtered_noise_index',
                       ('binned_filtered_data', 'auto_rms'): 'filtered_auto_rms'}
        for dataset_name, statistics in all_statistics.items():
            stats_dict = {} # feed -> band -> stat 

            for key, stat_data in statistics.items():
                for feed in range(1, self.NFEEDS+1):
                    for band in range(self.NBANDS*self.n_channels): 
                        if feed not in stats_dict:
                            stats_dict[feed] = {}
                        if band not in stats_dict[feed]:
                            stats_dict[feed][band] = {}
                        key_name = key2sql_map[(dataset_name.split('/')[-1], key)]
                        iband,ichannel = np.unravel_index(band, (self.NBANDS, self.n_channels))
                        stats_dict[feed][band][key_name] = np.nanmean(stat_data[feed-1, iband,ichannel])
                
            for feed in stats_dict:
                for band in stats_dict[feed]:
                    db.update_quality_statistics(file_info.obsid,
                                                 feed,
                                                 band,
                                                 stats_dict[feed][band])
        

    def auto_rms(self, data : np.ndarray) -> float:
        """ """
        N = len(data)//2 * 2
        rms = np.nanstd(data[:N:2]-data[1:N:2])/np.sqrt(2)
        return rms
    
    def power_spectrum(self, data : np.ndarray, sample_frequency=50.0) -> tuple:
        """ """
        N = len(data)
        data_norm = data - np.nanmedian(data)
        #data_norm = data * np.hanning(N)
        Pk = np.fft.fft(data_norm)
        Pk = np.abs(Pk[:N//2])**2
        Pk = Pk / N

        k = np.fft.fftfreq(N, d=1./sample_frequency)
        k = k[:N//2] 

        # Bin the power spectrum in log-spaced bins 
        N_bins = 100 
        k_bin_edges = np.logspace(np.log10(k[1]), np.log10(k[-1]), N_bins+1)
        k_bin_centers = 0.5 * (k_bin_edges[1:] + k_bin_edges[:-1])
        weights = np.histogram(k, bins=k_bin_edges)[0]
        Pk_binned = np.histogram(k, bins=k_bin_edges, weights=Pk)[0] / weights
        mask = np.isfinite(Pk_binned)

        return k_bin_centers[mask], Pk_binned[mask], weights[mask] #k[1:], Pk[1:]
    
    def model(self, k, sigma_white, sigma_red, alpha, k_ref=0.1):
        return (sigma_red)**2 * np.abs(k/k_ref)**alpha + (sigma_white)**2

    def fnoise_fit(self, data : np.ndarray, auto_rms : float, k_ref : float = 0.1, return_spectra=False) -> tuple:
        """
        Transform data to fourier spectrum and fit model:
        P(k) = sigma_white^2 + sigma_red^2 * (k/k_ref)^alpha
        """

        k, Pk, Pk_weights = self.power_spectrum(data)
        if len(Pk) == 0:
            return (np.nan, np.nan, np.nan)

        ref_amp = np.mean(Pk[np.argmin(np.abs(k-k_ref))])**0.5
        k_idx_high = np.argmin(np.abs(k-k_ref))+3 
        k_idx_low = np.argmin(np.abs(k-k_ref))-3
        def chi2(params):
            sigma_white, sigma_red, alpha = params
            return np.sum((np.log(Pk)- np.log(self.model(k, sigma_white, sigma_red, alpha, k_ref=k_ref)))**2*Pk_weights**0.5)

        res = minimize(chi2, [auto_rms, ref_amp, -1.5], bounds=[(0, None), (0, None), (-4, 4)])

        if return_spectra:
            return res.x, k, Pk
        else:
            return res.x

    def fit_statistics(self, file_info : COMAPData, plot_data=False, save_fig_str=None) -> dict:
        """ """                                        

        all_statistics = {} 
        with RetryH5PY(file_info.level2_path,'r') as f:
            feeds = f['spectrometer/feeds'][:]
            scan_edges = f['level2/scan_edges'][...]
            n_scans = len(scan_edges) 


            for target_tod_dataset in self.target_tod_datasets:
                statistics = {'sigma_white': np.zeros((self.NFEEDS, self.NBANDS, self.n_channels, n_scans)),
                            'sigma_red':   np.zeros((self.NFEEDS, self.NBANDS, self.n_channels, n_scans)),
                            'alpha':       np.zeros((self.NFEEDS, self.NBANDS, self.n_channels, n_scans)),
                            'auto_rms':    np.zeros((self.NFEEDS, self.NBANDS, self.n_channels, n_scans))}

                for (ifeed, feed) in enumerate(feeds):
                    if feed == 20:
                        continue 

                    feed_data = f[target_tod_dataset][feed-1, ...]
                    indices = itertools.product(
                        range(self.n_channels),
                        range(self.NBANDS),
                        enumerate(scan_edges)
                    )
                
                    for ichannel, iband, (iscan, (scan_start, scan_end)) in tqdm(indices, total=self.n_channels*n_scans*self.NBANDS, desc=f'Feed {feed:02d}'):
                        data = feed_data[iband, ichannel, scan_start:scan_end]
                        if all(np.isnan(data)):
                            print(f'SKIPPING FEED {feed} CHANNEL {ichannel}', np.sum(data))
                            continue
                        auto_rms = self.auto_rms(data)
                        data_filtered = data - np.array(medfilt.medfilt(data*1, 50)) 
                        fill_in = (np.abs(data_filtered) < 3*auto_rms) 
                        data[~fill_in] = np.interp(np.where(~fill_in)[0], np.where(fill_in)[0], data[fill_in])
                        sigma_white, sigma_red, alpha = self.fnoise_fit(data, auto_rms) 

                        if plot_data:
                            if (iscan == 0) and (ichannel == 0) and (iband == 0) and ('filtered' in target_tod_dataset):
                                try:
                                    (_, _, _), k_nf, Pk_nf = self.fnoise_fit(f['level2/binned_data'][feed-1, iband, ichannel, scan_start:scan_end], auto_rms, return_spectra=True) 
                                    (sigma_white, sigma_red, alpha), k, Pk = self.fnoise_fit(data, auto_rms, return_spectra=True) 
                                    auto_rms_bad = self.auto_rms(f['level2/binned_data'][feed-1, iband, ichannel, scan_start:scan_end])
                                except TypeError:
                                    continue
                                print('RATIO OF RMS', auto_rms, auto_rms_bad, auto_rms/auto_rms_bad)
                                fig, axes = pyplot.subplots(2, 1, figsize=(12, 8))
                                pyplot.sca(axes[0])
                                pyplot.plot(k, Pk, label=f'Feed {feed} Band {iband} Channel {ichannel}') 
                                pyplot.plot(k_nf, Pk_nf, label=f'No filter')
                                print(f'Target dataset {target_tod_dataset}')
                                print('Sigma white', sigma_white, 'sigma auto', auto_rms)
                                print('NOISE RATIO ', (sigma_white - auto_rms)/sigma_white)   
                                mdl = self.model(k, sigma_white, sigma_red, alpha)
                                pyplot.plot(k, mdl, label=f'Model {feed} Band {iband} Channel {ichannel}')
                                pyplot.axhline(sigma_white**2, color='k', ls='--', label='White Noise')
                                pyplot.axhline(auto_rms**2, color='k', ls='-', label='Auto RMS')
                                pyplot.axhline(sigma_red**2, color='r', ls='--', label='Red Noise')
                                pyplot.axvline(0.1, color='r', ls='--', label='k_ref')

                                pyplot.yscale('log')    
                                pyplot.xscale('log')
                                pyplot.ylim(5e-5,1e1)
                                pyplot.legend(prop={'size': 8})
                                pyplot.sca(axes[1])
                                pyplot.plot(data-np.nanmedian(data)) 
                                spikes = (np.abs(data - np.nanmedian(data)) > 5 * np.nanstd(data))
                                t_temp = np.arange(data.size)
                                #pyplot.plot(data_filtered)
                                pyplot.plot(t_temp[spikes],data[spikes],'rx')
                                if save_fig_str is not None:
                                    pyplot.savefig(save_fig_str.format(feed=feed, band=iband, channel=ichannel,obsid=file_info.obsid))
                                pyplot.show()
                        statistics['sigma_white'][feed-1, iband, ichannel,iscan] = sigma_white
                        statistics['sigma_red']  [feed-1, iband, ichannel,iscan] = sigma_red
                        statistics['alpha']      [feed-1, iband, ichannel,iscan] = alpha
                        statistics['auto_rms']   [feed-1, iband, ichannel,iscan] = auto_rms
                all_statistics[target_tod_dataset] = statistics
                
        return all_statistics
