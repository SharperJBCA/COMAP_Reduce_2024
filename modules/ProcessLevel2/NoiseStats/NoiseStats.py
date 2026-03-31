# NoiseStats.py
#
# Description:
#  Measure 1/f noise and white noise in the data
#   Works on both level 1 and level 2 data
#   All the statistics are stored in the level 2 data files
#
#  Level-1: Only auto_rms is computed (1/f fitting across 1024 channels is
#           prohibitively slow). sigma_white/sigma_red/alpha are not stored.
#  Level-2: Full 1/f noise model fit on binned data (~2 frequency channels).

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
from modules.utils.constants import NFEEDS, NBANDS, NCHANNELS, GROUND_FEED
from modules.utils.median_filter import medfilt

class NoiseStatsLevel1(BaseCOMAPModule):
    """Compute auto_rms for Level-1 (raw, 1024-channel) data.

    Full 1/f noise fitting is not performed at Level-1 because the
    per-channel fit is too slow for 19 feeds x 4 bands x 1024 channels.
    """

    def __init__(self, plot=False, output_dir='outputs/NoiseStats') -> None:
        super().__init__()
        self.NCHANNELS = NCHANNELS
        self.NBANDS = NBANDS
        self.NFEEDS = NFEEDS
        self.plot = plot
        self.output_dir = output_dir
        self.target_tod_dataset = 'spectrometer/tod'
        os.makedirs(self.output_dir, exist_ok=True)

    def get_nchannels(self, file_info : COMAPData) -> int:
        """
        Get the number of channels in the data
        """
        with RetryH5PY(file_info.level1_path, 'r') as f:
            return f[self.target_tod_dataset].shape[2]

    def run(self, file_info : COMAPData) -> None:

        if self.already_processed(file_info, 'level1_noise_stats'):
            return

        self.NCHANNELS = self.get_nchannels(file_info)
        statistics = self.fit_statistics(file_info)
        self.save_statistics(file_info, statistics)

        if self.plot:
            self.plot_statistics(file_info, statistics)

    def plot_statistics(self, file_info : COMAPData, statistics : dict) -> None:
        """
        Plot auto_rms vs. frequency for each feed.
        """

        with RetryH5PY(file_info.level1_path, 'r') as f:
            frequency = f['spectrometer/frequency'][:].flatten()
            sort_freq = np.argsort(frequency)
            feeds = f['spectrometer/feeds'][:]

        fig, ax = pyplot.subplots(figsize=(18, 8))
        for (ifeed, feed) in enumerate(feeds):
            if feed == GROUND_FEED:
                continue
            ax.plot(frequency[sort_freq], statistics['auto_rms'][feed-1, :, :].flatten()[sort_freq], label=f'Feed {feed:02d}')
        ax.set_ylabel('Auto RMS')
        ax.set_xlabel('Frequency [GHz]')
        ax.legend()
        fig.savefig(os.path.join(self.output_dir, f'{file_info.obsid}_noise_stats.png'))
        pyplot.close(fig)

    def save_statistics(self, file_info : COMAPData, statistics : dict) -> None:

        with RetryH5PY(file_info.level2_path, 'a') as f:
            if 'level1_noise_stats' not in f:
                f.create_group('level1_noise_stats')
            grp = f['level1_noise_stats']
            for key in statistics:
                if key in grp:
                    del grp[key]
                grp.create_dataset(key, data=statistics[key])

    @staticmethod
    def auto_rms(data : np.ndarray) -> float:
        """Differenced RMS estimator (insensitive to slow drifts)."""
        N = len(data)//2 * 2
        rms = np.nanstd(data[:N:2]-data[1:N:2])/np.sqrt(2)
        return rms

    def fit_statistics(self, file_info : COMAPData) -> dict:
        """Compute auto_rms per feed/band/channel from Level-1 TOD."""

        statistics = {'auto_rms': np.zeros((self.NFEEDS, self.NBANDS, self.NCHANNELS))}

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
                if feed == GROUND_FEED:
                    continue

                feed_data = RetryH5PY.read_dset(f[self.target_tod_dataset], [slice(ifeed, ifeed+1), slice(None), slice(None), slice(None)])[0,...]

                indices = itertools.product(
                    range(self.NBANDS),
                    range(self.NCHANNELS)
                )

                logging.debug('Computing auto_rms for feed %d', feed)
                for iband, ichannel in indices:
                    data = feed_data[iband, ichannel, start_index:end_index]
                    if np.all(np.isnan(data)):
                        continue
                    statistics['auto_rms'][feed-1, iband, ichannel] = self.auto_rms(data)

        return statistics

class NoiseStatsLevel2(BaseCOMAPModule):
    """Fit 1/f noise model to Level-2 (binned) data.

    Fits the model P(f) = sigma_red^2 * (f/f_ref)^alpha + sigma_white^2
    per feed/band/channel/scan. Results are stored in HDF5 and SQL.

    Note: sigma_white_noise = 0.0 if the red noise is so high that the
    knee frequency > sample_frequency/2.  This is not a bad fit but may
    imply bad data.
    """

    def __init__(self, plot=False, output_dir='outputs/NoiseStatsLevel2',
                 n_channels=2, output_group_name='level2_noise_stats',
                 target_tod_datasets=['level2/binned_filtered_data'],
                 overwrite=False,
                 planck_30_path=None,
                 planck_bright_cut=0.04) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.NFEEDS = NFEEDS
        self.NBANDS = NBANDS
        self.output_group_name = output_group_name
        self.target_tod_datasets = target_tod_datasets
        self.plot = plot
        self.output_dir = output_dir
        self.overwrite = overwrite
        self.planck_bright_cut = planck_bright_cut
        if planck_30_path:
            import healpy as hp
            self.planck_30_map = hp.read_map(planck_30_path, verbose=False)
        else:
            self.planck_30_map = None
        os.makedirs(self.output_dir, exist_ok=True)

    def get_nchannels(self, file_info : COMAPData) -> int:
        with RetryH5PY(file_info.level2_path, 'r') as f:
            return f[self.target_tod_datasets[0]].shape[2]

    def good_file(self, file_info : COMAPData) -> None:
        """Check if the file contains the target datasets."""
        with RetryH5PY(file_info.level2_path,'r') as f:
            for dset in self.target_tod_datasets:
                if dset not in f:
                    raise BadCOMAPFile(file_info.obsid, f'{dset} not found in {file_info.level2_path}')

    def run(self, file_info : COMAPData) -> None:

        if self.already_processed(file_info, self.output_group_name, self.overwrite):
            return

        self.good_file(file_info)
        self.nchannels = self.get_nchannels(file_info)

        statistics = self.fit_statistics(file_info)
        self.save_statistics(file_info, statistics)

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
            if feed == GROUND_FEED:
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

        # Store in HDF5
        with RetryH5PY(file_info.level2_path, 'a') as f:
            for dataset_name, statistics in all_statistics.items():
                dataset_name = dataset_name.split('/')[-1]
                output_group_name = f'{self.output_group_name}/{dataset_name}'
                if output_group_name not in f:
                    f.create_group(output_group_name)
                grp = f[output_group_name]
                for key in statistics:
                    if key in grp:
                        del grp[key]
                    grp.create_dataset(key, data=statistics[key])

        # Store in SQL database
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


    @staticmethod
    def auto_rms(data : np.ndarray) -> float:
        """Differenced RMS estimator (insensitive to slow drifts)."""
        N = len(data)//2 * 2
        rms = np.nanstd(data[:N:2]-data[1:N:2])/np.sqrt(2)
        return rms

    def power_spectrum(self, data : np.ndarray, sample_frequency=50.0) -> tuple:
        """Compute log-binned power spectrum.

        NaN values are replaced with zero after median subtraction so the
        FFT produces valid output even when a few samples are missing.
        """
        N = len(data)
        data_norm = data - np.nanmedian(data)
        # Replace remaining NaN with 0 so FFT doesn't produce all-NaN output
        nan_mask = np.isnan(data_norm)
        if np.any(nan_mask):
            data_norm[nan_mask] = 0.0
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

        return k_bin_centers[mask], Pk_binned[mask], weights[mask]

    def model(self, k, sigma_white, sigma_red, alpha, k_ref=0.1):
        return (sigma_red)**2 * np.abs(k/k_ref)**alpha + (sigma_white)**2

    def fnoise_fit(self, data : np.ndarray, auto_rms : float, k_ref : float = 0.1, return_spectra=False) -> tuple:
        """
        Transform data to fourier spectrum and fit model:
        P(f) = sigma_white^2 + sigma_red^2 * (f/f_ref)^alpha
        """

        k, Pk, Pk_weights = self.power_spectrum(data)
        if len(Pk) == 0:
            return (np.nan, np.nan, np.nan)

        ref_amp = np.mean(Pk[np.argmin(np.abs(k-k_ref))])**0.5
        def chi2(params):
            sigma_white, sigma_red, alpha = params
            mdl = self.model(k, sigma_white, sigma_red, alpha, k_ref=k_ref)
            if np.any(mdl <= 0):
                return 1e30
            return np.sum((np.log(Pk) - np.log(mdl))**2 * Pk_weights**0.5)

        res = minimize(chi2, [auto_rms, ref_amp, -1.5], bounds=[(0, None), (0, None), (-4, 4)])

        if not res.success:
            logging.debug('Noise fit did not converge: %s', res.message)
            result = (np.nan, np.nan, np.nan)
        else:
            result = tuple(res.x)

        if return_spectra:
            return result, k, Pk
        else:
            return result

    def _build_bright_mask(self, f, ifeed, scan_start, scan_end):
        """Return a boolean mask (True = bright source) for TOD samples in a scan.

        Uses the Planck 30 GHz map to identify samples pointing at bright
        regions that would corrupt the noise power-spectrum fit.
        """
        if self.planck_30_map is None:
            return None

        import healpy as hp
        from modules.utils.Coordinates import h2e, e2g, comap_longitude, comap_latitude

        az = f['spectrometer/pixel_pointing/pixel_az'][ifeed, scan_start:scan_end]
        el = f['spectrometer/pixel_pointing/pixel_el'][ifeed, scan_start:scan_end]
        mjd = f['spectrometer/MJD'][scan_start:scan_end]

        ra, dec = h2e(az, el, mjd, comap_longitude, comap_latitude)
        gl, gb = e2g(ra, dec)

        nside = hp.npix2nside(len(self.planck_30_map))
        hpx = hp.ang2pix(nside, np.radians(90.0 - gb), np.radians(gl))
        return self.planck_30_map[hpx] >= self.planck_bright_cut

    def fit_statistics(self, file_info : COMAPData) -> dict:
        """Fit 1/f noise model per feed/band/channel/scan."""

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
                    if feed == GROUND_FEED:
                        continue

                    feed_data = f[target_tod_dataset][feed-1, ...]

                    # Precompute bright-source mask per scan for this feed
                    bright_masks = {}
                    for iscan, (scan_start, scan_end) in enumerate(scan_edges):
                        bright_masks[iscan] = self._build_bright_mask(f, ifeed, scan_start, scan_end)

                    indices = itertools.product(
                        range(self.n_channels),
                        range(self.NBANDS),
                        enumerate(scan_edges)
                    )

                    for ichannel, iband, (iscan, (scan_start, scan_end)) in tqdm(indices, total=self.n_channels*n_scans*self.NBANDS, desc=f'Feed {feed:02d}'):
                        data = feed_data[iband, ichannel, scan_start:scan_end].copy()
                        if np.all(np.isnan(data)):
                            logging.debug('Skipping feed %d band %d channel %d (all NaN)', feed, iband, ichannel)
                            continue

                        # Mask bright-source samples by interpolating over them
                        bright = bright_masks[iscan]
                        if bright is not None and np.any(bright):
                            good = ~bright & np.isfinite(data)
                            if good.sum() > 2:
                                data[bright] = np.interp(
                                    np.where(bright)[0],
                                    np.where(good)[0],
                                    data[good],
                                )
                            else:
                                continue  # almost all samples are bright — skip

                        auto_rms = self.auto_rms(data)
                        data_filtered = data - np.array(medfilt.medfilt(data.copy(), 50))
                        fill_in = (np.abs(data_filtered) < 3*auto_rms)
                        if (np.sum(fill_in)/data.size) < 0.98: # skip if more than 2% of data is bad
                            continue
                        data[~fill_in] = np.interp(np.where(~fill_in)[0], np.where(fill_in)[0], data[fill_in])
                        sigma_white, sigma_red, alpha = self.fnoise_fit(data, auto_rms)

                        statistics['sigma_white'][feed-1, iband, ichannel,iscan] = sigma_white
                        statistics['sigma_red']  [feed-1, iband, ichannel,iscan] = sigma_red
                        statistics['alpha']      [feed-1, iband, ichannel,iscan] = alpha
                        statistics['auto_rms']   [feed-1, iband, ichannel,iscan] = auto_rms
                all_statistics[target_tod_dataset] = statistics

        return all_statistics
