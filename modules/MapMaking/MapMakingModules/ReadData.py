# ReadData.py
# 
# Description:
#  Read in the level 2 files and prepare them for map making.
#  Classes are designed to work with specific level 2 file formats. 


import logging 
from dataclasses import dataclass, field
import numpy as np 
import h5py 
import sys 
import os
from modules.SQLModule.SQLModule import db
from astropy.wcs import WCS
import healpy as hp 
import toml 
from astropy.io.fits.header import Header
import sys 
from matplotlib import pyplot 
from tqdm import tqdm

from astropy.time import Time
from datetime import datetime   

from modules.utils import bin_funcs
from modules.utils.median_filter import medfilt
from modules.utils.Coordinates import h2e_full, comap_longitude, comap_latitude

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
def sum_map_all_inplace(m): 
    """Sum array elements over all MPI processes"""
    if m.dtype == np.float64:
        mpi_type = MPI.DOUBLE
    else:
        mpi_type = MPI.FLOAT
    comm.Allreduce(MPI.IN_PLACE,
        [m, mpi_type],
        op=MPI.SUM
        )
    return m 

BAD_FEEDS = [2, 3, 10]

@dataclass
class Level2Data_Nov2024: 
    offset_length : int = 100

    sum_map    : np.ndarray = field(default_factory=np.empty(0,dtype=np.float32)) 
    weight_map : np.ndarray = field(default_factory=np.empty(0,dtype=np.float32))  
    sky_map    : np.ndarray = field(default_factory=np.empty(0,dtype=np.float32))
    hits_map   : np.ndarray = field(default_factory=np.empty(0,dtype=np.float32))
    pixels     : np.ndarray = field(default_factory=np.empty(0,dtype=np.int64)) 
    weights    : np.ndarray = field(default_factory=np.empty(0,dtype=np.float32))
    rhs        : np.ndarray = field(default_factory=np.empty(0,dtype=np.float32))

###
# QUICK CALIBRATION MODEL
# Fitted taua_model_func to ratios of model TauA fluxes over observed fluxes for each feed/channel/band
# saved npy file to : notebooks/source_fitting/TauA_best_fits.npy
# best_period = 365.25 
# best_phase = ~100 days this is fixed across all ratios.
###
def taua_model_func(times, phase, period, amplitude, grad, offset):
    model = amplitude*np.sin(2*np.pi*(times - 59000 - phase)/period ) + grad*(times-59000)/365.25 + offset
    return model


class Level2DataReader_Nov2024: 

    def __init__(self, tod_data_name : str = 'level2/binned_filtered_data', offset_length : int = 100, band_index=0, channel_shape=(4,2),
                 sigma_red_cutoff=0.4) -> None:
        self.tod_data_name = tod_data_name


        self.data_container = Level2Data_Nov2024
        self.data_container.offset_length = offset_length
        self.data_container.rhs        = np.empty(0, dtype=np.float32)
        self.data_container.weights    = np.empty(0, dtype=np.float32)
        self.data_container.pixels     = np.empty(0, dtype=np.int64)
        self.data_container.sum_map    = np.empty(0, dtype=np.float32)
        self.data_container.weight_map = np.empty(0, dtype=np.float32)
        self.data_container.hits_map   = np.empty(0, dtype=np.float32)
        self.data_container.sky_map    = np.empty(0, dtype=np.float32)


        self.data_container.tod = np.empty(0, dtype=np.float32)


        self.input_coord  = 'C' 
        self.output_coord = 'G' 
        self.feeds = []
        self.band_index = band_index
        self.band, self.channel = np.unravel_index(self.band_index, channel_shape)

        self.sigma_red_cutoff = sigma_red_cutoff

        self._n_tod  = 0
        self.n_tod_per_file = []

        self._last_pixels_index = 0 
        self._last_rhs_index    = 0

        self.planck_30_map = hp.read_map('/scratch/nas_cbassarc/sharper/work/CBASS_PolarisedTF/ancillary_data/planck_pr3/LFI_SkyMap_030-BPassCorrected_1024_R3.00_full.fits',verbose=False)

        self.taua_calibration_factors = np.load('/scratch/nas_cbassarc/sharper/work/COMAP/COMAP_Reduce_2024/notebooks/source_fitting/TauA_best_fits.npy', allow_pickle=True).flatten()[0]
    @property
    def sum_map(self) -> np.ndarray:
        return self.data_container.sum_map
    @sum_map.setter
    def sum_map(self, value: np.ndarray) -> None:
        self.data_container.sum_map = value
    
    @property
    def weight_map(self) -> np.ndarray:
        return self.data_container.weight_map
    @weight_map.setter
    def weight_map(self, value: np.ndarray) -> None:
        self.data_container.weight_map = value

    @property
    def sky_map(self) -> np.ndarray:
        return self.data_container.sky_map
    @sky_map.setter
    def sky_map(self, value: np.ndarray) -> None:
        self.data_container.sky_map = value

    @property
    def hits_map(self) -> np.ndarray:
        return self.data_container.hits_map
    @hits_map.setter
    def hits_map(self, value: np.ndarray) -> None:
        self.data_container.hits_map = value

    @property
    def pixels(self) -> np.ndarray:
        return self.data_container.pixels
    @pixels.setter
    def pixels(self, value: np.ndarray) -> None:
        self.data_container.pixels = value

    @property
    def weights(self) -> np.ndarray:
        return self.data_container.weights
    @weights.setter
    def weights(self, value: np.ndarray) -> None:
        self.data_container.weights = value

    @property
    def rhs(self) -> np.ndarray:
        return self.data_container.rhs
    @rhs.setter
    def rhs(self, value: np.ndarray) -> None:
        self.data_container.rhs = value

    @property
    def offset_length(self) -> int:
        return self.data_container.offset_length
    @offset_length.setter
    def offset_length(self, value: int) -> None:
        self.data_container.offset_length = value

    @property
    def n_offsets(self) -> int:
        return len(self.data_container.rhs)

    def setup_wcs(self, wcs_def, config_file='map_making_configs/wcs_definitions.toml') -> None:
        """
        Setup the WCS of the maps
        """
        self.wcs_def = wcs_def
        with open(config_file, 'r') as f:
            config = toml.load(f)

        header = Header(config[wcs_def])

        self.header = header 
        self.wcs = WCS(header) 

        self.data_container.sum_map    = np.zeros((self.header['NAXIS2']*self.header['NAXIS1']), dtype=np.float32)
        self.data_container.weight_map = np.zeros((self.header['NAXIS2']*self.header['NAXIS1']), dtype=np.float32)
        self.data_container.hits_map   = np.zeros((self.header['NAXIS2']*self.header['NAXIS1']), dtype=np.float32)

    def read_data(self, file: str,  band : int, channel : int, database, use_flags : bool = False, bad_feeds : list = []) -> tuple:


        with h5py.File(file, 'r') as f:
            obsid = int(os.path.basename(file).split('-')[1])
            feeds = f['spectrometer/feeds'][:-1] 
            scan_edges = f['level2/scan_edges'][...] 
            tod = []
            weights = []
            x   = [] 
            y   = [] 
            mjd0 = f['spectrometer/MJD'][0]

            if database: 
                observer_flags = db.get_quality_flags(obsid)

            for ifeed, feed in enumerate(feeds): 
                if feed in bad_feeds:
                    continue

                # Use feed 1 as a reference for the noise, if any scan exceeds the threshold skip the whole observation
                # Idea here: 
                #  If feed 1 is bad in any scan then probably the PWV is high
                #  sometimes we can get lucky and some scans seem good, but probably the noise is still too high and
                #  we should just skip the whole obsid. 
                sigma_reds_feed1 = f['level2_noise_stats/binned_filtered_data/sigma_red'][0, band, channel, :] 
                if any([sigma_red > 5.0 for sigma_red in sigma_reds_feed1]):
                    print('SKIP')
                    break 

                # if database: 
                #     if (feed, band) in observer_flags:
                #         if not observer_flags[(feed, band)].is_good:
                #             print(f'Feed {feed} is bad for band {band} in obsid {obsid}')
                #             logging.info(f'Feed {feed} is bad for band {band} in obsid {obsid}')
                #             continue

                if use_flags:
                    scan_flags = db.query_scan_flags(obsid, feed, band) 
                else:
                    scan_flags = np.ones(len(scan_edges), dtype=bool)

                calibration_factors = self.taua_calibration_factors['model_fits'][feed-1, band, channel]
                best_phase = self.taua_calibration_factors['best_phase']
                best_period = self.taua_calibration_factors['best_period']
                calibration_factor = taua_model_func(mjd0, best_phase, best_period, 
                                                     calibration_factors[0],
                                                     calibration_factors[1],
                                                     calibration_factors[2])
                if np.isnan(calibration_factor) | (calibration_factor < 0.3):
                    continue

                for ((iscan, scan_flag), (start, end)) in zip(enumerate(scan_flags),scan_edges):
                    #if scan_flag:
                    length = (end - start)//self.data_container.offset_length * self.data_container.offset_length
                    end = int(start + length)
                    tod_temp = f[self.tod_data_name][feed-1, band, channel, start:end]/ calibration_factor
                    tod_temp -= np.nanmedian(tod_temp)

                    if True:
                        sigma_red = f['level2_noise_stats/binned_filtered_data/sigma_red'][feed-1, band, channel, iscan] /calibration_factor
                        auto_rms  = f['level2_noise_stats/binned_filtered_data/auto_rms'][feed-1, band, channel, iscan] /calibration_factor
                        alphas    = f['level2_noise_stats/binned_filtered_data/alpha'][feed-1, band, channel, iscan]

                        if np.isnan(auto_rms):
                            continue

                        if auto_rms == 0:
                            continue


                        if sigma_red > self.sigma_red_cutoff/calibration_factor:
                            continue
                        if auto_rms > 1e-1/calibration_factor:
                           continue
                        #if alphas < -1.5:
                        #   continue    

                        weights_temp = np.ones_like(tod_temp)/ auto_rms**2
                        weights_temp[np.abs(tod_temp) > auto_rms*30 ] = 0
                    else:
                        weights_temp = np.ones_like(tod_temp)
                    
                    mjd = f['spectrometer/MJD'][start:end]

                    az = f['spectrometer/pixel_pointing/pixel_az'][ifeed, start:end]
                    edge_cut_percentile = 0.9
                    az_range = np.max(az) - np.min(az)
                    mean_az = np.mean(az)
                    cut_az_low = mean_az - az_range*0.5 * edge_cut_percentile
                    cut_az_high = mean_az + az_range*0.5 * edge_cut_percentile

                    el = f['spectrometer/pixel_pointing/pixel_el'][ifeed, start:end]
                    el_range = np.max(el) - np.min(el)
                    mean_el = np.mean(el)
                    cut_el_low = mean_el - el_range*0.5 * edge_cut_percentile
                    cut_el_high = mean_el + el_range*0.5 * edge_cut_percentile

                    ra, dec = h2e_full(az, el, mjd, comap_longitude, comap_latitude)

                    gl, gb = self.coordinate_transform(ra, dec)
                    gb_high = np.percentile(gb, 95)
                    gb_low  = np.percentile(gb, 5) 

                    hpx_pixels = hp.ang2pix(1024,np.radians(90-gb), np.radians(gl))

                    mask_nan = np.isnan(tod_temp) 
                    tod_temp[mask_nan] = 0 
                    weights_temp[mask_nan] = 0
                    #tod_med_filt = np.array(medfilt.medfilt(tod_temp*1, 1000))


                    # Do a close filter to remove the spikes
                    fill_in = (np.abs(tod_temp) > 10*auto_rms) & (self.planck_30_map[hpx_pixels] < 0.04)
                    tod_temp[fill_in] = 0.0
                    weights_temp[fill_in] = 0.0


                    data_filtered = tod_temp - np.array(medfilt.medfilt(tod_temp*1, 100)) 
                    fill_in = (np.abs(data_filtered) > 5*auto_rms) & (self.planck_30_map[hpx_pixels] < 0.04)
                    fill_in = np.convolve(fill_in.astype(int), np.ones(201), mode='same') > 0
                    tod_temp[fill_in] = 0.0
                    weights_temp[fill_in] = 0.0

                    if (np.sum(tod_temp==0)/ len(tod_temp)) > 0.95:
                        continue


                    # print('PLOTTING',feed,band,channel,iscan,obsid)
                    # pyplot.plot(np.abs(data_filtered))
                    # pyplot.plot(tod_temp, 'r',alpha=0.5)
                    # pyplot.axhline(10*auto_rms, color='k', linestyle='--')
                    # pyplot.axhline(5*auto_rms, color='g', linestyle='--')
                    # pyplot.savefig(f'plots_temp_mapmaking/tod_feed_{feed}_band_{band}_channel_{channel}_scan_{iscan}_obsid_{obsid}.png')
                    # pyplot.close() 
                    
                    #tod_temp[~fill_in] = np.interp(np.where(~fill_in)[0], np.where(fill_in)[0], tod_temp[fill_in])
                    planck_mask = (self.planck_30_map[hpx_pixels] < 0.025) & np.isfinite(tod_temp) & (el > cut_el_low) & (el < cut_el_high) & (np.abs(tod_temp) < auto_rms*30) & (tod_temp != 0)


                    # Don't want to apply this to constant el scans.
                    if (cut_el_high - cut_el_low) > 0.1:  
                        time = np.arange(len(tod_temp))
                        try:
                            pmdl = np.poly1d(np.polyfit(time[planck_mask], tod_temp[planck_mask], 5)) 
                        except TypeError: # if there is no data to fit, we will skip this scan
                            continue
                        tod_temp -= pmdl(time)   

                    if (cut_el_high - cut_el_low) > 0.1:  
                        try:
                            pmdl = np.poly1d(np.polyfit(el[planck_mask], tod_temp[planck_mask], 3)) 
                        except TypeError: # if there is no data to fit, we will skip this scan
                            continue
                        tod_temp -= pmdl(el) 


                    if (cut_el_high - cut_el_low) > 0.1:  
                        planck_mask = (self.planck_30_map[hpx_pixels] < 0.02) & np.isfinite(tod_temp) & (az > cut_az_low) & (az < cut_az_high) & (np.abs(tod_temp) < auto_rms*30)
                        try:
                            pmdl = np.poly1d(np.polyfit(az[planck_mask], tod_temp[planck_mask], 3)) 
                        except TypeError: # if there is no data to fit, we will skip this scan
                            continue
                        tod_temp -= pmdl(az) 

                    #pyplot.plot(tod_temp) 
                    #pyplot.title(f'Feed {feed} Band {band} Channel {channel} Scan {iscan} ObsID {obsid}')
                    #pyplot.plot()
                    # pyplot.plot(time[planck_mask],tod_temp[planck_mask], 'r')
                    # pyplot.plot( pmdl(time)   , 'g')
                    # pyplot.savefig(f'plots_map_making/tod_feed_{feed}_band_{band}_channel_{channel}_scan_{iscan}_obsid_{obsid}.png')
                    # pyplot.close()


                    mask_nan = np.isnan(tod_temp)  | np.isinf(tod_temp) | (tod_temp == 0)

                    if np.nanmax(np.abs(tod_temp)) > 1e3:
                        logging.info(f'Feed {feed} Band {band} Channel {channel} Scan {iscan} ObsID {obsid} has a large spike in the data, skipping')
                        continue
                    tod.append(tod_temp)
                    weights.append(weights_temp)
                    x.append(gl)
                    y.append(gb) 

                    # check data 
                    tod[-1][mask_nan] = 0 
                    weights[-1][mask_nan] = 0

                    weights[-1][(self.planck_30_map[hpx_pixels] > 0.045)] *= 1e-1# weight down high signal areas to avoid biasing

                    # Flag turn arounds in az and el --- tend to have systematic gradients 
                    mask_az = (az < cut_az_low) | (az > cut_az_high)
                    weights[-1][mask_az] = 0 
                    tod[-1][mask_az] = 0 


                    if (cut_el_high - cut_el_low) > 0.1: # Don't want to apply this to constant el scans. 
                       mask_el = (el < cut_el_low) | (el > cut_el_high)
                       weights[-1][mask_el] = 0 
                       tod[-1][mask_el] = 0
                    if np.isnan(np.sum(weights[-1][tod[-1] != 0])):
                        print('NAN!')
                        print(np.sum(tod[-1][tod[-1] != 0]))
                        print(np.sum(weights[-1][tod[-1] != 0]))


            try:
                tod = np.concatenate(tod).astype(np.float32)
                weights = np.concatenate(weights).astype(np.float32)
                x = np.concatenate(x)
                y = np.concatenate(y)
            except ValueError:
                tod = np.array([], dtype=np.float32)
                weights = np.array([], dtype=np.float32)
                x = np.array([], dtype=np.float32)
                y = np.array([], dtype=np.float32)

            return tod, weights, x, y
        
    def coordinate_transform(self, x, y) -> tuple:
        """
        Transform the coordinates from input to output
        """
        if self.input_coord != self.output_coord:
            rot = hp.rotator.Rotator(coord=[self.input_coord, self.output_coord])
            lon, lat = rot(x, y, lonlat=True)
            lon = np.mod(lon, 360)
            # print(self.input_coord, self.output_coord)
            # print('XY',np.min(x),np.max(x),np.min(y),np.max(y))
            # print('LONLAT',np.min(lon),np.max(lon),np.min(lat),np.max(lat))
            # print('CRVAL1',self.header['CRVAL1'],self.header['CRVAL2']) 
            # sys.exit()
            # from matplotlib import pyplot 
            # pyplot.plot(lon, lat, '-')
            # pyplot.plot(self.header['CRVAL1'], self.header['CRVAL2'], 'ro')
            # pyplot.savefig('test.png')
            # pyplot.close()
            return lon, lat

        return x, y
    
    def coords_to_pixels(self, x, y) -> np.ndarray:
        """
        Convert coordinates to pixels
        """
        xpix, ypix = self.wcs.all_world2pix(x, y, 0)
        print(self.header['NAXIS2'],self.header['NAXIS1'])
        mask = (ypix < 0) | (xpix < 0) | (xpix >= self.header['NAXIS1']) | (ypix >= self.header['NAXIS2'])
        xpix[mask] = 0
        ypix[mask] = 0
        pixels = np.ravel_multi_index((ypix.astype(int), xpix.astype(int)), (self.header['NAXIS2'],self.header['NAXIS1'])).astype(np.int64)
        pixels[mask] = -1
        return pixels
    

    def get_rhs(self, tod : np.ndarray, weights : np.ndarray) -> np.ndarray:
        """
        Get the right hand side of the map making equation
        """
        n_rhs = len(tod) // self.data_container.offset_length 
        rhs = np.zeros(n_rhs, dtype=np.float32)

        bin_funcs.bin_tod_to_rhs(rhs, tod, weights, self.data_container.offset_length) 

        return rhs 

    def fit_background(self, tod : np.ndarray, weights : np.ndarray, pixels : np.ndarray) -> np.ndarray:
        """ """ 
        temp_sum_map = np.zeros_like(self.data_container.sum_map).astype(np.float32) 
        temp_weight_map = np.zeros_like(self.data_container.weight_map).astype(np.float32)
        temp_hits_map = np.zeros_like(self.data_container.hits_map).astype(np.float32)
        temp_sky_map = np.zeros_like(self.data_container.sum_map).astype(np.float32) 
        bin_funcs.bin_tod_to_map(temp_sum_map,
                                    temp_weight_map,
                                    temp_hits_map,
                                    tod, weights, pixels) 
        mask = temp_weight_map > 0
        temp_sky_map[mask] = temp_sum_map[mask] / temp_weight_map[mask] 
        # Subtract a 2D gradient from non-zero pixels in temp_sky_map 
        temp_sky_map = np.reshape(temp_sky_map, (self.header['NAXIS2'], self.header['NAXIS1'])) 
        # Import fitting models from astropy
        from astropy.modeling import models, fitting
        # Create a 2D polynomial model
        p_init = models.Polynomial2D(degree=1)
        # Create a fitter
        fit_p = fitting.LevMarLSQFitter()
        # Generate x and y coordinates
        # Reshape first
        temp_sky_map_2d = np.reshape(temp_sky_map, (self.header['NAXIS2'], self.header['NAXIS1']))
        x = np.arange(self.header['NAXIS1'])
        y = np.arange(self.header['NAXIS2'])
        X, Y = np.meshgrid(x, y)

        # convert to wcs coordinates
        gl, gb = self.wcs.all_pix2world(X, Y, 0) 

        # Create mask for observed pixels
        nonzero_hits = temp_hits_map > 0
        if np.sum(nonzero_hits) == 0:
            return np.zeros_like(temp_sky_map_2d)
        lower_hit_cut = np.percentile(temp_hits_map[nonzero_hits], 5)
        nonzero = (temp_sky_map_2d != 0) & (np.abs(gb) > 0.75) & (temp_hits_map.reshape(self.header['NAXIS2'],self.header['NAXIS1']) > lower_hit_cut)

        # Fit only to observed pixels
        p = fit_p(p_init, X[nonzero], Y[nonzero], temp_sky_map_2d[nonzero])

        # Generate full model
        model_sky_map = p(X, Y)
        nonzero = (temp_sky_map_2d != 0)
        # Mask unobserved regions
        model_sky_map[~nonzero] = 0     
        return model_sky_map

    def accummulate_data(self, file: str, band : int, channel : int, database, use_flags : bool = True, bad_feeds : list = []) -> None:
        """
        Read in the level 2 file and accumulate the data
        """
        tod, weights, x, y = self.read_data(file, band, channel, database, use_flags=use_flags, bad_feeds=bad_feeds) 
        if len(tod) == 0:
            return 
        weights[tod == 0] = 0
        #x, y = self.coordinate_transform(x, y) 
        pixels = self.coords_to_pixels(x, y) 
        weights[pixels == -1] = 0
        #try:
        #    model_sky_map = self.fit_background(tod, weights, pixels)
        #    tod -= model_sky_map.flatten()[pixels] 
        #except TypeError:
        #    return 

        # pyplot.plot(tod)
        # pyplot.title(f'Band {band} Channel {channel} ObsID {os.path.basename(file)}')
        # pyplot.savefig(f'plots_temp_mapmaking/tod_band_{band}_channel_{channel}_obsid_{os.path.basename(file)}.png')
        # pyplot.close() 
        rhs = self.get_rhs(tod, weights) 


         
                


        bin_funcs.bin_tod_to_map(self.data_container.sum_map, 
                                 self.data_container.weight_map, 
                                 self.data_container.hits_map, 
                                 tod, weights, pixels)
        
        
        print('SUM TOD',np.sum(tod),np.sum(weights),np.sum(rhs), np.sum(self.data_container.sum_map),np.sum(self.data_container.weight_map))
        if np.isnan(np.sum(self.data_container.sum_map)):
            print('NAN SUM MAP!')
            print(f'FILE {file}')
            sys.exit() 
        if np.isnan(np.sum(self.data_container.weight_map)):
            print('NAN WEIGHT MAP!')
            print(f'FILE {file}')
            sys.exit()

        self.data_container.rhs[self._last_rhs_index:self._last_rhs_index + len(rhs)] = rhs
        self._last_rhs_index += len(rhs)

        self.data_container.weights[self._last_pixels_index:self._last_pixels_index + len(weights)] = weights
        self.data_container.pixels[self._last_pixels_index:self._last_pixels_index + len(pixels)] = pixels 
        self.data_container.tod[self._last_pixels_index:self._last_pixels_index + len(tod)] = tod
        self._last_pixels_index += len(pixels)

    def count_tod(self, file: str, band : int, channel : int, use_flags : bool = True, bad_feeds : list = []) -> int:
        """
        Count the number of TODs in the file
        """
        with h5py.File(file, 'r') as f:
            obsid = int(os.path.basename(file).split('-')[1])
            feeds = f['spectrometer/feeds'][:-1] 
            scan_edges = f['level2/scan_edges'][...] 
            n_tod = 0
            for feed in feeds: 
                if feed in bad_feeds:
                    continue
                if use_flags:
                    scan_flags = db.query_scan_flags(obsid, feed, band) 
                else:
                    scan_flags = np.ones(len(scan_edges), dtype=bool)
                for (scan_flag, (start, end)) in zip(scan_flags,scan_edges):
                    if scan_flag:
                        length = (end - start)//self.data_container.offset_length * self.data_container.offset_length
                        n_tod += int(length)

            return n_tod

    def read_files(self, files: list, database = None, band :int = 0, channel : int = 0, feeds : list = [f for f in range(1,20)], use_flags=False) -> None:
        """
        Read in the level 2 files
        """
        logging.info('Reading in level 2 files')

        # bad_feeds is the inverse of feeds
        bad_feeds = [f for f in range(1,20) if f not in feeds]

        # First count total tod size 
        if len(files) == 0:
            print('No files found')
            return False
        for file in files:
            n_tod = self.count_tod(file, self.band, self.channel, use_flags=use_flags, bad_feeds=bad_feeds) 
            self.n_tod_per_file.append(n_tod)
            self._n_tod += n_tod
        self.data_container.tod    = np.zeros(self._n_tod, dtype=np.float32) 
        self.data_container.weights= np.zeros(self._n_tod, dtype=np.float32)
        self.data_container.pixels = np.zeros(self._n_tod, dtype=np.int64)
        self.data_container.rhs    = np.zeros(self._n_tod//self.data_container.offset_length, dtype=np.float32)

        n_files = len(files)
        logging.info(f'Total number of files: {n_files}')
        print(f'Total number of files: {n_files}')
        n_file_reads = 0
        for ifile, file in enumerate(files):
            percent_done = ifile/float(n_files) * 100
            if (percent_done - n_file_reads*10 > 10):
                logging.info(f'File reading is {percent_done:.2f} % complete')
                print(f'File reading is {percent_done:.2f} % complete')
                n_file_reads += 1
            print(f'Reading in file: {file}')
            self.accummulate_data(file, self.band, self.channel, database, use_flags=use_flags, bad_feeds=bad_feeds) 

        comm.Barrier() 

        self.data_container.sum_map = sum_map_all_inplace(self.data_container.sum_map)
        self.data_container.weight_map = sum_map_all_inplace(self.data_container.weight_map)
        self.data_container.hits_map = sum_map_all_inplace(self.data_container.hits_map)

        self.data_container.sky_map = np.zeros_like(self.data_container.sum_map).astype(np.float32) 
        mask = self.data_container.weight_map > 0
        self.data_container.sky_map[mask] = self.data_container.sum_map[mask] / self.data_container.weight_map[mask] 
        
        bin_funcs.subtract_sky_map_from_rhs(self.data_container.rhs, 
                                            self.weights,
                                            self.data_container.sky_map,
                                            self.data_container.pixels,
                                            self.data_container.offset_length)
        
        return True