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
    pixels     : np.ndarray = field(default_factory=np.empty(0,dtype=np.int32)) 
    weights    : np.ndarray = field(default_factory=np.empty(0,dtype=np.float32))
    rhs        : np.ndarray = field(default_factory=np.empty(0,dtype=np.float32))


class Level2DataReader_Nov2024: 

    def __init__(self, tod_data_name : str = 'level2/binned_filtered_data') -> None:
        self.tod_data_name = tod_data_name


        self.data_container = Level2Data_Nov2024
        self.data_container.offset_length = 100
        self.data_container.rhs        = np.empty(0, dtype=np.float32)
        self.data_container.weights    = np.empty(0, dtype=np.float32)
        self.data_container.pixels     = np.empty(0, dtype=np.int32)
        self.data_container.sum_map    = np.empty(0, dtype=np.float32)
        self.data_container.weight_map = np.empty(0, dtype=np.float32)
        self.data_container.hits_map   = np.empty(0, dtype=np.float32)
        self.data_container.sky_map    = np.empty(0, dtype=np.float32)


        self.data_container.tod = np.empty(0, dtype=np.float32)


        self.input_coord  = 'C' 
        self.output_coord = 'G' 
        self.feeds = []
        self.band  = 0
        self.channel = 0

        self._n_tod  = 0
        self.n_tod_per_file = []

        self._last_pixels_index = 0 
        self._last_rhs_index    = 0

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

    def read_data(self, file: str,  band : int, channel : int, use_flags : bool = False, bad_feeds : list = []) -> tuple:

        with h5py.File(file, 'r') as f:
            obsid = int(os.path.basename(file).split('-')[1])
            feeds = f['spectrometer/feeds'][:-1] 
            scan_edges = f['level2/scan_edges'][...] 
            tod = []
            weights = []
            x   = [] 
            y   = [] 
            for ifeed, feed in enumerate(feeds): 
                if feed in bad_feeds:
                    continue

                if use_flags:
                    scan_flags = db.query_scan_flags(obsid, feed, band) 
                else:
                    scan_flags = np.ones(len(scan_edges), dtype=bool)
                for ((iscan, scan_flag), (start, end)) in zip(enumerate(scan_flags),scan_edges):
                    #if scan_flag:
                    length = (end - start)//self.data_container.offset_length * self.data_container.offset_length
                    end = int(start + length)
                    tod_temp = f[self.tod_data_name][feed-1, band, channel, start:end]  # _filtered
                    tod_temp -= np.nanmedian(tod_temp)
                    #tod_temp -= np.array(medfilt.medfilt(tod_temp*1, 1000))
                    #tod_temp[np.where(np.isnan(tod_temp))] = tod_temp[np.where(np.isnan(tod_temp))[0]-1]
                    if True:
                        sigma_red = f['level2_noise_stats/binned_filtered_data/sigma_red'][feed-1, band, channel, iscan] 
                        auto_rms  = f['level2_noise_stats/binned_filtered_data/auto_rms'][feed-1, band, channel, iscan]
                        alphas    = f['level2_noise_stats/binned_filtered_data/alpha'][feed-1, band, channel, iscan]


                        if sigma_red > 0.5:
                            continue
                        if auto_rms > 1e-1:
                            continue
                        if alphas < -1.5:
                            continue    

                        weights_temp = np.ones_like(tod_temp)/ auto_rms**2
                        weights_temp[np.abs(tod_temp) > auto_rms*30 ] = 0
                    else:
                        weights_temp = np.ones_like(tod_temp)
                    
                    tod.append(tod_temp)
                    weights.append(weights_temp)
                    az = f['spectrometer/pixel_pointing/pixel_az'][ifeed, start:end]
                    el = f['spectrometer/pixel_pointing/pixel_el'][ifeed, start:end]
                    mjd = f['spectrometer/MJD'][start:end]
                    ra, dec = h2e_full(az, el, mjd, comap_longitude, comap_latitude)
                    x.append(ra)
                    y.append(dec) 

                    # check data 
                    mask = np.isnan(tod[-1])
                    tod[-1][mask] = 0 
                    weights[-1][mask] = 0

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
        pixels = np.ravel_multi_index((ypix.astype(int), xpix.astype(int)), (self.header['NAXIS2'],self.header['NAXIS1'])).astype(np.int32)
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

    def accummulate_data(self, file: str, band : int, channel : int, use_flags : bool = True, bad_feeds : list = []) -> None:
        """
        Read in the level 2 file and accumulate the data
        """
        tod, weights, x, y = self.read_data(file, band, channel, use_flags=use_flags, bad_feeds=bad_feeds) 
        if len(tod) == 0:
            return 
        x, y = self.coordinate_transform(x, y) 
        pixels = self.coords_to_pixels(x, y) 
        weights[pixels == -1] = 0
        #try:
        #    model_sky_map = self.fit_background(tod, weights, pixels)
        #    tod -= model_sky_map.flatten()[pixels] 
        #except TypeError:
        #    return 

        print('SUM TOD',np.sum(tod),np.sum(weights))
        rhs = self.get_rhs(tod, weights) 


         
                


        bin_funcs.bin_tod_to_map(self.data_container.sum_map, 
                                 self.data_container.weight_map, 
                                 self.data_container.hits_map, 
                                 tod, weights, pixels)

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

    def read_files(self, files: list, band :int = 0, channel : int = 0, feeds : list = [f for f in range(1,20)], use_flags=False) -> None:
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
        self.data_container.pixels = np.zeros(self._n_tod, dtype=np.int32)
        self.data_container.rhs    = np.zeros(self._n_tod//self.data_container.offset_length, dtype=np.float32)

        for file in files:
            logging.info(f'Reading in file: {file}')
            print(f'Reading in file: {file}')
            self.accummulate_data(file, self.band, self.channel, use_flags=use_flags, bad_feeds=bad_feeds) 

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