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
class Level2LineData_Nov2024: 
    offset_length : int = 100

    sum_map    : np.ndarray = field(default_factory=np.empty(0,dtype=np.float32)) 
    weight_map : np.ndarray = field(default_factory=np.empty(0,dtype=np.float32))  
    sky_map    : np.ndarray = field(default_factory=np.empty(0,dtype=np.float32))
    hits_map   : np.ndarray = field(default_factory=np.empty(0,dtype=np.float32))
    pixels     : np.ndarray = field(default_factory=np.empty(0,dtype=np.int64)) 
    weights    : np.ndarray = field(default_factory=np.empty(0,dtype=np.float32))
    rhs        : np.ndarray = field(default_factory=np.empty(0,dtype=np.float32))


class Level2LineDataReader_Nov2024: 

    def __init__(self, tod_data_name : str = 'level2/binned_filtered_data') -> None:
        self.tod_data_name = tod_data_name


        self.data_container = Level2LineData_Nov2024
        self.data_container.sum_map    = np.empty(0, dtype=np.float32)
        self.data_container.weight_map = np.empty(0, dtype=np.float32)
        self.data_container.hits_map   = np.empty(0, dtype=np.float32)
        self.data_container.sky_map    = np.empty(0, dtype=np.float32)
        self.input_coord  = 'C' 
        self.output_coord = 'G' 
        self.feeds = []
        self.band  = 0
        self.channel = 0

        self._n_tod  = 0
        self.n_tod_per_file = []

        self._last_pixels_index = 0 

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

    def setup_wcs(self, wcs_def, line_frequency, line_segment_width, config_file='map_making_configs/wcs_definitions.toml') -> None:
        """
        Setup the WCS of the maps
        """
        self.wcs_def = wcs_def
        with open(config_file, 'r') as f:
            config = toml.load(f)

        header = Header(config[wcs_def])

        self.header = header 
        self.wcs = WCS(header) 

        self.header['NAXIS3'] = line_segment_width
        self.header['CTYPE3'] = 'FREQ'
        self.header['CRVAL3'] = line_frequency
        self.header['CDELT3'] = 2./1024. 
        self.header['CRPIX3'] = 1

        self.data_container.sum_map    = np.zeros((line_segment_width,self.header['NAXIS2']*self.header['NAXIS1']), dtype=np.float32)
        self.data_container.weight_map = np.zeros((line_segment_width,self.header['NAXIS2']*self.header['NAXIS1']), dtype=np.float32)
        self.data_container.hits_map   = np.zeros((self.header['NAXIS2']*self.header['NAXIS1']), dtype=np.float32)

    def read_data(self, level1_file : str, level2_file : str, line_frequency : float, line_segment_width : int, database, use_flags : bool = False, bad_feeds : list = []) -> tuple:

        with h5py.File(level1_file, 'r') as f_lvl1:
            frequencies = f_lvl1['spectrometer/frequency'][...] 
            for i, freq in enumerate(frequencies):
                if np.min(freq) < line_frequency < np.max(freq):
                    band = i
                    statistics_channel = 0
                    channel_idx = np.argmin((freq - line_frequency)**2)
                    channel_start = int(np.max([channel_idx - line_segment_width//2, 0]))
                    channel_end   = int(np.min([channel_idx + line_segment_width//2, len(freq)]))

                    self.header['CRVAL3'] = freq[channel_start]
                    self.header['CDELT3'] = (freq[channel_start] - freq[channel_start+1]) 

                    break
            
            with h5py.File(level2_file, 'r') as f_lvl2:
                obsid = int(os.path.basename(level2_file).split('-')[1])
                feeds = f_lvl2['spectrometer/feeds'][:-1] 
                scan_edges = f_lvl2['level2/scan_edges'][...] 
                tod = []
                weights = []
                x   = [] 
                y   = [] 
                if database: 
                    observer_flags = db.get_quality_flags(obsid)

                for ifeed, feed in enumerate(feeds): 
                    if feed in bad_feeds:
                        continue

                    if database: 
                        if (feed, band) in observer_flags:
                            if not observer_flags[(feed, band)].is_good:
                                print(f'Feed {feed} is bad for band {band} in obsid {obsid}')
                                logging.info(f'Feed {feed} is bad for band {band} in obsid {obsid}')
                                continue

                    if use_flags:
                        scan_flags = db.query_scan_flags(obsid, feed, band) 
                    else:
                        scan_flags = np.ones(len(scan_edges), dtype=bool)
                    for ((iscan, scan_flag), (start, end)) in zip(enumerate(scan_flags),scan_edges):
                        logging.info(f'Obsid {obsid} Feed {feed} Band {band} Scan {iscan} Start {start} End {end}')
                        length = (end - start)//self.data_container.offset_length * self.data_container.offset_length
                        end = int(start + length)



                        if True:
                            sigma_red = f_lvl2['level2_noise_stats/binned_filtered_data/sigma_red'][feed-1, band, statistics_channel, iscan] 
                            auto_rms  = f_lvl2['level2_noise_stats/binned_filtered_data/auto_rms'][feed-1, band, statistics_channel, iscan]
                            alphas    = f_lvl2['level2_noise_stats/binned_filtered_data/alpha'][feed-1, band, statistics_channel, iscan]


                            if sigma_red > 0.5:
                                continue
                            if auto_rms > 1e-1:
                                continue
                            if alphas < -1.5:
                                continue    

                            weights_temp = np.ones((line_segment_width,length))/ auto_rms**2
                        else:
                            weights_temp = np.ones((line_segment_width,length))
                        
                        az = f_lvl2['spectrometer/pixel_pointing/pixel_az'][ifeed, start:end]
                        el = f_lvl2['spectrometer/pixel_pointing/pixel_el'][ifeed, start:end]
                        mjd = f_lvl2['spectrometer/MJD'][start:end]
                        ra, dec = h2e_full(az, el, mjd, comap_longitude, comap_latitude)

                        tod_temp = f_lvl1['spectrometer/tod'][ifeed, band, channel_start:channel_end, start:end]
                        tod_lvl2 = f_lvl2['level2/binned_filtered_data'][ifeed, band, statistics_channel, start:end]
                        tod_lvl2_no_filter = f_lvl2['level2/binned_data'][ifeed, band, statistics_channel, start:end]
                        gain_filter = tod_lvl2_no_filter - tod_lvl2
                        gain     = f_lvl2['level2/vane/gain'][0,ifeed,band,channel_start:channel_end] 
                    
                        # Calibrate the spectrum 
                        tod_temp = tod_temp / gain[:,None] 



                        # Fit the gain filter solution to each channel
                        mask = np.isfinite(gain_filter)
                        if not np.any(mask):
                            continue
                        if np.nansum(gain_filter) == 0:
                            continue
                        # Remove the atmosphere offsets and tau
                        atmos_offsets = f_lvl2['level2/atmosphere/offsets'][ifeed,band,channel_start:channel_end,iscan]
                        atmos_tau = f_lvl2['level2/atmosphere/tau'][ifeed,band,channel_start:channel_end,iscan] 

                        tod_temp[...] -= (atmos_offsets[...,np.newaxis] +\
                                                         atmos_tau[...,np.newaxis]/np.sin(np.radians(el)))

                        for i in range(tod_temp.shape[0]): 
                            tod_temp[i] -= np.median(tod_temp[i])
                            pmdl = np.poly1d(np.polyfit(gain_filter[mask],tod_temp[i,mask], 1))
                            tod_temp[i,mask] -= pmdl(gain_filter[mask])
                        #    edges = [0,1,2,3,tod_temp.shape[0]-4,tod_temp.shape[0]-3,tod_temp.shape[0]-2,tod_temp.shape[0]-1]
                        #    pfit = np.poly1d(np.polyfit(edges, tod_temp[edges,i], 1))
                        #    tod_temp[:,i] -= pfit(np.arange(tod_temp.shape[0]))
                            

                        print(tod_temp.shape, weights_temp.shape, ra.shape, dec.shape)
                        tod.append(tod_temp)
                        weights.append(weights_temp)
                        x.append(ra)
                        y.append(dec) 

                        # check data 
                        mask = np.isnan(tod[-1]) | np.isnan(weights[-1])
                        tod[-1][mask] = 0 
                        weights[-1][mask] = 0

                try:
                    tod = np.concatenate(tod,axis=1).astype(np.float32)
                    weights = np.concatenate(weights,axis=1).astype(np.float32)
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
    

    def accummulate_data(self, level1_file: str, level2_file : str, line_frequency : float, line_segment_width : int, database, use_flags : bool = True, bad_feeds : list = []) -> None:
        """
        Read in the level 2 file and accumulate the data
        """
        tod, weights, x, y = self.read_data(level1_file, level2_file, line_frequency, line_segment_width, database, use_flags=use_flags, bad_feeds=bad_feeds) 
        if len(tod) == 0:
            return 
        x, y = self.coordinate_transform(x, y) 
        pixels = self.coords_to_pixels(x, y) 
        weights[:,pixels == -1] = 0

        for ifreq in range(line_segment_width):
            bin_funcs.bin_tod_to_map(self.data_container.sum_map[ifreq], 
                                    self.data_container.weight_map[ifreq], 
                                    self.data_container.hits_map, 
                                    tod[ifreq], weights[ifreq], pixels)

        self._last_pixels_index += len(pixels)


    def read_files(self, files: list, database = None, line_frequency : float = 31.22332 , line_segment_width : int = 30, feeds : list = [f for f in range(1,20)], use_flags=False) -> None:
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
        n_files = len(files)
        logging.info(f'Total number of files: {n_files}')
        n_file_reads = 0
        for ifile, (level1_file, level2_file) in enumerate(files):
            percent_done = ifile/float(n_files) * 100
            if (percent_done - n_file_reads*10 > 10):
                logging.info(f'File reading is {percent_done:.2f} % complete')
                n_file_reads += 1
            print(f'Reading in file: {level2_file}')
            self.accummulate_data(level1_file, level2_file, line_frequency, line_segment_width, 
                                  database, use_flags=use_flags, bad_feeds=bad_feeds) 

        comm.Barrier() 

        for ifreq in range(line_segment_width):
            self.data_container.sum_map[ifreq]    = sum_map_all_inplace(self.data_container.sum_map[ifreq])
            self.data_container.weight_map[ifreq] = sum_map_all_inplace(self.data_container.weight_map[ifreq])
        self.data_container.hits_map   = sum_map_all_inplace(self.data_container.hits_map)

        self.data_container.sky_map = np.zeros_like(self.data_container.sum_map).astype(np.float32) 
        mask = self.data_container.weight_map > 0
        self.data_container.sky_map[mask] = self.data_container.sum_map[mask] / self.data_container.weight_map[mask] 
                
        return True