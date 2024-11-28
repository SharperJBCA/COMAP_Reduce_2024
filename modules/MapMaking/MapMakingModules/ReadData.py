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

from modules.SQLModule.SQLModule import db
from astropy.wcs import WCS
import healpy as hp 

from modules.utils import bin_funcs


@dataclass
class Level2Data_Nov2024: 

    offset_length : int = 100
    input_coord   : str = 'C' 
    output_coord  : str = 'G'
    feeds : list = field(default_factory=list)
    band  : int  = 0

    sum_map    : np.ndarray = field(default_factory=np.empty(0,dtype=float)) 
    weight_map : np.ndarray = field(default_factory=np.empty(0,dtype=float))  
    hits_map   : np.ndarray = field(default_factory=np.empty(0,dtype=float))
    pixels     : np.ndarray = field(default_factory=np.empty(0,dtype=int)) 
    rhs        : np.ndarray = field(default_factory=np.empty(0,dtype=float))

    _n_tod : int  = 0
    n_tod_per_file : list = field(default_factory=list)

    _last_pixels_index : int = 0 
    _last_rhs_index    : int = 0

    def setup_wcs(self, header) -> None:
        """
        Setup the WCS of the maps
        """
        self.header = header 
        self.wcs = WCS(header) 

        self.sum_map    = np.zeros((self.header['NAXIS2'], self.header['NAXIS1']))
        self.weight_map = np.zeros((self.header['NAXIS2'], self.header['NAXIS1'])) 
        self.hits_map   = np.zeros((self.header['NAXIS2'], self.header['NAXIS1']))

    def read_data(self, file: str,  band : int, channel : int, use_flags : bool = True) -> tuple:

        with h5py.File(file, 'r') as f:
            obsid = int(f['comap'].attrs['obsid'])
            feeds = f['spectrometer/feeds'][:-1] 
            scan_edges = f['level2/scan_edges'][...] 
            tod = []
            weights = []
            x   = [] 
            y   = [] 
            for feed in feeds: 
                if use_flags:
                    scan_flags = db.query_scan_flags(obsid, feed, band) 
                else:
                    scan_flags = np.ones(len(scan_edges), dtype=bool)
                for (iscan, scan_flag, (start, end)) in zip(enumerate(scan_flags),scan_edges):
                    if scan_flag:
                        length = (end - start)//self.offset_length * self.offset_length
                        end = int(start + length)

                        tod.append(f['level2/binned_filtered_data'][feed, band, channel, start:end])
                        auto_rms = f['level2_noise_stats/auto_rms'][feed, band, channel, iscan]
                        weights.append(np.ones_like(tod[-1])/ auto_rms**2)
                        x.append(f['spectrometer/pixel_pointing/pixel_ra'][feed, start:end])
                        y.append(f['spectrometer/pixel_pointing/pixel_dec'][feed, start:end]) 

            tod = np.concatenate(tod)
            weights = np.concatenate(weights)
            x = np.concatenate(x)
            y = np.concatenate(y)

            return tod, weights, x, y
        
    def coordinate_transform(self, x, y) -> tuple:
        """
        Transform the coordinates from input to output
        """
        if self.input_coord != self.output_coord:
            rot = hp.rotator.Rotator(coord=[self.input_coord, self.output_coord])
            lon, lat = rot(x, y, lonlat=True)
            return lon, lat

        return x, y
    
    def coords_to_pixels(self, x, y) -> np.ndarray:
        """
        Convert coordinates to pixels
        """
        xpix, ypix = self.wcs.all_world2pix(x, y, 0)
        return np.ravel_multi_index((ypix.astype(int), xpix.astype(int)), self.sum_map.shape)
    

    def get_rhs(self, tod : np.ndarray, weights : np.ndarray) -> np.ndarray:
        """
        Get the right hand side of the map making equation
        """
        n_rhs = self._n_tod // self.offset_length 
        rhs = np.zeros(n_rhs, dtype=float)

        bin_funcs.bin_tod_to_rhs(rhs, tod, weights, self.offset_length) 

        return rhs 


    def accummulate_data(self, file: str, band : int, use_flags : bool = True) -> None:
        """
        Read in the level 2 file and accumulate the data
        """
        tod, weights, x, y = self.read_data(file, band, use_flags=use_flags) 
        x, y = self.coordinate_transform(x, y) 
        pixels = self.coords_to_pixels(x, y) 
        rhs = self.get_rhs() 

        self.sum_map    += bin_funcs.bin_tod_to_map(self.sum_map, self.weight_map, self.hits_map, tod, weights, pixels)

        self.rhs[self._last_rhs_index:self._last_rhs_index + len(rhs)] = rhs
        self._last_rhs_index += len(rhs)

        self.weights[self._last_pixels_index:self._last_pixels_index + len(weights)] = weights
        self.pixels[self._last_pixels_index:self._last_pixels_index + len(pixels)] = pixels 
        self._last_pixels_index += len(pixels)

    def count_tod(self, file: str, band : int, use_flags : bool = True) -> int:
        """
        Count the number of TODs in the file
        """
        with h5py.File(file, 'r') as f:
            obsid = int(f['comap'].attrs['obsid'])
            feeds = f['spectrometer/feeds'][:-1] 
            scan_edges = f['level2/scan_edges'][...] 
            n_tod = 0
            for feed in feeds: 
                if use_flags:
                    scan_flags = db.query_scan_flags(obsid, feed, band) 
                else:
                    scan_flags = np.ones(len(scan_edges), dtype=bool)
                for (scan_flag, (start, end)) in zip(scan_flags,scan_edges):
                    if scan_flag:
                        length = (end - start)//self.offset_length * self.offset_length
                        n_tod += int(length)

            return n_tod

    def read_files(self, files: list, use_flags=True) -> None:
        """
        Read in the level 2 files
        """
        logging.info('Reading in level 2 files')

        # First count total tod size 
        for file in files:
            n_tod = self.count_tod(file, use_flags=use_flags) 
            self.n_tod_per_file.append(n_tod)
            self._n_tod += n_tod

        self.weights= np.zeros(self._n_tod, dtype=float)
        self.pixels = np.zeros(self._n_tod, dtype=int)
        self.rhs    = np.zeros(self._n_tod//self.offset_length, dtype=float)

        for file in files:
            logging.info(f'Reading in file: {file}')
            self.accummulate_data(file, use_flags=use_flags) 
