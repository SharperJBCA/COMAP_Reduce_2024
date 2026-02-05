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


from modules.ProcessLevel2.GainFilterAndBin.GainFilters import GainFilterBase 

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

    def __init__(self, tod_data_name : str = 'level2/binned_filtered_data', auto_rms_cut = 0.1, sigma_red_cut=0.5, alphas_cut=-1.5) -> None:
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

        self.gain_filter = GainFilterBase()

        self.auto_rms_cut = auto_rms_cut
        self.sigma_red_cut = sigma_red_cut
        self.alphas_cut = alphas_cut
        self.statistics_channel = 0 

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

        n_pixels = header['NAXIS1'] * header['NAXIS2']
        self.data_container.sum_map = np.zeros((line_segment_width, n_pixels), dtype=np.float32)
        self.data_container.weight_map = np.zeros((line_segment_width, n_pixels), dtype=np.float32)
        self.data_container.hits_map = np.zeros(n_pixels, dtype=np.float32)

    def select_band(self, frequencies, line_frequency, line_segment_width):
        """Select band and channel indices covering line frequency."""
        for i, freq in enumerate(frequencies):
            if np.min(freq) < line_frequency < np.max(freq):
                channel_idx = np.argmin((freq - line_frequency) ** 2)
                start = max(channel_idx - line_segment_width // 2, 0)
                end = min(channel_idx + line_segment_width // 2, len(freq))
                return i, start, end
        return None, None, None
    
    def flags_are_bad(self, obsid, feed, band, iscan, f_lvl2, database) -> bool:
        """Return True if flags/noise stats indicate bad data."""
        # Observer flags from DB (optional)
        if database:
            observer_flags = db.get_quality_flags(obsid).get((feed, band), None)
            if observer_flags is not None and not observer_flags.is_good:
                return True

        # Noise stats (use feed-1 because arrays are 0-based, feeds are 1-based)
        # Use a safe statistics_channel (mid-channel) if not available elsewhere.
        sigma_arr = f_lvl2['level2_noise_stats/binned_filtered_data/sigma_red']
        rms_arr   = f_lvl2['level2_noise_stats/binned_filtered_data/auto_rms']
        alpha_arr = f_lvl2['level2_noise_stats/binned_filtered_data/alpha']

        n_stats_ch = sigma_arr.shape[2]
        stats_ch = getattr(self, "statistics_channel", None)
        if stats_ch is None:
            stats_ch = n_stats_ch // 2  # midpoint channel

        sigma_red = sigma_arr[feed-1, band, stats_ch, iscan]
        auto_rms  = rms_arr[feed-1, band, stats_ch, iscan]
        alpha     = alpha_arr[feed-1, band, stats_ch, iscan]

        # Simple cuts (tune as needed)
        if sigma_red > self.sigma_red_cut:
            return True
        if auto_rms > self.auto_rms_cut:
            return True
        if alpha < self.alpha_cut:
            return True

        return False

    def apply_atmosphere_correction(self, tod, atmos_offsets, atmos_tau, elevation):
        """
        Subtract atmosphere offsets and tau from TOD.
        Shapes:
        tod:            (n_chan, n_time)
        atmos_offsets:  (n_chan,)
        atmos_tau:      (n_chan,)
        elevation:      (n_time,)
        """
        return tod - (atmos_offsets[..., None] + atmos_tau[..., None] / np.sin(np.radians(elevation)))

    def read_data(self, level1_file : str, level2_file : str, line_frequency : float, line_segment_width : int, database, use_flags : bool = False, bad_feeds : list = []) -> tuple:
        """Read a Level 1/Level 2 file pair and return a list of TOD segments."""
        tod_blocks, weight_blocks, ra_blocks, dec_blocks = [], [], [], []

        with h5py.File(level1_file, 'r') as f_lvl1, h5py.File(level2_file, 'r') as f_lvl2:
            frequencies = f_lvl1['spectrometer/frequency'][...]
            band, channel_start, channel_end = self.select_band(frequencies, line_frequency, line_segment_width)
            if band is None:
                return np.array([], dtype=np.float32),np.array([], dtype=np.float32),np.array([], dtype=np.float32),np.array([], dtype=np.float32)

            obsid = int(os.path.basename(level2_file).split('-')[1])
            feeds = f_lvl2['spectrometer/feeds'][:-1] 
            scan_edges = f_lvl2['level2/scan_edges'][...] 

            for ifeed, feed in enumerate(feeds): 
                if feed in bad_feeds:
                    continue

                for iscan, (start, end) in enumerate(scan_edges):
                    logging.info(f'Obsid {obsid} Feed {feed} Band {band} Scan {iscan} Start {start} End {end}')
                    if self.flags_are_bad(obsid, feed, band, iscan, f_lvl2, database):
                        continue

                    length = (end - start)//self.data_container.offset_length * self.data_container.offset_length
                    end = int(start + length)
                    if length <= 0:
                        continue



                    # Define noise weights 
                    auto_rms = f_lvl2['level2_noise_stats/binned_filtered_data/auto_rms'][feed-1, band, self.statistics_channel, iscan]
                    # Guard against zero/NaN
                    if not np.isfinite(auto_rms) or auto_rms <= 0:
                        continue
                    weights_seg = np.ones((line_segment_width, length), dtype=np.float64) / (auto_rms ** 2)

                    # --- Coordinates
                    az = f_lvl2['spectrometer/pixel_pointing/pixel_az'][ifeed, start:end]  # ifeed!
                    el = f_lvl2['spectrometer/pixel_pointing/pixel_el'][ifeed, start:end]
                    mjd = f_lvl2['spectrometer/MJD'][start:end]
                    ra, dec = h2e_full(az, el, mjd, comap_longitude, comap_latitude)

                    # Calibrate the Level 1 data 
                    tod_full = f_lvl1['spectrometer/tod'][ifeed, band, :, start:end]
                    gain = f_lvl2['level2/vane/gain'][0,ifeed,band,:] 
                    tsys = f_lvl2['level2/vane/system_temperature'][0,ifeed,band,:]
                    gsafe = np.where(np.isfinite(gain) & (gain != 0), gain, np.nan)
                    tod_full = tod_full / gsafe[:, None]


                    # Apply atmosphere correction
                    atmos_offsets = f_lvl2['level2/atmosphere/offsets'][feed, band, :, iscan]
                    atmos_tau = f_lvl2['level2/atmosphere/tau'][feed, band, :, iscan]
                    tod_full = self.apply_atmosphere_correction(tod_full, atmos_offsets, atmos_tau, el)


                    # Apply gain filter to the TOD 
                    median_offsets = np.nanmedian(tod_full, axis=1)
                    residual, mask, gains = self.gain_filter(tod_full[np.newaxis],
                                                                tsys[np.newaxis],
                                                                median_offsets[np.newaxis])

                    tod_full = residual[0]  # (n_chan, length)
                    tod_seg = tod_full[channel_start:channel_end, :]  # (line_segment_width, length)

                    # Apply any additional filters here, e.g.,
                    # tod_seg = self.frequency_filter(tod_seg, *args, **kwargs) 


                    #  Clean NaNs/Infs
                    bad = ~np.isfinite(tod_seg) | ~np.isfinite(weights_seg)
                    if np.any(bad):
                        tod_seg = np.where(bad, 0.0, tod_seg)
                        weights_seg = np.where(bad, 0.0, weights_seg)

                    #  Append blocks
                    tod_blocks.append(tod_seg.astype(np.float32))
                    weight_blocks.append(weights_seg.astype(np.float32))
                    ra_blocks.append(ra.astype(np.float64))
                    dec_blocks.append(dec.astype(np.float64))

        # --- Concatenate across scans/feeds on the time axis
        if len(tod_blocks) == 0:
            return (np.array([], dtype=np.float32),
                    np.array([], dtype=np.float32),
                    np.array([], dtype=np.float32),
                    np.array([], dtype=np.float32))

        try:
            tod = np.concatenate(tod_blocks, axis=1).astype(np.float32)
            weights = np.concatenate(weight_blocks, axis=1).astype(np.float32)
            x = np.concatenate(ra_blocks).astype(np.float32)
            y = np.concatenate(dec_blocks).astype(np.float32)
        except ValueError:
            # Inconsistent shapes—return empty to be safe
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
        # pyplot.plot(np.nanmedian(tod,axis=0))
        # pyplot.savefig('test.png')
        # pyplot.close()
        # pyplot.plot(np.nanmedian(weights,axis=0))
        # pyplot.savefig('test_weights.png')
        # import sys 
        # sys.exit()

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