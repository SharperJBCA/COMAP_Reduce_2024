# ReadData_rewrite.py
#
# Streaming reader for COMAP Level-2 -> map-making with dual-weight paths:
# - solver weights (may be Planck-downweighted) for destriping
# - map weights (never downweighted) for FITS outputs
#
from __future__ import annotations
import os
import logging
from dataclasses import dataclass
from typing import Iterable, Sequence, Optional, Tuple
from contextlib import suppress
import numpy as np
import h5py
import healpy as hp
import toml
from tqdm import tqdm 
from astropy.io.fits.header import Header
from astropy.wcs import WCS

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# --- project deps
from modules.utils import bin_funcs
from modules.utils.median_filter import medfilt
try:
    from modules.utils.Coordinates import h2e_full, comap_longitude, comap_latitude
except ModuleNotFoundError:
    from modules.utils.Coordinates_py import h2e_full, comap_longitude, comap_latitude


def sum_map_all_inplace(m: np.ndarray) -> np.ndarray:
    """In-place MPI allreduce sum, supports float32/64."""
    mpi_type = MPI.DOUBLE if m.dtype == np.float64 else MPI.FLOAT
    comm.Allreduce(MPI.IN_PLACE, [m, mpi_type], op=MPI.SUM)
    return m


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

def tpoint_model(params, A, E):
    """
    A, E in radians.
    params = [IA, IE, NPAE, AW, AN, CA, offset]  # or whatever you ended up with
    Returns dA, dE in radians (actual - nominal).
    """
    IA, IE, NPAE, AW, AN, CA, offset = params

    dA_cosE = IA + NPAE*np.tan(E) + AW*np.cos(A)*np.tan(E) + AN*np.sin(A)*np.tan(E)
    dE      = IE + CA*np.cos(E)  + AW*np.sin(A + offset) + AN*np.cos(A + offset)
    dA      = dA_cosE / np.cos(E)

    return dA, dE


def pointing_offsets_deg(az_deg, el_deg, params):
    """
    az_deg, el_deg: arrays in degrees (nominal pointing)
    params: fitted TPOINT params
    Returns (az_offset_deg, el_offset_deg) to *add* to az/el.
    """
    A = np.deg2rad(az_deg)
    E = np.deg2rad(el_deg)

    dA_rad, dE_rad = tpoint_model(params, A, E)

    az_off_deg = np.rad2deg(dA_rad)
    el_off_deg = np.rad2deg(dE_rad)
    return az_off_deg, el_off_deg



@dataclass
class Level2Data:
    """Minimal state needed downstream by the destriper and writer."""
    offset_length: int
    # Per-pixel aggregates (for output maps)
    sum_map:    np.ndarray
    weight_map: np.ndarray
    hits_map:   np.ndarray
    sky_map:    np.ndarray
    # Per-sample, kept only for solver
    pixels:         np.ndarray    # int64, length = n_tod
    weights: np.ndarray    # float32, length = n_tod
    # Right-hand side for solver
    rhs: np.ndarray               # float32, length = n_offsets

def _legendre_cols(xn, deg):
    """Legendre columns L0..Ldeg on x normalized to [-1,1]."""
    cols = [np.ones_like(xn)]
    if deg >= 1: cols.append(xn)
    for n in range(2, deg+1):
        # P_n(x) via recurrence: ( (2n-1)x P_{n-1} - (n-1) P_{n-2} ) / n
        Pn_1, Pn_2 = cols[-1], cols[-2]
        Pn = ((2*n-1)*xn*Pn_1 - (n-1)*Pn_2) / n
        cols.append(Pn)
    return cols  # list of arrays

def _normalize_to_unit(x):
    """Return x in [-1,1], span; if span=0 -> zeros."""
    x = np.asarray(x)
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    span = float(xmax - xmin)
    if not np.isfinite(span) or span <= 0:
        return np.zeros_like(x, dtype=np.float64), 0.0
    xn = (x - (xmin + xmax)/2.0) / (span/2.0)
    return xn.astype(np.float64, copy=False), span

def _irls_wls(X, y, w, max_iter=5, huber_k=1.345, clip_sigma=4.0, mask=None):
    """
    Robust WLS via IRLS with Huber weights and sigma clipping on residuals.
    X: (n,m) float64, y: (n,), w: base weights (n,)
    mask: boolean, True keeps points in the fit (e.g., Planck-quiet & high-|b|)
    Returns beta (m,), y_fit (n,)
    """
    n, m = X.shape
    if mask is None:
        mask = np.ones(n, dtype=bool)
    y = y.astype(np.float64, copy=False)
    w = w.astype(np.float64, copy=False)

    # initial fit mask
    mfit = mask & np.isfinite(y) & (w > 0)
    if mfit.sum() < max(m+2, 32):
        return np.zeros(m), np.zeros_like(y)

    beta = np.zeros(m)
    for _ in range(max_iter):
        # Weighted LS on current mask
        Xw = X[mfit] * np.sqrt(w[mfit])[:, None]
        yw = y[mfit] * np.sqrt(w[mfit])
        # solve in float64, fallback safe
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        r = y - X @ beta
        # robust weights (Huber)
        # scale estimate: MAD -> sigma
        mad = np.nanmedian(np.abs(r[mfit] - np.nanmedian(r[mfit]))) * 1.4826
        sigma = mad if np.isfinite(mad) and mad > 0 else np.sqrt(np.average(r[mfit]**2, weights=w[mfit]))
        if not np.isfinite(sigma) or sigma <= 0:
            break
        t = r / (huber_k * sigma)
        hub = np.ones_like(t)
        mask_hard = np.abs(t) > 1.0
        hub[mask_hard] = 1.0 / np.abs(t[mask_hard])
        # update fit mask with sigma-clip (protect against wild points)
        mfit = mask & (np.abs(r) < clip_sigma * sigma) & np.isfinite(y) & (w > 0)
        # incorporate robust reweighting
        w = w * hub
        if mfit.sum() < max(m+2, 32):
            break

    y_fit = X @ beta
    return beta, y_fit

def remove_trends_from_scan(
    tod, w_solver, *, time_idx, el, az_deg, gl, gb,
    planck_map=None,
    deg_time=2, deg_el=3, deg_az=2, az_fourier_k=0,
    min_span_time=0.05, min_span_el=0.1, min_span_az_deg=5.0,
    use_time=True, use_el=True, use_az=True,
    highlat_cut=0.75,             # |b|>cut included even without Planck
    planck_quiet_cut=0.03,        # K_RJ at 30 GHz for "quiet"
):
    """
    Build a robust, weighted trend model of the TOD and subtract it.
    Returns detrended_tod (float32).
    """
    n = tod.size
    # ---- masks for 'safe sky' used in fitting the trend ----
    if planck_map is not None:
        hpx = hp.ang2pix(1024, np.radians(90.0 - gb), np.radians(gl))
        quiet = planck_map[hpx] < planck_quiet_cut
    else:
        quiet = np.ones(n, dtype=bool)

    highlat = np.abs(gb) > highlat_cut
    fit_mask = quiet | highlat

    # ---- build design matrix ----
    cols = [np.ones(n, dtype=np.float64)]  # intercept always
    # time
    if use_time:
        tn = (time_idx - time_idx.min()) / max(1, (time_idx.max() - time_idx.min()))
        tn = 2.0*tn - 1.0  # to [-1,1]
        span_t = float(tn.max() - tn.min())
        if span_t >= min_span_time and deg_time >= 1:
            cols.extend(_legendre_cols(tn, deg_time)[1:])  # skip constant (already in)
    # elevation
    if use_el:
        eln, span_el = _normalize_to_unit(el)
        if span_el >= min_span_el and deg_el >= 1:
            cols.extend(_legendre_cols(eln, deg_el)[1:])
    # azimuth (polynomial basis)
    if use_az:
        azn, span_az = _normalize_to_unit(az_deg)
        if span_az >= min_span_az_deg and deg_az >= 1:
            cols.extend(_legendre_cols(azn, deg_az)[1:])

    X = np.vstack(cols).T  # (n, m)

    # ---- robust weighted fit on fit_mask only ----
    beta, y_fit = _irls_wls(X, tod, w_solver, mask=fit_mask)

    # ---- subtract trend ----
    tod_detr = tod.astype(np.float32, copy=True)
    tod_detr -= y_fit.astype(np.float32, copy=False)
    return tod_detr

class Level2DataReader:
    """
    Reader that streams L2 HDF5, applies light filtering, builds:
      - per-pixel maps (sum/weight/hits) from *map* weights (no Planck downweight)
      - per-sample arrays (pixels, weights_solver) for destriping operator
      - RHS vector for destriping (built with solver weights)
    """
    def __init__(
        self,
        *,
        tod_dataset: str = "level2/binned_filtered_data",
        offset_length: int = 100,
        band_index: int = 0,
        channel_shape: Tuple[int, int] = (4, 2),
        # thresholds / knobs
        sigma_red_cutoff: float = 0.4,     # scans with sigma_red above this are dropped
        auto_rms_cap: float = 1e-1,        # scans with auto_rms above this are dropped
        spike_sigma_strong: float = 15.0,  # |tod| > this * auto_rms are zeroed
        spike_sigma_resid: float = 10.0,    # |resid| > this * auto_rms after median filter are zeroed
        resid_dilate: int = 201,           # length of 1D dilation for residual spikes
        medfilt_len: int = 100,            # median filter length for residual
        # Planck masking
        planck_30_path: Optional[str] = None,
        planck_quiet_cut_strong: float = 0.04,  # only zero strong spikes where Planck map < cut
        planck_quiet_cut_soft: float = 0.045,   # only downweight solver weights where Planck map < cut? (we use > cut to downweight)
        planck_downweight: float = 0.1,         # factor applied to solver weights in bright Planck regions
        # Calibration
        calib_path: Optional[str] = None,       # TauA calibration .npy (dict-like)
        calib_min_factor: float = 0.3,
        # Coordinates
        input_coord: str = "C",
        output_coord: str = "G",
        # WCS def file
        wcs_config_file: str = "map_making_configs/wcs_definitions.toml",
        # DB handle (optional)
        database=None,
        jackknife: str = None,   # None | 'odd' | 'even'
        jackknife_reset: str = "per_file_feed", 
    ):
        self.tod_ds = tod_dataset
        self.offset_length = int(offset_length)
        self.band_index = int(band_index)
        self.band, self.channel = np.unravel_index(self.band_index, channel_shape)

        # thresholds
        self.sigma_red_cutoff = float(sigma_red_cutoff)
        self.auto_rms_cap = float(auto_rms_cap)
        self.spike_sigma_strong = float(spike_sigma_strong)
        self.spike_sigma_resid = float(spike_sigma_resid)
        self.resid_dilate = int(resid_dilate)
        self.medfilt_len = int(medfilt_len)

        # planck
        with suppress(ValueError,KeyError):
            self.planck_30_map = hp.read_map(planck_30_path, verbose=False) if planck_30_path else None
        self.planck_quiet_cut_strong = float(planck_quiet_cut_strong)
        self.planck_quiet_cut_soft = float(planck_quiet_cut_soft)
        self.planck_downweight = float(planck_downweight)

        # calibration
        self.calib = np.load(calib_path, allow_pickle=True).item() if calib_path else None
        self.calib_min_factor = float(calib_min_factor)

        # coords / wcs
        self.input_coord = input_coord
        self.output_coord = output_coord
        self.wcs_config_file = wcs_config_file
        self.wcs_def = None
        self.header: Optional[Header] = None
        self.wcs: Optional[WCS] = None
        self._nx = 0
        self._ny = 0
        self._npix = 0

        # DB
        self.database = database

        # data container
        self.data = Level2Data(
            offset_length=self.offset_length,
            sum_map=np.empty(0, np.float32),
            weight_map=np.empty(0, np.float32),
            hits_map=np.empty(0, np.float32),
            sky_map=np.empty(0, np.float32),
            pixels=np.empty(0, np.int64),
            weights=np.empty(0, np.float32),
            rhs=np.empty(0, np.float32),
        )

        # cursors for preallocated arrays
        self._cur_tod = 0
        self._cur_off = 0
        self._n_tod = 0
        self._n_off = 0

        # jack knife keys 
        self.jackknife = (jackknife.lower() if jackknife is not None else None)
        if self.jackknife not in (None, 'odd', 'even'):
            raise ValueError("jackknife must be None, 'odd', or 'even'")
        self.jackknife_reset = jackknife_reset


        self.pointing_params = np.loadtxt('/scratch/nas_comap1/sharper/student_pipeline_test/COMAP_Reduce_2024/notebooks/pointing_params.txt',delimiter=',')

    # ---------------- WCS ----------------

    def setup_wcs(self, wcs_def: str):
        self.wcs_def = wcs_def
        cfg = toml.load(self.wcs_config_file)[wcs_def]
        self.header = Header(cfg)
        self.wcs = WCS(self.header)
        self._ny = int(self.header["NAXIS2"])
        self._nx = int(self.header["NAXIS1"])
        self._npix = self._nx * self._ny
        self.data.sum_map = np.zeros(self._npix, np.float32)
        self.data.weight_map = np.zeros(self._npix, np.float32)
        self.data.hits_map = np.zeros(self._npix, np.float32)
        self.data.sky_map = np.zeros(self._npix, np.float32)

    # ------------- counting / prealloc -------------

    def _count_in_file(self, file: str, feeds_keep: Sequence[int], use_flags: bool) -> Tuple[int, int]:
        """Return (n_samples_kept, n_offsets_kept) after truncation to multiples of L and high-level skips."""
        L = self.offset_length
        nsamp = 0
        noff = 0
        with h5py.File(file, "r") as f:
            scan_edges = f["level2/scan_edges"][...]
            feeds = f["spectrometer/feeds"][:-1]
            # early skip via feed 1 high sigma_red
            s1 = f["level2_noise_stats/binned_filtered_data/sigma_red"][0, self.band, self.channel, :]
            if np.any(s1 > 5.0):
                return 0, 0
            for feed in feeds:
                if feed not in feeds_keep:
                    continue
                flags = (np.ones(len(scan_edges), bool) if not use_flags or self.database is None
                         else self.database.query_scan_flags(int(os.path.basename(file).split('-')[1]), feed, self.band))
                for flag, (start, end) in zip(flags, scan_edges):
                    if not flag:
                        continue
                    length = (end - start) // L * L
                    if length <= 0:
                        continue
                    nsamp += length
                    noff += length // L
        return nsamp, noff

    def _preallocate(self, files: Sequence[str], feeds_keep: Sequence[int], use_flags: bool):
        self._n_tod = 0
        self._n_off = 0
        for file in files:
            ns, no = self._count_in_file(file, feeds_keep, use_flags)
            self._n_tod += ns
            self._n_off += no
        # per-sample arrays only for solver
        self.data.pixels = np.zeros(self._n_tod, np.int64)
        self.data.weights = np.zeros(self._n_tod, np.float32)
        # rhs for solver
        self.data.rhs = np.zeros(self._n_off, np.float32)
        self._cur_tod = 0
        self._cur_off = 0
        if rank == 0:
            logging.info(f"[reader] prealloc nsamp={self._n_tod} n_off={self._n_off}")

    # ------------- coordinate transforms -------------

    def _coordinate_transform(self, lon_in: np.ndarray, lat_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.input_coord == self.output_coord:
            return lon_in, lat_in
        rot = hp.rotator.Rotator(coord=[self.input_coord, self.output_coord])
        lon, lat = rot(lon_in, lat_in, lonlat=True)
        return np.mod(lon, 360.0), lat

    def _coords_to_pixels(self, gl: np.ndarray, gb: np.ndarray) -> np.ndarray:
        xpix, ypix = self.wcs.all_world2pix(gl, gb, 0)
        m = (xpix >= 0) & (xpix < self._nx) & (ypix >= 0) & (ypix < self._ny)
        p = np.full(gl.shape, -1, np.int64)
        if m.any():
            p[m] = np.ravel_multi_index((ypix[m].astype(int), xpix[m].astype(int)), (self._ny, self._nx))
        return p

    # ---------------- calibration ----------------

    def _calibration_factor(self, feed: int, mjd0: float) -> Optional[float]:
        if self.calib is None:
            return 1.0
        params = self.calib["model_fits"][feed - 1, self.band, self.channel]
        phase = self.calib["best_phase"]
        period = self.calib["best_period"]
        amp, grad, off = params
        val = amp * np.sin(2 * np.pi * (mjd0 - 59000.0 - phase) / period) + grad * (mjd0 - 59000.0) / 365.25 + off
        if not np.isfinite(val) or val < self.calib_min_factor:
            return None
        return float(val)

    # ------------- per-scan processing -------------

    def _process_scan(self, f, ifeed: int, feed: int, start: int, end: int, mjd0: float) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Returns (tod, weights_solver, gl, gb) for this scan (length is multiple of L).
        We modify weights for the solver only; map weights are derived later.
        """
        L = self.offset_length
        length = (end - start) // L * L
        if length <= 0:
            return None

        tod = f[self.tod_ds][feed - 1, self.band, self.channel, start:start + length].astype(np.float32, copy=False)
        tod -= np.nanmedian(tod)

        # per-scan noise proxies
        sigma_red_scan = f["level2_noise_stats/binned_filtered_data/sigma_red"][feed - 1, self.band, self.channel, :]
        auto_rms_scan = f["level2_noise_stats/binned_filtered_data/auto_rms"][feed - 1, self.band, self.channel, :]

        # basic checks
        auto_rms = np.nanmedian(auto_rms_scan)
        if not np.isfinite(auto_rms) or auto_rms <= 0:
            return None
        if auto_rms > self.auto_rms_cap:
            return None
        if np.nanmax(sigma_red_scan) > self.sigma_red_cutoff:
            return None

        # calibration (optional)
        cf = self._calibration_factor(feed, float(f["spectrometer/MJD"][0]))
        if cf is None:
            return None
        tod /= cf
        auto_rms /= cf  # if the calibration scales the whole timestream, keep thresholds consistent

        # base weights (Gaussian approx)
        w_solver = np.full_like(tod, 1.0 / (auto_rms * auto_rms), dtype=np.float32)

        # coordinates
        mjd = f["spectrometer/MJD"][start:start + length]
        az = f["spectrometer/pixel_pointing/pixel_az"][ifeed, start:start + length]
        el = f["spectrometer/pixel_pointing/pixel_el"][ifeed, start:start + length]
        # Apply offset corrections 
        #az_offset_deg, el_offset_deg = pointing_offsets_deg(az, el, self.pointing_params[ifeed])
        #az = az - az_offset_deg
        #el = el - el_offset_deg

        ra, dec = h2e_full(az, el, mjd, comap_longitude, comap_latitude)
        gl, gb = self._coordinate_transform(ra, dec)

        # spike handling and residual filtering
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            # strong spikes where Planck is quiet
            if self.planck_30_map is not None:
                hpx = hp.ang2pix(1024, np.radians(90.0 - gb), np.radians(gl))
                quiet = self.planck_30_map[hpx] < self.planck_quiet_cut_strong
                mspk = (np.abs(tod) > self.spike_sigma_strong * auto_rms) & quiet
                tod[mspk] = 0.0
                w_solver[mspk] = 0.0

            # median residual spikes
            resid = tod - np.array(medfilt.medfilt(tod.astype(np.float64), self.medfilt_len)).astype(np.float32)
            mspk2 = np.abs(resid) > (self.spike_sigma_resid * auto_rms)
            if mspk2.any():
                kernel = np.ones(self.resid_dilate, dtype=np.int8)
                mspk2 = np.convolve(mspk2.astype(np.int8), kernel, mode="same") > 0
                tod[mspk2] = 0.0
                w_solver[mspk2] = 0.0

        time_idx = np.arange(len(tod), dtype=np.float64)
        not_zero = (tod != 0) & (w_solver != 0)
        n_not_zero = np.sum(not_zero) 
        if n_not_zero < 0.95*tod.size:
            tod[...] = 0.0
            w_solver[...] = 0.0
        else:
            #from matplotlib import pyplot 
            #coord = el 
            #pyplot.plot(coord,tod*1,'.')
            tod[not_zero] = remove_trends_from_scan(
                tod[not_zero] , w_solver[not_zero] ,
                time_idx=time_idx[not_zero] ,
                el=el[not_zero] , az_deg=az[not_zero] ,
                gl=gl[not_zero] , gb=gb[not_zero] ,
                planck_map=self.planck_30_map,
                deg_time=3, deg_el=3, deg_az=3,     # safe defaults
                min_span_time=0.05,                  # require >5% range
                min_span_el=0.1,                     # ~0.1 deg in elevation
                min_span_az_deg=0.1,
                use_time=True, use_el=True, use_az=True,
                highlat_cut=0.75,
                planck_quiet_cut=0.03,
            )
            # pyplot.plot(coord,tod*1,'.')
            # nbins = 10
            # azedge = np.linspace(np.min(coord),np.max(coord),nbins+1)
            # azmids = (azedge[1:] + azedge[:-1])/2.
            # azbin = np.histogram(coord, azedge, weights=tod)[0]/np.histogram(coord, azedge)[0] 
            # pyplot.plot(azmids, azbin,lw=2)
            # pyplot.text(0.05,0.9,np.max(azbin)-np.min(azbin),transform=pyplot.gca().transAxes)

            # pyplot.savefig(f'Feed{feed:02d}.png')
            # pyplot.close()


        # NaNs → zeroed with zero weight
        bad = ~np.isfinite(tod)
        if bad.any():
            tod[bad] = 0.0
            w_solver[bad] = 0.0

        # planck-based **regularisation** ONLY for solver weights (not map weights)
        if self.planck_30_map is not None:
            hpx = hp.ang2pix(1024, np.radians(90.0 - gb), np.radians(gl))
            bright = self.planck_30_map[hpx] > self.planck_quiet_cut_soft
            # downweight *but do not zero* unless already spiky
            w_solver[bright] *= np.float32(self.planck_downweight)

        return tod, w_solver, gl, gb

    # ------------- accumulate one file -------------

    def _accumulate_file(self, file: str, feeds_keep: Sequence[int], use_flags: bool):
        obsid = int(os.path.basename(file).split("-")[1])
        if self.database: 
            quality_flags = self.database.get_quality_flags(obsid)
        else:
            quality_flags = None

        with h5py.File(file, "r") as f:
            scan_edges = f["level2/scan_edges"][...]
            feeds = f["spectrometer/feeds"][:-1]

            # quick global skip by feed 1
            s1 = f["level2_noise_stats/binned_filtered_data/sigma_red"][0, self.band, self.channel, :]
            if np.any(s1 > 5.0):
                if rank == 0:
                    logging.info(f"[reader] skip obsid={obsid} (feed1 high sigma_red)")
                return

            for ifeed, feed in enumerate(feeds):
                if quality_flags:
                    if not (feed, self.band) in quality_flags:
                        continue
                    if not quality_flags[(feed, self.band)].is_good:
                        continue

                if feed not in feeds_keep:
                    continue
                flags = (np.ones(len(scan_edges), bool) if not use_flags or self.database is None
                         else self.database.query_scan_flags(obsid, feed, self.band))
                good_scan_index = 0
                for flag, (start, end) in zip(flags, scan_edges):
                    if not flag:
                        continue

                    group_is_even = (good_scan_index % 2 == 0)
                    group = 'even' if group_is_even else 'odd'
                    good_scan_index += 1  

                    # If jackknife requested, drop non-matching group early
                    if self.jackknife is not None and group != self.jackknife:
                        continue

                    out = self._process_scan(f,ifeed, feed, int(start), int(end), float(f["spectrometer/MJD"][0]))
                    if out is None:
                        continue


                    tod, w_solver, gl, gb = out

                    # pixels
                    p = self._coords_to_pixels(gl, gb)

                    # zero invalid samples for both paths
                    inv = (p == -1) | (w_solver <= 0.0)
                    if inv.any():
                        tod[inv] = 0.0
                        w_solver[inv] = 0.0

                    if self.planck_30_map is not None:
                        hpx = hp.ang2pix(1024, np.radians(90.0 - gb), np.radians(gl))
                        bright = self.planck_30_map[hpx] > self.planck_quiet_cut_soft
                        # compute w_map by "undoing" the downweight where it was applied
                        w_map = w_solver.copy()
                        w_map[bright] = np.where(
                            self.planck_downweight > 0.0,
                            w_map[bright] / np.float32(self.planck_downweight),
                            w_map[bright],
                        )
                    else:
                        w_map = w_solver  # identical if no planck regularisation

                    # Bin to per-pixel maps using w_map 
                    bin_funcs.bin_tod_to_map(self.data.sum_map, self.data.weight_map, self.data.hits_map,
                                             tod, w_solver, p)

                    # ----- solver path: rhs + store arrays -----
                    rhs_scan = np.zeros(len(tod) // self.offset_length, np.float32)
                    bin_funcs.bin_tod_to_rhs(rhs_scan, tod, w_solver, self.offset_length)
                    n_off = rhs_scan.size
                    self.data.rhs[self._cur_off:self._cur_off + n_off] = rhs_scan
                    self._cur_off += n_off

                    n_samp = len(tod)
                    s = slice(self._cur_tod, self._cur_tod + n_samp)
                    self.data.pixels[s] = p
                    self.data.weights[s] = w_solver
                    self._cur_tod += n_samp

    # ------------- top-level -------------

    def read_files(self, files: Sequence[str], *, feeds: Optional[Sequence[int]] = None, use_flags: bool = False) -> bool:
        if self.wcs is None:
            raise RuntimeError("Call setup_wcs(...) before read_files().")

        feeds_keep = list(range(1, 20)) if feeds is None else list(feeds)
        if not files:
            if rank == 0:
                logging.warning("[reader] no files to read.")
            return False

        # prealloc (rank-local; per-rank nsamp differs)
        self._preallocate(files, feeds_keep, use_flags)

        if rank == 0:
            files = tqdm(files) 
        # stream through files
        for i, file in enumerate(files):
            if rank == 0 and (i % max(1, len(files) // 10) == 0):
                logging.info(f"[reader] {i}/{len(files)}")
            self._accumulate_file(file, feeds_keep, use_flags)

        # MPI reduce the maps
        sum_map_all_inplace(self.data.sum_map)
        sum_map_all_inplace(self.data.weight_map)
        sum_map_all_inplace(self.data.hits_map)

        # sky_map for later Z subtraction
        m = self.data.weight_map > 0
        self.data.sky_map[:] = 0.0
        self.data.sky_map[m] = self.data.sum_map[m] / self.data.weight_map[m]

        # Destriper RHS needs Z d = F^T N^{-1} P m  subtraction using *solver* weights
        bin_funcs.subtract_sky_map_from_rhs(self.data.rhs,
                                            self.data.weights,
                                            self.data.sky_map,
                                            self.data.pixels,
                                            self.offset_length)
        return True
