# CalibratorFitting.py
#
# Description:
#  Implements fitting routines for TauA/CasA/CygA/Jupiter/etc... calibration observations 
# 
# History:
#  24/03/25 - SH: Initial version based on original pipeline implementation. 

import numpy as np
import numpy.ma as ma
from tqdm import tqdm
import h5py 
from itertools import product 
import sys 
import healpy as hp
from scipy.ndimage import gaussian_filter
from astropy.modeling import models, fitting
import astropy.units as u
from astropy.modeling.functional_models import Gaussian2D
from astropy.modeling.models import custom_model
from matplotlib import pyplot 
from astroquery.jplhorizons import Horizons
import time 
from datetime import datetime
import logging 
from scipy.stats import binned_statistic_2d
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz, EarthLocation


try:
    from modules.utils.data_handling import read_2bit
    from modules.SQLModule.SQLModule import COMAPData, db
    from modules.pipeline_control.Pipeline import BadCOMAPFile, RetryH5PY,BaseCOMAPModule
    try:
        from modules.utils.Coordinates import comap_longitude, comap_latitude, h2e_full, CalibratorList, pa
    except:
        from modules.utils.Coordinates_py import comap_longitude, comap_latitude, h2e_full, CalibratorList, pa

except ModuleNotFoundError:
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))    
    from modules.utils.data_handling import read_2bit
    from modules.SQLModule.SQLModule import COMAPData,db
    from modules.pipeline_control.Pipeline import BadCOMAPFile, RetryH5PY,BaseCOMAPModule
    try:
        from modules.utils.Coordinates import comap_longitude, comap_latitude, h2e_full, CalibratorList, pa
    except:
        from modules.utils.Coordinates_py import comap_longitude, comap_latitude, h2e_full, CalibratorList, pa




def _get_body_radec_from_mjd(mjd, target_id="599"):
    """
    Generic helper: get RA/Dec (ICRF, deg) for a solar-system body
    at a given UTC MJD using JPL Horizons (via astroquery).

    target_id:
        '599' = Jupiter, '699' = Saturn, etc.
    """
    # Horizons wants JD, not MJD
    jd = mjd + 2400000.5

    location = {'lon':comap_longitude*u.deg,
                'lat':comap_latitude*u.deg,
                'elevation':1222.0 * u.m}

    # 500 = geocentric observer
    obj = Horizons(id=target_id,
                   location='500',   # Earth geocentre
                   epochs=jd)
    print(obj)
    eph = obj.ephemerides(quantities=1)  # RA/Dec in degrees (ICRF)

    ra = float(eph['RA'][0])   # deg
    dec = float(eph['DEC'][0]) # deg
    return ra, dec

def get_jupiter_radec_from_mjd(mjd):
    return _get_body_radec_from_mjd(mjd, target_id='599')


def fit_gaussian2d_with_errors(x, y, z, sigma=None):
    amp0 = np.nanmax(z) - np.nanmedian(z)
    i0   = int(np.nanargmax(z))
    x0   = x[i0]; y0 = y[i0]
    sig0 = (4.5/60.)/2.355  # degrees
    theta0 = 0.0

    g0 = models.Gaussian2D(amplitude=amp0, x_mean=x0, y_mean=y0,
                           x_stddev=sig0, y_stddev=sig0, theta=theta0)
    g0.theta.fixed = True

    bg0 = models.Const2D(amplitude=np.nanmedian(z))
    model0 = g0 + bg0

    fitter = fitting.LevMarLSQFitter(calc_uncertainties=True)

    if sigma is not None:
        # LevMarLSQFitter expects weights = 1/σ
        w = np.ones_like(z, dtype=float) / sigma
        best = fitter(model0, x, y, z, weights=w)
    else:
        best = fitter(model0, x, y, z)

    cov = fitter.fit_info.get('param_cov')
    if cov is not None and np.all(np.isfinite(cov)):
        errs = np.sqrt(np.diag(cov))
    else:
        errs = np.full(best.parameters.size, np.nan)

    p = best.parameters
    p_gauss = p[:6]
    e_gauss = errs[:6] if errs is not None else np.full(6, np.nan)
    return p_gauss, e_gauss, best

class CalibratorFitting(BaseCOMAPModule):

    def __init__(self, calibrator_fit_group_name='calibrator_fit', overwrite=False, fit_radius=15./60.): 
        super().__init__()

        self.overwrite = overwrite 
        self.calibrator_fit_group_name = calibrator_fit_group_name
        self.fit_radius = fit_radius

    def already_processed(self, file_info : COMAPData, overwrite : bool = False) -> bool:
        """
        Check if the gain filter and bin has already been applied 
        """
        if overwrite:
            return False
    
        with RetryH5PY(file_info.level2_path, 'r') as ds: 
            if f'level2/{self.calibrator_fit_group_name}' in ds:
                return True
            return False
        
    def save_data(self, file_info : COMAPData) -> None:
        with RetryH5PY(file_info.level2_path, 'a') as ds:
            if not 'level2' in ds:
                ds.create_group('level2')
            grp = ds['level2']
            if f'{self.calibrator_fit_group_name}' in grp:
                del grp[f'{self.calibrator_fit_group_name}']
            grp.create_group(f'{self.calibrator_fit_group_name}')
            grp = grp[f'{self.calibrator_fit_group_name}']

            print('creating group', grp.name) 
            print('creating amplitudes')
            ds = grp.create_dataset('amplitudes', data=self.best_fits[0])
            ds.attrs['unit'] = 'K'
            ds = grp.create_dataset('amplitude_errors', data=self.errs_fits[0])
            ds.attrs['unit'] = 'K'

            print('creating offsets')
            ds = grp.create_dataset('ra_offset', data=self.best_fits[1])
            ds.attrs['unit'] = 'degrees'
            ds = grp.create_dataset('ra_offset_errors', data=self.errs_fits[1]) 
            ds.attrs['unit'] = 'degrees'

            ds = grp.create_dataset('dec_offset', data=self.best_fits[2])
            ds.attrs['unit'] = 'degrees'
            ds = grp.create_dataset('dec_offset_errors', data=self.errs_fits[2])
            ds.attrs['unit'] = 'degrees'


            ds = grp.create_dataset('az_offset', data=self.delta_az[0])
            ds.attrs['unit'] = 'degrees'
            ds = grp.create_dataset('az_offset_errors', data=self.delta_az[1])
            ds.attrs['unit'] = 'degrees'

            ds = grp.create_dataset('el_offset', data=self.delta_el[0])
            ds.attrs['unit'] = 'degrees'
            ds = grp.create_dataset('el_offset_errors', data=self.delta_el[1]) 
            ds.attrs['unit'] = 'degrees'

            ds = grp.create_dataset('major_std', data=self.best_fits[3])
            ds.attrs['unit'] = 'degrees'
            ds = grp.create_dataset('major_std_errors', data=self.errs_fits[3])
            ds.attrs['unit'] = 'degrees'

            ds = grp.create_dataset('minor_std', data=self.best_fits[4])
            ds.attrs['unit'] = 'degrees'
            ds = grp.create_dataset('minor_std_errors', data=self.errs_fits[4])
            ds.attrs['unit'] = 'degrees'

            ds = grp.create_dataset('position_angle', data=self.best_fits[5])
            ds.attrs['unit'] = 'degrees'
            ds = grp.create_dataset('position_angle_errors', data=self.errs_fits[5])
            ds.attrs['unit'] = 'degrees'

            ds = grp.create_dataset('chi2', data=self.chi2_fits)

            ds = grp.create_dataset('flux', data=self.get_flux_density(self.best_fits))
            ds.attrs['unit'] = 'Jy'
            ds = grp.create_dataset('flux_errors', data=self.get_flux_density_errors(self.best_fits, self.errs_fits))
            ds.attrs['unit'] = 'Jy'

    def get_flux_density(self, best_fits : np.ndarray) -> np.ndarray:
        """
        Get the flux density from the best fits
        """
        k = 1.380649e-23  # Boltzmann constant in J/K
        c = 2.99792458e8  # Speed of light in m/s
        frequencies = np.array([[26.5,27.5],[28.5,29.5],[30.5,31.5],[32.5,33.5]])*1e9  # Frequencies in Hz 
        amp, ra_offset, dec_offset, major_std, minor_std, position_angle = best_fits 

        const = 2 * k * (frequencies/c)**2 * 1e26 # Convert to Jy
        flux = 2*np.pi * amp * np.radians(major_std) * np.radians(minor_std) * const[None,...] 

        return flux
    
    def get_flux_density_errors(self, best_fits : np.ndarray, best_fit_errs : np.ndarray) -> np.ndarray:
        """
        Get the flux density from the best fits
        """
        flux = self.get_flux_density(best_fits)

        return np.abs(flux * best_fit_errs[0] / best_fits[0])


    def get_relative_coordinates(self, az : np.ndarray, el : np.ndarray, mjd : np.ndarray, longitude : float, latitude : float, src_ra : float, src_dec : float, obsid : int) -> np.ndarray:
        """
        Get the relative sky coordinates for feed feed 
        """
        try:
            if obsid < 28852:
                azpos = np.gradient(az,20e-3) > 0
                az[azpos] += 100./3600. # Corrects for raster scan drift
            ra, dec = h2e_full(az, el, mjd, longitude, latitude)
        except ValueError:
            raise BadCOMAPFile('Bad coordinates')
        
        rot = hp.Rotator(rot=[src_ra, src_dec])
        rel_ra, rel_dec = rot(ra, dec, lonlat=True)

        return rel_ra, rel_dec
    
    def rotate_pa(self, delta_ra, delta_dec, pa):
        delta_az = np.cos(pa)*delta_ra + np.sin(pa)*delta_dec 
        delta_el =-np.sin(pa)*delta_ra + np.cos(pa)*delta_dec 
        return delta_az, delta_el 
    def rotate_pa_error(self, error_ra, error_dec, pa):
        cq, sq = np.cos(pa), np.sin(pa) 
        R = np.array([[ cq, sq],
                      [-sq, cq]])
        cov = np.array([[error_ra**2, 0],
                        [0          , error_dec**2]])
        cov_azel = R @ cov @ R.T 
        return np.diag(cov_azel)**0.5

    def fit_sources(self, file_info : COMAPData) -> None:

        with RetryH5PY(file_info.level2_path,'r') as f:
            feeds = f['spectrometer/feeds'][:]
            if f['level2/scan_edges'].shape[0] == 0:
                raise BadCOMAPFile(file_info.obsid, "No Scan Edges found in file")

            scan_start,scan_end = f['level2/scan_edges'][0,:]
            mjd = f['spectrometer/MJD'][scan_start:scan_end]
            obsid = file_info.obsid

            if file_info.source.lower() == 'jupiter':
                src_ra, src_dec = get_jupiter_radec_from_mjd(np.nanmedian(mjd))

                location = EarthLocation.from_geodetic(
                    lon=comap_longitude * u.deg,
                    lat=comap_latitude * u.deg,
                    height=1200 * u.m,
)
                altaz_frame = AltAz(obstime=Time(np.nanmedian(mjd),format='mjd'), location=location)

                src_coord = SkyCoord(src_ra*u.deg, src_dec*u.deg, frame='icrs')
                src_altaz = src_coord.transform_to(altaz_frame)
            else:
                src_ra, src_dec = CalibratorList[file_info.source] 

            min_obs_pa = pa(np.array([src_ra]), np.array([src_dec]), np.array([np.min(mjd)]), comap_longitude, comap_latitude) 
            max_obs_pa = pa(np.array([src_ra]), np.array([src_dec]), np.array([np.max(mjd)]), comap_longitude, comap_latitude)
            mean_pa = np.mean([max_obs_pa,min_obs_pa])

            n_feeds, n_bands, n_channels, n_tod = f['level2/binned_filtered_data'].shape
            self.nparameters = 6
            self.best_fits = np.zeros((self.nparameters,n_feeds,n_bands,n_channels))
            self.errs_fits = np.zeros((self.nparameters,n_feeds,n_bands,n_channels))
            self.delta_az = np.zeros((2,n_feeds,n_bands,n_channels))
            self.delta_el = np.zeros((2,n_feeds,n_bands,n_channels))
            self.chi2_fits = np.zeros((n_feeds,n_bands,n_channels)) + np.inf
            for ifeed, feed in enumerate(feeds):
                if feed == 20:
                    continue

                az = f['spectrometer/pixel_pointing/pixel_az'][ifeed,scan_start:scan_end]
                el = f['spectrometer/pixel_pointing/pixel_el'][ifeed,scan_start:scan_end]
                tod =  f['level2/binned_filtered_data'][feed-1,...]
                rel_ra, rel_dec = self.get_relative_coordinates(az,el,mjd,comap_longitude,comap_latitude, src_ra, src_dec, obsid)
                mask = (rel_ra**2 + rel_dec**2) < self.fit_radius**2 

                beam_sigma = 4.5/60.0/2.355
                r2 = rel_ra**2 + rel_dec**2

                for iband, ichan in tqdm(product(range(n_bands), range(n_channels)),desc=f'Processing Feed {feed}'):
                    data = tod[iband,ichan,scan_start:scan_end]

                    N = data.size//2 * 2
                    rms_data = np.nanstd(data[:N:2]-data[1:N:2])/np.sqrt(2)

                    annulus = (r2 > (1.5*beam_sigma)**2) & (r2 < (3.0*beam_sigma)**2)

                    if np.count_nonzero(annulus) > 20:
                        rms_data = np.nanstd(data[annulus])
                    else:
                        N = data.size//2 * 2
                        rms_data = np.nanstd(data[:N:2] - data[1:N:2]) / np.sqrt(2)

                    if (rms_data == 0) | np.isnan(rms_data):
                        continue
                    mask_data = mask & np.isfinite(data)

                    data = data[mask_data] 
                    rel_ra_masked = rel_ra[mask_data]
                    rel_dec_masked = rel_dec[mask_data]
                    if len(data) < 10:
                        continue


                    data_processed = data - np.nanmedian(data)
                    #print(np.sum(rel_ra_masked), np.sum(rel_dec_masked),np.sum(data_processed),rms_data)
                    try:
                        p_gauss, e_gauss, best = fit_gaussian2d_with_errors(rel_ra_masked,rel_dec_masked, data_processed, sigma=rms_data)
                    except np.linalg.LinAlgError:
                        continue

                    self.best_fits[:,feed-1,iband,ichan] = p_gauss
                    self.errs_fits[:,feed-1,iband,ichan] = e_gauss 

                    delta_az, delta_el = self.rotate_pa(p_gauss[1],p_gauss[2],np.radians(mean_pa))
                    delta_az_error, delta_el_error = self.rotate_pa_error(e_gauss[1],e_gauss[2],np.radians(mean_pa))
                    self.delta_az[0,feed-1,iband,ichan] = delta_az 
                    self.delta_el[0,feed-1,iband,ichan] = delta_el
                    self.delta_az[1,feed-1,iband,ichan] = delta_az_error 
                    self.delta_el[1,feed-1,iband,ichan] = delta_el_error

                    residual = data_processed-best(rel_ra_masked, rel_dec_masked) 
                    mask_close = (rel_ra_masked**2 + rel_dec_masked**2) < (4.5/60./2.355)**2
                    chi2 = np.sum((residual[mask_close])**2/rms_data**2)/(np.sum(mask_close)-self.nparameters)
                    self.chi2_fits[feed-1,iband,ichan] = chi2
                

    def make_diagnostic_plots(self, file_info, feed, band, chan,
                              out_prefix="calfit_debug"):
        """
        Make 2D binned images of:
          - data
          - best-fit Gaussian model
          - residual
        for a given (feed, band, chan).
        """
        with RetryH5PY(file_info.level2_path, 'r') as f:
            scan_start, scan_end = f['level2/scan_edges'][0, :]
            mjd  = f['spectrometer/MJD'][scan_start:scan_end]
            az   = f['spectrometer/pixel_pointing/pixel_az'][feed, scan_start:scan_end]
            el   = f['spectrometer/pixel_pointing/pixel_el'][feed, scan_start:scan_end]
            tod  = f['level2/binned_filtered_data'][feed, band, chan, scan_start:scan_end]

            # Source position
            if file_info.source.lower() == 'jupiter':
                src_ra, src_dec = get_jupiter_radec_from_mjd(np.nanmedian(mjd))
            else:
                src_ra, src_dec = CalibratorList[file_info.source]

        rel_ra, rel_dec = self.get_relative_coordinates(
            az, el, mjd, comap_longitude, comap_latitude, src_ra, src_dec,file_info.obsid
        )

        mask = (rel_ra**2 + rel_dec**2) < self.fit_radius**2
        mask &= np.isfinite(tod)

        if np.sum(mask) < 10:
            print("Not enough data points for diagnostic plot.")
            return

        rel_ra = rel_ra[mask]
        rel_dec = rel_dec[mask]
        data = tod[mask]
        data_processed = data - np.nanmedian(data)
        # Rebuild best-fit Gaussian2D + Const2D from stored params
        p = self.best_fits[:, feed, band, chan]  # [amp, x0, y0, sigx, sigy, theta]
        amp, x0, y0, sx, sy, theta = p
        g = models.Gaussian2D(
            amplitude=amp, x_mean=x0, y_mean=y0,
            x_stddev=sx, y_stddev=sy, theta=theta
        )
        bg = models.Const2D(amplitude=0.0)  # already median-subtracted
        model = g + bg

        model_vals = model(rel_ra, rel_dec)
        residual = data_processed - model_vals

        # Bin onto a grid for prettier images
        nbins = 32
        xbins = np.linspace(rel_ra.min(), rel_ra.max(), nbins + 1)
        ybins = np.linspace(rel_dec.min(), rel_dec.max(), nbins + 1)

        def bin2d(vals):
            H, xe, ye, _ = binned_statistic_2d(
                rel_ra, rel_dec, vals,
                statistic='mean', bins=[xbins, ybins]
            )
            return H.T, xe, ye

        data_img, xe, ye = bin2d(data_processed)
        model_img, _, _  = bin2d(model_vals)
        resid_img, _, _  = bin2d(residual)

        fig, axes = pyplot.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        titles = ["Data (median-subtracted)", "Best-fit model", "Residual"]
        imgs = [data_img, model_img, resid_img]

        for ax, img, title in zip(axes, imgs, titles):
            im = ax.imshow(
                img,
                origin="lower",
                extent=[xe[0], xe[-1], ye[0], ye[-1]],
                aspect="equal"
            )
            ax.set_title(title)
            ax.set_xlabel("ΔRA [deg]")
            ax.set_ylabel("ΔDec [deg]")
            pyplot.colorbar(im, ax=ax, shrink=0.8)

        # Text panel with parameters
        chi2 = self.chi2_fits[feed, band, chan]
        fig.suptitle(
            f"OBSID {file_info.obsid}  feed={feed+1}  band={band}  chan={chan}\n"
            f"amp={amp:.3g} K, x0={x0:.4f}°, y0={y0:.4f}°, "
            f"sx={sx*60:.2f}', sy={sy*60:.2f}', chi2_red={chi2:.2f}"
        )

        outfile = f"{out_prefix}_obs{file_info.obsid}_f{feed+1}_b{band}_c{chan}.png"
        fig.savefig(outfile, dpi=200)
        pyplot.close(fig)
        print("Saved", outfile)


    def run(self, file_info : COMAPData) -> None:
        """ """
        #if  file_info.source  == 'jupiter':
        #    logging.info(f'Calibrator fitting not implemented for {file_info.source}')
        #    return 
        if not file_info.source_group == 'Calibrator':
            return

        if self.already_processed(file_info, self.overwrite):
            return
        
        self.fit_sources(file_info)
        self.save_data(file_info)

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dbpath", help="Path to COMAP SQLite database")
    parser.add_argument("--obsid", type=int, nargs="+", required=True,
                        help="One or more obsids to process")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing calibrator_fit group")
    parser.add_argument("--plot", action="store_false",
                        help="Make diagnostic plots")
    parser.add_argument("--chi2_max", type=float, default=None,
                        help="Flag fits with chi2_red > chi2_max")
    parser.add_argument("--flux_min", type=float, default=None,
                        help="Flag fits with flux < flux_min (Jy)")
    parser.add_argument("--feed", type=int, default=None,
                        help="Specific feed (1-based index) to plot")
    parser.add_argument("--band", type=int, default=None,
                        help="Specific band index to plot")
    parser.add_argument("--chan", type=int, default=None,
                        help="Specific channel index to plot")
    args = parser.parse_args()

    db.connect(args.dbpath)

    filelist = db.query_obsid_list(args.obsid, return_list=True, return_dict=False)

    for file_info in filelist:
        print(f"Processing obsid {file_info.obsid}")
        calibrator_fitting = CalibratorFitting(overwrite=args.overwrite)
        calibrator_fitting.run(file_info)

        # Basic printout for a reference channel
        print("Example chi2[0,0,0] =", calibrator_fitting.chi2_fits[0, 0, 0])
        for i in range(6):
            print("param", i,
                  calibrator_fitting.best_fits[i, 0, 0, 0],
                  "+/-",
                  calibrator_fitting.errs_fits[i, 0, 0, 0])

        if args.plot:
            # Either use specific indices or find suspicious ones
            calibrator_fitting.make_diagnostic_plots(
                    file_info,
                    feed=args.feed-1,  # convert to 0-based
                    band=args.band,
                    chan=args.chan,
                    out_prefix="calfit_specific"
                )

    db.disconnect()
