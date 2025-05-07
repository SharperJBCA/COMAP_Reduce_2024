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
from astropy.modeling import models, fitting
import astropy.units as u
from astropy.modeling.functional_models import Gaussian2D
from matplotlib import pyplot 
import time 
try:
    from modules.utils.Coordinates import comap_longitude, comap_latitude, h2e_full, CalibratorList, pa
    from modules.utils.data_handling import read_2bit
    from modules.SQLModule.SQLModule import COMAPData, db
    from modules.pipeline_control.Pipeline import RetryH5PY
except ModuleNotFoundError:
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))    
    from modules.utils.Coordinates import comap_longitude, comap_latitude, h2e_full, CalibratorList, pa
    from modules.utils.data_handling import read_2bit
    from modules.SQLModule.SQLModule import COMAPData,db
    from modules.pipeline_control.Pipeline import RetryH5PY,BaseCOMAPModule

class CalibratorFitting(BaseCOMAPModule):

    def __init__(self, calibrator_fit_group_name='calibrator_fit', overwrite=False, fit_radius=15./60.): 

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

            ds = grp.create_dataset('amplitudes', data=self.best_fits[0])
            ds.attrs['unit'] = 'K'
            ds = grp.create_datset('amplitude_errors', data=self.errs_fits[0])
            ds.attrs['unit'] = 'K'

            ds = grp.create_dataset('ra_offset', data=self.best_fits[1])
            ds.attrs['unit'] = 'degrees'
            ds = grp.create_dataset('ra_offset_errors', data=self.errs_fits[1]) 
            ds.attrs['unit'] = 'degrees'

            ds = grp.create_dataset('dec_offset', data=self.best_fits[2])
            ds.attrs['unit'] = 'degrees'
            ds = grp.create_dataset('dec_offset_errors', data=self.errs_fits[2])
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

            ds = grp.create_dataset('flux', data=self.get_flux_density(self.best_fits))
            ds.attrs['unit'] = 'Jy'
            ds = grp.create_dataset('flux_errors', data=self.get_flux_density(self.errs_fits))
            ds.attrs['unit'] = 'Jy'

    def get_relative_coordinates(self, az : np.ndarray, el : np.ndarray, mjd : np.ndarray, longitude : float, latitude : float, src_ra, src_dec) -> np.ndarray:
        """
        Get the relative sky coordinates for feed feed 
        """
        ra, dec = h2e_full(az, el, mjd, longitude, latitude)
        
        rot = hp.Rotator(rot=[src_ra, src_dec])
        rel_ra, rel_dec = rot(ra, dec, lonlat=True)

        return rel_ra, rel_dec

    def fit_source(self, data : np.ndarray, rel_ra : np.ndarray, rel_dec : np.ndarray) -> np.ndarray:
        """
        Fit a source to the data 
        """
        # Initial guesses
        amplitude_guess = np.max(data) - np.median(data)  # Peak above background
        x_mean_guess = 0.0
        y_mean_guess = 0.0
        x_stddev_guess = np.sqrt(np.sum(rel_ra**2 * data**2) / np.sum(data**2))
        y_stddev_guess = np.sqrt(np.sum(rel_dec**2 * data**2) / np.sum(data**2))
        theta_guess = 0.0
        background_guess = np.median(data)

        # Create Model
        init_model = models.Gaussian2D(
            amplitude=amplitude_guess,
            x_mean=x_mean_guess,
            y_mean=y_mean_guess,
            x_stddev=x_stddev_guess,
            y_stddev=y_stddev_guess,
            theta=theta_guess
        ) + models.Const2D(amplitude=background_guess)

        # Fit Model
        fitter = fitting.LevMarLSQFitter()
        best_fit_model = fitter(init_model, rel_ra, rel_dec, data)

        best_fit = best_fit_model[0]  # The Gaussian2D part
        if best_fit.y_stddev.value > best_fit.x_stddev.value:
            minor_axis = best_fit.x_stddev.value*1
            major_axis = best_fit.y_stddev.value*1
            position_angle = np.mod(best_fit.theta.value + np.pi/2., 2*np.pi)
        else:
            minor_axis = best_fit.y_stddev.value*1
            major_axis = best_fit.x_stddev.value*1
            position_angle =np.mod(best_fit.theta.value, 2*np.pi)

        model_data = best_fit_model(rel_ra, rel_dec)
        peak_idx = np.argmax(model_data)
        return (best_fit.amplitude.value, best_fit.x_mean.value, best_fit.y_mean.value, major_axis, minor_axis, position_angle), peak_idx

    def fit_source_bootstrap(self, data : np.ndarray, rel_ra : np.ndarray, rel_dec : np.ndarray, n_bootstraps=20) -> np.ndarray:

        best_fits = np.zeros((6,n_bootstraps))
        peak_idx = np.zeros(n_bootstraps)
        for i in range(n_bootstraps):
            mask = np.random.choice(data.size, data.size, replace=True)
            best_fits[:,i], peak_idx[i] = self.fit_source(data[mask], rel_ra[mask], rel_dec[mask])


        return np.mean(best_fits, axis=1), np.std(best_fits, axis=1), int(np.median(peak_idx))

    def fit_sources(self, file_info : COMAPData) -> None:

        with RetryH5PY(file_info.level2_path,'r') as f:
            src_ra, src_dec = CalibratorList[file_info.source] 
            feeds = f['spectrometer/feeds'][:]
            scan_start,scan_end = f['level2/scan_edges'][0,:]
            mjd = f['spectrometer/MJD'][scan_start:scan_end]

            min_obs_pa = pa(np.array([src_ra]), np.array([src_dec]), np.array([np.min(mjd)]), comap_longitude, comap_latitude) 
            max_obs_pas = pa(np.array([src_ra]), np.array([src_dec]), np.array([np.max(mjd)]), comap_longitude, comap_latitude)
            print(min_obs_pa, max_obs_pas)

            n_feeds, n_bands, n_channels, n_tod = f['level2/binned_filtered_data'].shape
            self.best_fits = np.zeros((6,n_feeds,n_bands,n_channels))
            self.errs_fits = np.zeros((6,n_feeds,n_bands,n_channels))
            for ifeed, feed in enumerate(feeds):
                az = f['spectrometer/pixel_pointing/pixel_az'][ifeed,scan_start:scan_end]
                el = f['spectrometer/pixel_pointing/pixel_el'][ifeed,scan_start:scan_end]
                rel_ra, rel_dec = self.get_relative_coordinates(az,el,mjd,comap_longitude,comap_latitude, src_ra, src_dec)

                mask = (rel_ra**2 + rel_dec**2) < self.fit_radius**2 


                for iband, ichan in product(range(n_bands), range(n_channels)):
                    data = f['level2/binned_filtered_data'][ifeed,iband,ichan,scan_start:scan_end]
                    mask_data = mask & np.isfinite(data)
                    data = data[mask_data] 
                    data_processed = data - np.median(data)
                    mjd_masked = mjd[mask_data]
                    t0 = time.time()
                    self.best_fits[:,ifeed,iband,ichan],self.errs_fits[:,ifeed,iband,ichan], peak_idx= self.fit_source_bootstrap(data_processed, rel_ra[mask_data], rel_dec[mask_data])
                    obs_pas = pa(np.array([src_ra]), np.array([src_dec]), np.array([mjd_masked[peak_idx]]), comap_longitude, comap_latitude)

                    print('parallactic angle',obs_pas, min_obs_pa, max_obs_pas)
                    print('run time', time.time()-t0, (time.time()-t0)*8*19/60.)
                    print(f'Feed {feed} Band {iband} Channel {ichan} - Amplitude: {self.best_fits[0,ifeed,iband,ichan]} +/- {self.errs_fits[0,ifeed,iband,ichan]}')
                    print(f'Feed {feed} Band {iband} Channel {ichan} - RA Offset: {self.best_fits[1,ifeed,iband,ichan]*60} +/- {self.errs_fits[1,ifeed,iband,ichan]*60}')
                    print(f'Feed {feed} Band {iband} Channel {ichan} - Dec Offset: {self.best_fits[2,ifeed,iband,ichan]*60} +/- {self.errs_fits[2,ifeed,iband,ichan]*60}')
                    print(f'Feed {feed} Band {iband} Channel {ichan} - Major Std: {self.best_fits[3,ifeed,iband,ichan]*60} +/- {self.errs_fits[3,ifeed,iband,ichan]*60}')
                    print(f'Feed {feed} Band {iband} Channel {ichan} - Minor Std: {self.best_fits[4,ifeed,iband,ichan]*60} +/- {self.errs_fits[4,ifeed,iband,ichan]*60}')
                    print(f'Feed {feed} Band {iband} Channel {ichan} - Position Angle: {self.best_fits[5,ifeed,iband,ichan]} +/- {self.errs_fits[5,ifeed,iband,ichan]}')
                sys.exit()


    def run(self, file_info : COMAPData) -> None:
        """ """
        if not file_info.source_group == 'Calibrator':
            return

        if self.already_processed(file_info, self.overwrite):
            return
        
        self.fit_sources(file_info)
        #self.save_data(file_info)

if __name__ == '__main__':
    import matplotlib 
    matplotlib.use('Agg')
    db.connect(sys.argv[1])
    filelist = db.query_obsid_list([44070], return_list=True, return_dict=False)
    calibrator_fitting = CalibratorFitting()
    calibrator_fitting.run(filelist[0])
    db.disconnect() 
