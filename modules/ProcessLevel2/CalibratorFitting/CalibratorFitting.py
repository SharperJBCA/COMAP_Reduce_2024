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
import time 
from datetime import datetime
try:
    from modules.utils.Coordinates import comap_longitude, comap_latitude, h2e_full, CalibratorList, pa
    from modules.utils.data_handling import read_2bit
    from modules.SQLModule.SQLModule import COMAPData, db
    from modules.pipeline_control.Pipeline import BadCOMAPFile, RetryH5PY,BaseCOMAPModule
except ModuleNotFoundError:
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))    
    from modules.utils.Coordinates import comap_longitude, comap_latitude, h2e_full, CalibratorList, pa
    from modules.utils.data_handling import read_2bit
    from modules.SQLModule.SQLModule import COMAPData,db
    from modules.pipeline_control.Pipeline import BadCOMAPFile, RetryH5PY,BaseCOMAPModule


from scipy import optimize
from scipy.optimize import minimize 
from scipy.optimize import approx_fprime

def fisher_matrix(params, x, y, data, error, epsilon=1e-6):
    """
    Calculate the Fisher information matrix for the 2D Gaussian model.
    
    Parameters:
    -----------
    params : array-like
        Parameters of the 2D Gaussian (amplitude, x_mean, y_mean, x_stddev, y_stddev, theta)
    x, y : array-like
        Coordinate arrays
    data : array-like
        Observed data values
    error : array-like or float
        Uncertainties on the data
    epsilon : float, optional
        Step size for numerical differentiation
        
    Returns:
    --------
    numpy.ndarray
        Fisher information matrix
    """

    def error_func(params, x, y, data, error):
        """
        Calculate the error function for the 2D Gaussian model.
        """
        amplitude, x_mean, y_mean, x_stddev, y_stddev, theta = params
        model = Gaussian2D(amplitude=amplitude, x_mean=x_mean, y_mean=y_mean,
                            x_stddev=x_stddev, y_stddev=y_stddev, theta=theta)
        model_data = model(x, y)
        
        # Calculate the residuals and apply error weighting
        residuals = (data - model_data) / error
        return -np.sum(residuals**2)

    n_params = len(params)
    fisher = np.zeros((n_params, n_params))
    
    # Compute negative second derivatives of log-likelihood
    for i in range(n_params):
        for j in range(i, n_params):
            # Second partial derivative with respect to parameters i and j
            if i == j:
                # Diagonal elements - second derivative with respect to the same parameter
                def f(p):
                    params_modified = params.copy()
                    params_modified[i] = p
                    return error_func(params_modified, x, y, data, error)
                
                # Second derivative using finite difference
                d2l_dp2 = approx_fprime(np.array([params[i]]), 
                                        lambda p: approx_fprime(p, f, epsilon)[0], 
                                        epsilon)[0]
                fisher[i, j] = -d2l_dp2
            else:
                # Off-diagonal elements - mixed partial derivatives
                def f_ij(p):
                    params_modified = params.copy()
                    params_modified[i] = p[0]
                    params_modified[j] = p[1]
                    return error_func(params_modified, x, y, data, error)
                
                # Create a function that computes the derivative with respect to parameter j
                def df_dj(p_i):
                    #print(params[j], p_i[0])
                    #print(np.array([p_i[0], params[j]]))

                    def func_test(p_j):
                        if isinstance(p_j, np.ndarray):
                            p_j = p_j.flatten()[0]
                        a = np.array([p_i[0], p_j])
                        chi =  f_ij(a)
                        return chi

                    return approx_fprime(params[j], 
                                        func_test, #lambda p_j: f_ij(np.array([p_i[0], p_j])), 
                                        epsilon)[0]
                # Compute the mixed derivative
                d2l_didj = approx_fprime(np.array([params[i]]), df_dj, epsilon)[0]
                fisher[i, j] = -d2l_didj
                fisher[j, i] = fisher[i, j]  # Matrix is symmetric
    
    return fisher



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


    def get_relative_coordinates(self, az : np.ndarray, el : np.ndarray, mjd : np.ndarray, longitude : float, latitude : float, src_ra : float, src_dec : float) -> np.ndarray:
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
        x_mean_guess = rel_ra[np.argmax(data)]
        y_mean_guess = rel_dec[np.argmax(data)]
        x_stddev_guess = 4.5/60./2.355
        y_stddev_guess = 4.5/60./2.355
        theta_guess = 0.0

        samples = np.linspace(-1,1,len(data))
        #guesses = (amplitude_guess, x_mean_guess, y_mean_guess, x_stddev_guess, y_stddev_guess, theta_guess, 0, 0, 0, 0)
        #result = minimize(error_func, guesses, args=(rel_ra, rel_dec, samples, data), method='Nelder-Mead', options={'maxiter': 1000})
        #model_data = Gaussian2DPlusPoly1D(rel_ra, rel_dec, samples, *result.x)
        #peak_idx = np.argmax(model_data)
        niter = 2 
        fitter = fitting.LevMarLSQFitter()
        data_temp = data*1. 

        gauss_model = Gaussian2D(amplitude=amplitude_guess, x_mean=x_mean_guess, y_mean=y_mean_guess,
                            x_stddev=x_stddev_guess, y_stddev=y_stddev_guess, theta=theta_guess)
        poly_model = models.Polynomial1D(degree=3, c0=0, c1=0, c2=0, c3=0)
        for i in range(niter):
            best_model_gauss = fitter(gauss_model, rel_ra, rel_dec, data_temp) 
            data_temp = data - best_model_gauss(rel_ra, rel_dec) 
            gfilter = gaussian_filter(data_temp, 200)
            data_temp = data - gfilter

            
        data_model = best_model_gauss(rel_ra, rel_dec)
        peak_idx =  np.argmax(data_model)
        parameter_list = [p for p in best_model_gauss.parameters] 
        return parameter_list, peak_idx, data_model, gfilter
    #(best_fit.amplitude.value, best_fit.x_mean.value, best_fit.y_mean.value, major_axis, minor_axis, position_angle), peak_idx

    def fit_source_bootstrap(self, data : np.ndarray, rel_ra : np.ndarray, rel_dec : np.ndarray, n_bootstraps=20) -> np.ndarray:

        best_fits = np.zeros((self.nparameters,n_bootstraps))
        peak_idxs = np.zeros(n_bootstraps)
        #for i in range(n_bootstraps):
        #    mask = np.random.uniform(low=0,high=data.size, size=data.size).astype(int)
        #    #np.random.choice(data.size, data.size, replace=True)
        best_fit_values, peak_idx, data_model, gfilter = self.fit_source(data, rel_ra, rel_dec)
            #best_fits[:,i] = best_fit_values
            #peak_idxs[i] = peak_idx
        
        #best_fit_values = np.nanmean(best_fits, axis=1)
        #std_errs = np.nanstd(best_fits, axis=1)
        #peak_idx = int(np.nanmedian(peak_idxs))
        stddev = np.nanstd(data-data_model-gfilter)
        fm = fisher_matrix(best_fit_values,  rel_ra, rel_dec, data-gfilter, stddev)
        cov_matrix = np.linalg.inv(fm)
        # Standard errors are the square roots of the diagonal elements
        std_errors_fm = np.sqrt(np.diag(cov_matrix))

        #print(std_errs)
        return best_fit_values, std_errors_fm, int(np.median(peak_idx)), data_model, gfilter

    def fit_sources(self, file_info : COMAPData) -> None:

        with RetryH5PY(file_info.level2_path,'r') as f:
            src_ra, src_dec = CalibratorList[file_info.source] 
            feeds = f['spectrometer/feeds'][:]
            if f['level2/scan_edges'].shape[0] == 0:
                raise BadCOMAPFile(file_info.obsid, "No Scan Edges found in file")

            scan_start,scan_end = f['level2/scan_edges'][0,:]
            mjd = f['spectrometer/MJD'][scan_start:scan_end]

            min_obs_pa  = pa(np.array([src_ra]), np.array([src_dec]), np.array([np.min(mjd)]), comap_longitude, comap_latitude) 
            max_obs_pas = pa(np.array([src_ra]), np.array([src_dec]), np.array([np.max(mjd)]), comap_longitude, comap_latitude)

            n_feeds, n_bands, n_channels, n_tod = f['level2/binned_filtered_data'].shape
            self.nparameters = 6
            self.best_fits = np.zeros((self.nparameters,n_feeds,n_bands,n_channels))
            self.errs_fits = np.zeros((self.nparameters,n_feeds,n_bands,n_channels))
            self.chi2_fits = np.zeros((n_feeds,n_bands,n_channels)) + np.inf
            for ifeed, feed in enumerate(feeds):
                if feed == 20:
                    continue

                az = f['spectrometer/pixel_pointing/pixel_az'][ifeed,scan_start:scan_end]
                el = f['spectrometer/pixel_pointing/pixel_el'][ifeed,scan_start:scan_end]

                #print('AZEL SUM', ifeed, feed, np.sum(az), np.sum(el))
                rel_ra, rel_dec = self.get_relative_coordinates(az,el,mjd,comap_longitude,comap_latitude, src_ra, src_dec)
                mask = (rel_ra**2 + rel_dec**2) < self.fit_radius**2 


                for iband, ichan in tqdm(product(range(n_bands), range(n_channels)),desc=f'Processing Feed {feed}'):
                    data = f['level2/binned_filtered_data'][feed-1,iband,ichan,scan_start:scan_end]

                    N = data.size//2 * 2
                    rms_data = np.nanstd(data[:N:2]-data[1:N:2])/np.sqrt(2)
                    if rms_data == 0:
                        continue

                    mask_data = mask & np.isfinite(data)
                    data = data[mask_data] 
                    if len(data) < 10:
                        continue

                    data_processed = data - np.median(data)

                    try:
                        self.best_fits[:,feed-1,iband,ichan],self.errs_fits[:,feed-1,iband,ichan], peak_idx, data_model, gfilter = self.fit_source_bootstrap(data_processed, 
                                                                                                                                    rel_ra[mask_data], 
                                                                                                                                    rel_dec[mask_data])
                    except np.linalg.LinAlgError:
                        continue
                    residual = data_processed-data_model-gfilter 
                    rel_ra_masked = rel_ra[mask_data]
                    rel_dec_masked = rel_dec[mask_data]
                    mask_close = (rel_ra_masked**2 + rel_dec_masked**2) < (4.5/60./2.355)**2
                    chi2 = np.sum((residual[mask_close])**2/rms_data**2)/(np.sum(mask_close)-self.nparameters)
                    self.chi2_fits[feed-1,iband,ichan] = chi2
                    # amp, ra_offset, dec_offset, major_std, minor_std, position_angle = self.best_fits[:,ifeed,iband,ichan]
                    # print('OFFSETS', ra_offset*60, dec_offset*60)
                    # print('CHI2', chi2)
                    # pyplot.plot(data_processed-data_model-gfilter)
                    # pyplot.plot(data_model)
                    # pyplot.grid(which='both',ls=':',alpha=0.5,color='k')
                    # pyplot.xlabel('sample')
                    # pyplot.ylabel('K')
                    # pyplot.title(f'Feed {feed} Band {iband} Chan {ichan} MJD {mjd_masked[peak_idx]}')
                    # pyplot.show()
                


    def run(self, file_info : COMAPData) -> None:
        """ """
        if not file_info.source_group == 'Calibrator':
            return

        if self.already_processed(file_info, self.overwrite):
            return
        
        self.fit_sources(file_info)
        self.save_data(file_info)

if __name__ == '__main__':
    import matplotlib 
    matplotlib.use('Agg')
    db.connect(sys.argv[1])
    filelist = db.query_obsid_list([44070], return_list=True, return_dict=False)
    calibrator_fitting = CalibratorFitting()
    calibrator_fitting.run(filelist[0])
    db.disconnect() 
