import numpy as np 
from scipy.linalg import solve

from modules.utils import mean_filter, median_filter
from multiprocessing import Pool
import numpy as np
from scipy.optimize import minimize
from modules.pipeline_control.Pipeline import RetryH5PY
import logging 

def solve_timepoint(args):
    """
    Helper function to solve for a single timepoint
    
    Arguments
    ---------
    args : tuple
        Contains (t, d_t, templates, initial_guess)
    """
    t, d_t, templates, initial_guess = args
    
    def objective_function(x, d_t, templates):
        residual = d_t - templates.dot(x)
        return np.sum(residual**2)
    
    # Set bounds: first parameter must be positive, second unconstrained
    bounds = [(0, None), (None, None)]  # [(min, max), (min, max)]
    
    result = minimize(
        objective_function,
        initial_guess,
        args=(d_t, templates),
        method='L-BFGS-B',
        bounds=bounds
    )
    
    return t, result.x

class GainFilterBase:

    def __init__(self, end_cut = 50):
        super().__init__()
        self.end_cut = end_cut


    def __call__(self, data : np.ndarray,
                system_temperature : np.ndarray,
                median_offsets : np.ndarray,
                feed : int, elevation : np.ndarray = None, **kwargs) -> np.ndarray:

        normed_data = data/median_offsets[...,np.newaxis]
        normed_data -= np.nanmean(normed_data,axis=-1, keepdims=True)
        normed_data[np.isnan(normed_data)] = 0
        n_tod = data.shape[-1]

        gain_solution = np.zeros(n_tod)
        gain_temp, mask = self.gain_subtraction_fit(normed_data, system_temperature)
        if gain_temp.shape[0] == n_tod:
            gain_solution[...] = gain_temp

        # If an elevation track is supplied, identify a residual atmosphere
        # component in the gain solution (a piece with the 1/sin(El) airmass
        # shape) and split it off, so the airmass-shaped variation is removed
        # from the data without being attributed to instrumental gain.
        atmos_solution = np.zeros(n_tod)
        if elevation is not None:
            airmass = 1.0 / np.sin(elevation)
            airmass = airmass - np.mean(airmass)
            denom = float(np.dot(airmass, airmass))
            if denom > 0:
                b = float(np.dot(gain_solution, airmass)) / denom
                atmos_solution = b * airmass
                gain_solution = gain_solution - atmos_solution

        # Rescale both components back to data units and subtract.
        total = (gain_solution + atmos_solution)[np.newaxis, np.newaxis, :] * median_offsets[...,np.newaxis]
        residual = data - total

        return residual, mask, gain_solution

    def normalise_data(self, data : np.ndarray, auto_rms : np.ndarray) -> np.ndarray:
        """
        Normalise the data by the auto-rms
        """
        #data_means = np.nanmean(data, axis=2)
        median_value = np.nanmedian(data,axis=-1) 
        return data/median_value[:,:,np.newaxis] - 1.0, median_value
        # (data - data_means[..., np.newaxis]) / auto_rms[..., np.newaxis]

    def solve_gain_solution(self, d, templates):
        """
        Arguments
        ---------
        d : array of shape (frequency*time) - observed data as a single vector
        templates : array of shape (frequency*time, 2) - templates of 1/Tsys and 1 (plus additional templates if needed)
        
        A = (n_channels X n_templates)
        d = (n_channels X n_time)
        therefore g = (n_templates X n_time)
        we are storing the gain solution in the 3rd template solution

        We are solve A g = d  

        g_hat = (A^T A)^-1 A^T b 
        
        """
        b = templates.T.dot(d) 
        A = templates.T.dot(templates)

        try:
            g = solve(A, b)
        except np.linalg.LinAlgError:
            g = np.zeros_like(templates).T

        return g 



    def solve_gain_solution_minimise(self, d, templates):
        """
        Arguments
        ---------
        d : array of shape (frequency*time) - observed data as a single vector
        templates : array of shape (frequency*time, 2) - templates of 1/Tsys and 1
        """
        n_templates = templates.shape[1]
        n_time = d.shape[1]
        
        # Initial guess
        initial_guess = np.zeros(n_templates)
        
        # Create arguments for each timepoint
        args = [(t, d[:, t], templates, initial_guess) for t in range(n_time)]
        
        # Solve in parallel using multiprocessing
        with Pool() as pool:
            results = list(pool.imap(solve_timepoint, args))
        
        # Sort results by timepoint and create solution array
        results.sort()  # Sort by timepoint (first element of each tuple)
        g = np.zeros((n_templates, n_time))
        for t, solution in results:
            g[:, t] = solution
            
        return g

    def gain_subtraction_fit(self, data_normed : np.ndarray, system_temperature : np.ndarray, return_all : bool = False) -> np.ndarray:
        """ 
        data_normed (n_bands, n_channels, n_tod)
        """ 

        # Prepare the data and mask bad channels 
        n_bands, n_channels, n_tod = data_normed.shape

        templates = np.ones((n_bands, n_channels, 3))
        n_templates = templates.shape[-1]
        v = np.linspace(-1,1,n_channels*n_bands).reshape((n_bands,n_channels))
        if n_bands > 1:
            v[0] = v[0,::-1]
            v[2] = v[2,::-1]
        templates[..., 0] = 1./system_temperature  
        templates[..., 1] = v/system_temperature

        # Remove edge frequencies and the bad middle frequency
        bad_values = np.isnan(system_temperature) | (system_temperature <= 0) | np.isinf(templates[...,1])
        templates[:, :self.end_cut ,:] = 0
        templates[:, -self.end_cut:,:] = 0
        templates[:, 512-self.end_cut//2:512+self.end_cut//2 ,:] = 0
        templates[bad_values,:] = 0
        data_normed[:, :self.end_cut ,:] = 0
        data_normed[:, -self.end_cut:,:] = 0
        data_normed[:, 512-self.end_cut//2:512+self.end_cut//2 ,:] = 0
        data_normed[bad_values,:] = 0 

        mask = templates[..., 0] != 0

        templates    = templates.reshape(  (n_bands * n_channels, n_templates)) 
        data_reshape = data_normed.reshape((n_bands * n_channels, n_tod))
        if np.sum(bad_values) == bad_values.size:
            return np.zeros(n_tod), mask

        not_zeros = np.sum(templates, axis=1) != 0  


        g = self.solve_gain_solution(data_reshape[not_zeros], templates[not_zeros])

        if return_all:
            return g, mask 
        else:
            #logging.info(f'Gain filter shape {g.shape}')
            return g[-1], mask # dG/G is in the 3rd template
        
    @staticmethod
    def PS_1f(sigma_red, alpha, n_samples, sample_rate=50.0, k_ref=0.1):
        """1/f prior PSD. k_ref must match the value used to fit sigma_red in
        NoiseStats (0.1 Hz)."""
        freqs = np.fft.rfftfreq(n_samples, d=1/sample_rate)
        PS = np.zeros_like(freqs)
        PS[1:] = (sigma_red)**2 * np.abs(freqs[1:]/k_ref)**alpha
        return PS
    
    def remove_edges(self, data, system_temperature):
        # Remove edge frequencies and the bad middle frequency
        bad_values = np.isnan(system_temperature) | (system_temperature <= 0) | np.isinf(system_temperature)
        data[:self.end_cut ,:] = 0
        data[-self.end_cut:,:] = 0
        data[512-self.end_cut//2:512+self.end_cut//2 ,:] = 0
        data[bad_values,:] = 0
        return data 
    


class GainFilterWithPrior(GainFilterBase):

    def __init__(self, end_cut = 100, noise_prior_path='ancillary_data/noise_priors/noise_priors.npy'):
        super().__init__()
        self.end_cut = end_cut

        self.noise_priors = np.load(noise_prior_path, allow_pickle=True).flatten()[0] 

    def __call__(self, data : np.ndarray,
                system_temperature : np.ndarray,
                median_offsets : np.ndarray,
                feed : int,
                alpha=-1,
                sigma_red=0.5,
                elevation : np.ndarray = None) -> np.ndarray:

        normed_data = data/median_offsets[...,np.newaxis]
        normed_data -= np.nanmean(normed_data,axis=-1, keepdims=True)
        normed_data[np.isnan(normed_data)] = 0
        n_tod = data.shape[-1]

        sigma_red, alpha = self.noise_priors[feed]
        residual = np.zeros_like(normed_data)
        mask = np.zeros(median_offsets.shape, dtype=bool)
        gains = []
        for iband in range(normed_data.shape[0]):
            gain_solution = np.zeros(n_tod)
            atmos_solution = np.zeros(n_tod)
            gain_temp, mbest, gain_templates, sys_templates, atmos_temp = self.gain_subtract_fit_with_prior(
                normed_data[iband:iband+1],
                system_temperature[iband:iband+1],
                sigma_red, alpha,
                elevation=elevation)

            gain_temp = gain_temp[0]
            if gain_temp.shape[0] == n_tod:
                gain_solution[...] = gain_temp
            if atmos_temp is not None and atmos_temp.shape[0] == n_tod:
                atmos_solution[...] = atmos_temp

            # Both components scaled back to data units (broadcast over freq).
            total = (gain_solution + atmos_solution)[np.newaxis, np.newaxis, :] \
                    * median_offsets[iband:iband+1, :, np.newaxis]

            residual[iband:iband+1] = data[iband:iband+1] - gain_templates * total[np.newaxis,:]
            mask[iband] = np.sum(np.abs(residual[iband,:,:]),axis=1) > 0
            gains.append(gain_solution)

        gains = np.array(gains)
        # print(residual.shape,median_offsets.shape, np.nanmedian(median_offsets))
        # print(mbest.shape, gain_templates.shape, gain_temp.shape, sys_templates.shape)
        # from matplotlib import pyplot 
        # import sys 
        # pyplot.plot(data[0,100,:1000])
        # pyplot.plot(residual[0,100,:1000])
        # pyplot.savefig('test.png')
        # sys.exit()
        return residual, mask, gains

    def gain_subtract_fit_with_prior(self, data_normed : np.ndarray, system_temperature : np.ndarray, sigma_red : float, alpha : float, elevation : np.ndarray = None, return_all : bool = False, sample_rate : float = 50.0) -> np.ndarray:
        """
        Gain subtraction with the 1/f noise prior. If an elevation track is
        passed, also jointly fits an atmosphere component with fixed time shape
        1/sin(El) and one scalar amplitude, removed from the gain solution
        (and additionally returned for separate subtraction).
        """

        n_bands, nfreqs, ntod = data_normed.shape

        sys_templates  = np.ones((n_bands,nfreqs, 2))
        gain_templates = np.ones((n_bands,nfreqs, 1))
        v = np.linspace(-1,1,nfreqs*n_bands).reshape((n_bands,nfreqs))
        if n_bands > 1:
            v[0] = v[0,::-1]
            v[2] = v[2,::-1]
        sys_templates[..., 0] = 1./system_temperature
        sys_templates[..., 1] = v/system_temperature
        for iband in range(n_bands):
            sys_templates[iband]  = self.remove_edges(sys_templates[iband] , system_temperature[iband])
            gain_templates[iband] = self.remove_edges(gain_templates[iband], system_temperature[iband])
            data_normed[iband]    = self.remove_edges(data_normed[iband]   , system_temperature[iband])
        sys_templates  = sys_templates.reshape((n_bands * nfreqs, 2))
        gain_templates = gain_templates.reshape((n_bands * nfreqs, 1))
        data_normed    = data_normed.reshape((n_bands * nfreqs, ntod))

        cov = self.PS_1f(sigma_red, alpha, ntod, sample_rate=sample_rate)

        PtP = np.linalg.pinv(np.dot(sys_templates.T, sys_templates))
        Z = np.eye(n_bands*nfreqs) - sys_templates.dot(PtP).dot(sys_templates.T)
        RHS = np.fft.rfft(gain_templates.T.dot(Z).dot(data_normed))  # (1, n_fft)
        z   = gain_templates.T.dot(Z).dot(gain_templates)[0]         # (1,)
        z_scalar = float(z[0])

        # 1/cov, treating cov=0 (DC bin of PS_1f) as an infinite penalty so
        # the gain DC is pinned to zero.
        inv_cov = np.where(cov > 0, np.divide(1.0, cov, where=cov > 0,
                                              out=np.full_like(cov, np.inf)),
                           np.inf)

        atmos_amplitude = None
        if elevation is not None and np.all(np.isfinite(elevation)) and z_scalar > 0:
            # Joint MAP fit of gain (1/f prior) and a single airmass-shaped
            # atmosphere amplitude (flat prior). Derivation: the atmosphere
            # coefficient b is determined by the data component that the gain
            # filter does NOT absorb -- weighted by P_high(f), the "leak" of
            # the prior past the Wiener filter.
            airmass = 1.0 / np.sin(elevation)
            airmass = airmass - np.mean(airmass)
            a_ft = np.fft.rfft(airmass)
            n_fft = a_ft.size

            # Parseval weights for one-sided FFT of real signal.
            wts = np.full(n_fft, 2.0)
            wts[0] = 1.0
            if ntod % 2 == 0:
                wts[-1] = 1.0

            P_high = np.where(np.isinf(inv_cov), 1.0,
                              inv_cov / (z_scalar + inv_cov))

            num = float(np.sum(wts * np.real(np.conj(a_ft) * RHS[0]) * P_high)) / z_scalar
            den = float(np.sum(wts * np.abs(a_ft)**2 * P_high))
            b = num / den if den > 0 else 0.0
            atmos_amplitude = b * airmass

            RHS_g = RHS - z_scalar * b * a_ft[np.newaxis, :]
        else:
            RHS_g = RHS

        a_best_fit_ft = RHS_g / (z + 1./cov)
        a_best_fit    = np.fft.irfft(a_best_fit_ft, n=ntod)

        m_best_fit = np.linalg.pinv(sys_templates.T.dot(sys_templates)).dot(sys_templates.T).dot(data_normed - gain_templates.dot(a_best_fit))

        return a_best_fit, m_best_fit, gain_templates, sys_templates, atmos_amplitude
