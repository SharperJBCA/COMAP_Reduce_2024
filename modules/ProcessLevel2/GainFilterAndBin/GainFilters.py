import numpy as np 
from scipy.linalg import solve

from modules.utils import mean_filter, median_filter
from multiprocessing import Pool
import numpy as np
from scipy.optimize import minimize

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
        self.end_cut = end_cut


    def __call__(self, data : np.ndarray, 
                system_temperature : np.ndarray, 
                median_offsets : np.ndarray,
                auto_rms : np.ndarray) -> np.ndarray: 

        #N = data.shape[-1]//2 * 2
        #auto_rms = np.nanstd(data[...,:N:2] - data[...,1:N:2],axis=-1)/np.sqrt(2)
        normed_data = data/median_offsets[...,np.newaxis] - 1.0 
        #, median_value = self.normalise_data(data, auto_rms) 
        normed_data[np.isnan(normed_data)] = 0 
        n_tod = data.shape[-1]

        gain_solution = np.zeros(n_tod)
        gain_solution[...] = self.gain_subtraction_fit(normed_data, system_temperature)

        # Rescale the gain solution to the original data 
        gain_solution = gain_solution[np.newaxis, np.newaxis, :] * median_offsets[...,np.newaxis] #* auto_rms[..., np.newaxis]

        # Subtract the gain solution from the data
        residual = data - gain_solution

        # Now fit the atmosphere model to the residual 
        return residual

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
        n = templates.shape[1]
        b = templates.T.dot(d) 
        A = templates.T.dot(templates)

        # Regularisation
        #P = np.zeros((1,n)) 
        #P[0,0] = 1 
        #lb = 10
        #A = lb * P.T.dot(P) + A 
        #q = np.zeros((1,d.shape[1]))
        #b = lb * P.T.dot(q) + b
        #A -= np.identity(A.shape[0])   
        g = solve(A, b)

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
        v = np.linspace(-1,1,n_channels*4).reshape((4,n_channels))
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


        templates    = templates.reshape(  (n_bands * n_channels, n_templates)) 
        data_reshape = data_normed.reshape((n_bands * n_channels, n_tod))
        if np.sum(bad_values) == bad_values.size:
            print('All bad values')
            print(system_temperature)
            return np.zeros(n_tod)

        not_zeros = np.sum(templates, axis=1) != 0  
        g = self.solve_gain_solution(data_reshape[not_zeros], templates[not_zeros])

        # subtract a linear fit
        idx = np.arange(g.shape[1])
        #g[2] -= np.polyval(np.polyfit(idx, g[2], 1), idx)

        if return_all:
            return g
        else:
            return g[-1] # dG/G is in the 3rd template
