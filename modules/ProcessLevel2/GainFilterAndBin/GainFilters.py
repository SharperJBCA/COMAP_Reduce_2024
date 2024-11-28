import numpy as np 
from scipy.linalg import solve

class GainFilterBase:

    def __init__(self, end_cut = 50):
        self.end_cut = end_cut


    def __call__(self, data : np.ndarray, system_temperature : np.ndarray, auto_rms : np.ndarray) -> np.ndarray: 

        #N = data.shape[-1]//2 * 2
        #auto_rms = np.nanstd(data[...,:N:2] - data[...,1:N:2],axis=-1)/np.sqrt(2)
        normed_data = self.normalise_data(data, auto_rms) 
        normed_data[np.isnan(normed_data)] = 0 
        n_tod = data.shape[-1]

        gain_solution = np.zeros(n_tod)
        gain_solution[...] = self.gain_subtraction_fit(normed_data, system_temperature)

        # Rescale the gain solution to the original data 
        gain_solution = gain_solution[np.newaxis, np.newaxis, :] * auto_rms[..., np.newaxis]

        # Subtract the gain solution from the data
        residual = data - gain_solution

        # Now fit the atmosphere model to the residual 
        return residual

    def normalise_data(self, data : np.ndarray, auto_rms : np.ndarray) -> np.ndarray:
        """
        Normalise the data by the auto-rms
        """
        data_means = np.nanmean(data, axis=2)
        return (data - data_means[..., np.newaxis]) / auto_rms[..., np.newaxis]

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
        A -= np.identity(A.shape[0])   # Regularisation
        g = solve(A, b)

        return g 


    def gain_subtraction_fit(self, data_normed : np.ndarray, system_temperature : np.ndarray):
        """ 
        data_normed (n_bands, n_channels, n_tod)
        """ 

        # Prepare the data and mask bad channels 
        n_bands, n_channels, n_tod = data_normed.shape

        templates = np.ones((n_bands, n_channels, 3))
        n_templates = templates.shape[-1]
        v = np.linspace(-1,1,1024*4).reshape((4,1024))
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
            return np.zeros(data_reshape.T.flatten().shape)

        g = self.solve_gain_solution(data_reshape, templates)

        # subtract a linear fit
        idx = np.arange(g.shape[1])
        g[2] -= np.polyval(np.polyfit(idx, g[2], 1), idx)

        return g[2] # dG/G is in the 3rd template
