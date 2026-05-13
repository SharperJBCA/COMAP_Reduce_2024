import numpy as np
from scipy.linalg import solve

from modules.utils import mean_filter, median_filter
from multiprocessing import Pool
import numpy as np
from scipy.optimize import minimize
from modules.pipeline_control.Pipeline import RetryH5PY
import logging


def build_atmos_templates(elevation=None, azimuth=None, min_relative_amplitude=1e-3):
    """Build the (K, ntod) matrix of fixed-shape systematics templates that
    are jointly fit with the gain. Each row has its mean removed (DC is
    projected out via the Tsys templates already). Templates whose
    peak-to-peak amplitude is below ``min_relative_amplitude`` times the
    largest template's are dropped: on a constant-elevation scan the airmass
    template collapses to ~constant and would make the joint MAP normal
    equations rank-deficient, producing destructive cancellation between the
    gain and atmosphere solutions.

    elevation : radians (matches the existing convention in process_data).
    azimuth   : degrees -- circular mean is removed with proper 0/360 wrap.

    Returns (templates, names) where templates is (K, ntod) (or None) and
    names is the list of surviving labels in row order. Names are kept in
    sync with the surviving rows after the amplitude filter so that
    diagnostics label things correctly.
    """
    tmpls, names = [], []
    if elevation is not None and np.all(np.isfinite(elevation)):
        airmass = 1.0 / np.sin(elevation)
        tmpls.append(airmass - np.mean(airmass))
        names.append('airmass')
    if azimuth is not None and np.all(np.isfinite(azimuth)):
        az_rad = np.radians(azimuth)
        mean_dir = np.arctan2(np.mean(np.sin(az_rad)),
                              np.mean(np.cos(az_rad)))
        delta_rad = np.angle(np.exp(1j * (az_rad - mean_dir)))
        delta = np.degrees(delta_rad)
        tmpls.append(delta - np.mean(delta))
        names.append('azimuth')
    if not tmpls:
        return None, []
    amps = np.array([np.ptp(t) for t in tmpls])
    keep = amps > min_relative_amplitude * amps.max()
    tmpls = [t for t, k in zip(tmpls, keep) if k]
    names = [n for n, k in zip(names, keep) if k]
    return (np.array(tmpls), names) if tmpls else (None, [])

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

    def __init__(self, end_cut = 50, debug_dir = None):
        super().__init__()
        self.end_cut = end_cut
        self.debug_dir = debug_dir
        self._debug_call_count = 0

    def _save_debug_plot(self, feed, iband, gain_solution, atmos_solution,
                          atmos_templates, residual_band, P_high=None,
                          ghat_ft=None, RHS_ft=None, n_tod=None, b_vec=None,
                          atmos_names=None):
        """Save a 2x2 diagnostic PNG. Called only when self.debug_dir is set."""
        if not self.debug_dir:
            return
        import os
        from matplotlib import pyplot as plt
        os.makedirs(self.debug_dir, exist_ok=True)
        n = self._debug_call_count

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        title_b = '' if b_vec is None else (
            ' | b=[' + ', '.join(f'{v:.3g}' for v in b_vec) + ']')
        fig.suptitle(f'feed {feed:02d} band {iband} call {n}{title_b}', fontsize=11)

        ax = axes[0, 0]
        ax.plot(gain_solution, lw=0.5, color='C0', label=r'$\hat g(t)$')
        if atmos_templates is not None and b_vec is not None:
            labels = (atmos_names if atmos_names is not None else [f'tmpl{i}' for i in range(atmos_templates.shape[0])])
            for k, lab in enumerate(labels):
                ax.plot(b_vec[k] * atmos_templates[k], lw=0.5,
                        label=fr'$b_{{{lab}}} \tau_{{{lab}}}(t)$')
        ax.plot(gain_solution + atmos_solution, lw=0.7, color='k', alpha=0.6,
                label=r'$\hat s(t)$ total')
        ax.set_xlabel('sample'); ax.set_ylabel('amplitude (dG/G units)')
        ax.legend(fontsize=8); ax.set_title('Time-series decomposition')

        ax = axes[0, 1]
        if atmos_templates is not None:
            for k, lab in enumerate((atmos_names if atmos_names is not None else [f'tmpl{i}' for i in range(atmos_templates.shape[0])])):
                t = atmos_templates[k]
                tn = t / max(np.linalg.norm(t), 1e-30)
                proj = residual_band @ tn   # (nfreqs,)
                ax.plot(proj, lw=0.6, label=lab)
            ax.axhline(0, color='k', lw=0.4)
            ax.set_xlabel('channel'); ax.set_ylabel(r'$r_{\nu}\cdot\hat\tau$')
            ax.legend(fontsize=8)
            ax.set_title('Per-channel residual projection onto template')
        else:
            ax.text(0.5, 0.5, 'no atmos templates', ha='center', va='center')

        ax = axes[1, 0]
        if ghat_ft is not None and n_tod is not None:
            freqs = np.fft.rfftfreq(n_tod, d=1/50.0)
            ax.loglog(freqs[1:], np.abs(ghat_ft[1:]), lw=0.6, label=r'$|\hat g_{ft}(f)|$')
            if RHS_ft is not None:
                ax.loglog(freqs[1:], np.abs(RHS_ft[1:]), lw=0.6, alpha=0.6,
                          label=r'$|R(f)|$')
            if atmos_templates is not None:
                for k, lab in enumerate((atmos_names if atmos_names is not None else [f'tmpl{i}' for i in range(atmos_templates.shape[0])])):
                    tf = np.fft.rfft(atmos_templates[k])
                    ax.loglog(freqs[1:], np.abs(tf[1:]), lw=0.5, alpha=0.6,
                              label=fr'$|\tilde\tau_{{{lab}}}|$')
            ax.set_xlabel('frequency [Hz]'); ax.set_ylabel('amplitude')
            ax.legend(fontsize=8); ax.set_title('Fourier amplitudes')
        else:
            ax.text(0.5, 0.5, '(GainFilterBase: no Fourier diagnostics)',
                    ha='center', va='center')

        ax = axes[1, 1]
        if P_high is not None and n_tod is not None:
            freqs = np.fft.rfftfreq(n_tod, d=1/50.0)
            ax.semilogx(freqs[1:], P_high[1:], lw=0.7)
            ax.set_xlabel('frequency [Hz]'); ax.set_ylabel(r'$P_{\rm high}(f)$')
            ax.set_ylim(-0.05, 1.05); ax.set_title('Prior leakage weight')
            ax.grid(True, which='both', alpha=0.3)
            if atmos_templates is not None:
                for k, lab in enumerate((atmos_names if atmos_names is not None else [f'tmpl{i}' for i in range(atmos_templates.shape[0])])):
                    tf = np.abs(np.fft.rfft(atmos_templates[k]))
                    if tf[1:].max() > 0:
                        peak = freqs[1:][np.argmax(tf[1:])]
                        ax.axvline(peak, color=f'C{k}', ls='--', lw=0.6,
                                   label=f'{lab} peak: {peak:.3g} Hz')
                ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, '(no prior in use)', ha='center', va='center')

        fig.tight_layout()
        fname = os.path.join(self.debug_dir,
                             f'feed{feed:02d}_band{iband}_call{n:03d}.png')
        fig.savefig(fname, dpi=80)
        plt.close(fig)


    def __call__(self, data : np.ndarray,
                system_temperature : np.ndarray,
                median_offsets : np.ndarray,
                feed : int,
                elevation : np.ndarray = None,
                azimuth : np.ndarray = None,
                **kwargs) -> np.ndarray:

        normed_data = data/median_offsets[...,np.newaxis]
        normed_data -= np.nanmean(normed_data,axis=-1, keepdims=True)
        normed_data[np.isnan(normed_data)] = 0
        n_tod = data.shape[-1]

        gain_solution = np.zeros(n_tod)
        gain_temp, mask = self.gain_subtraction_fit(normed_data, system_temperature)
        if gain_temp.shape[0] == n_tod:
            gain_solution[...] = gain_temp

        # If pointing tracks are supplied, identify systematics in the gain
        # solution that share fixed shapes with airmass (1/sin El) and an
        # azimuth gradient (delta_az with 0/360 wrap), split them off so they
        # are not attributed to instrumental gain.
        atmos_solution = np.zeros(n_tod)
        T_atm, atmos_names = build_atmos_templates(elevation=elevation, azimuth=azimuth)
        b_vec = None
        if T_atm is not None and T_atm.shape[1] == n_tod:
            M = T_atm @ T_atm.T
            r = T_atm @ gain_solution
            try:
                b_vec, *_ = np.linalg.lstsq(M, r, rcond=1e-8)
                atmos_solution = b_vec @ T_atm
                gain_solution = gain_solution - atmos_solution
            except np.linalg.LinAlgError:
                b_vec = None

        # Rescale both components back to data units and subtract.
        total = (gain_solution + atmos_solution)[np.newaxis, np.newaxis, :] * median_offsets[...,np.newaxis]
        residual = data - total

        if self.debug_dir:
            for iband in range(residual.shape[0]):
                self._save_debug_plot(feed=feed, iband=iband,
                                       gain_solution=gain_solution,
                                       atmos_solution=atmos_solution,
                                       atmos_templates=T_atm,
                                       atmos_names=atmos_names,
                                       residual_band=residual[iband],
                                       n_tod=n_tod,
                                       b_vec=b_vec)
            self._debug_call_count += 1

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

        # Orthogonalise the Tsys columns (0, 1) against the gain column (2)
        # over the good channels. See the analogous step in
        # gain_subtract_fit_with_prior for the rationale: without this,
        # nearly-constant Tsys makes 1/Tsys and the constant gain template
        # degenerate, and the freq-flat amplitude gets split between m and g.
        gain_col = templates[:, 2:3]                                     # (Nf, 1)
        gain_norm_sq = float(gain_col.T @ gain_col)
        if gain_norm_sq > 0:
            proj = (templates[:, :2].T @ gain_col) / gain_norm_sq        # (2, 1)
            templates[:, :2] = templates[:, :2] - gain_col @ proj.T      # (Nf, 2)

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

    def __init__(self, end_cut = 100, noise_prior_path='ancillary_data/noise_priors/noise_priors.npy', debug_dir=None):
        super().__init__(debug_dir=debug_dir)
        self.end_cut = end_cut

        self.noise_priors = np.load(noise_prior_path, allow_pickle=True).flatten()[0]

    def __call__(self, data : np.ndarray,
                system_temperature : np.ndarray,
                median_offsets : np.ndarray,
                feed : int,
                alpha=-1,
                sigma_red=0.5,
                elevation : np.ndarray = None,
                azimuth : np.ndarray = None) -> np.ndarray:

        normed_data = data/median_offsets[...,np.newaxis]
        normed_data -= np.nanmean(normed_data,axis=-1, keepdims=True)
        normed_data[np.isnan(normed_data)] = 0
        n_tod = data.shape[-1]

        sigma_red, alpha = self.noise_priors[feed]
        residual = np.zeros_like(normed_data)
        mask = np.zeros(median_offsets.shape, dtype=bool)
        gains = []
        atmos_templates, atmos_names = build_atmos_templates(elevation=elevation, azimuth=azimuth)
        for iband in range(normed_data.shape[0]):
            # sigma_red from noise_priors is in K (fit on binned_data in K),
            # but the gain time series this filter works in is dG/G (normed).
            # Convert so the prior is in the same units as the signal it
            # constrains; this is what makes 1/cov comparable to z and lets
            # the Wiener filter suppress non-1/f-shaped signal (transients,
            # source crossings) from being absorbed into the gain estimate.
            typical_Tsys = float(np.nanmedian(median_offsets[iband]))
            sigma_red_normed = sigma_red / typical_Tsys if typical_Tsys > 0 else sigma_red
            gain_solution = np.zeros(n_tod)
            atmos_solution = np.zeros(n_tod)
            gain_temp, mbest, gain_templates, sys_templates, atmos_temp, dbg = self.gain_subtract_fit_with_prior(
                normed_data[iband:iband+1],
                system_temperature[iband:iband+1],
                sigma_red_normed, alpha,
                atmos_templates=atmos_templates)

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

            if self.debug_dir:
                self._save_debug_plot(feed=feed, iband=iband,
                                       gain_solution=gain_solution,
                                       atmos_solution=atmos_solution,
                                       atmos_templates=atmos_templates,
                                       atmos_names=atmos_names,
                                       residual_band=residual[iband],
                                       P_high=dbg['P_high'],
                                       ghat_ft=dbg['a_best_fit_ft'],
                                       RHS_ft=dbg['RHS_ft'],
                                       n_tod=n_tod,
                                       b_vec=dbg['b_vec'])

        if self.debug_dir:
            self._debug_call_count += 1
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

    def gain_subtract_fit_with_prior(self, data_normed : np.ndarray, system_temperature : np.ndarray, sigma_red : float, alpha : float, atmos_templates : np.ndarray = None, return_all : bool = False, sample_rate : float = 50.0) -> np.ndarray:
        """
        Gain subtraction with the 1/f noise prior. If atmos_templates is
        passed (shape (K, ntod), one row per fixed-shape systematic, each
        already mean-removed) the K scalar amplitudes are jointly fit with
        the gain time series and split off into the returned atmosphere
        track.
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

        # Orthogonalise the Tsys columns against the constant gain template
        # over the good channels. Without this, when Tsys varies little across
        # the band, [1/Tsys, v/Tsys] each carry a large constant component
        # that is degenerate with the gain template -- and the freq-flat
        # amplitude in the data ends up split between m and g rather than
        # being attributed cleanly to the gain coefficient.
        gain_norm_sq = float(gain_templates.T @ gain_templates)
        if gain_norm_sq > 0:
            proj = (sys_templates.T @ gain_templates) / gain_norm_sq   # (2, 1)
            sys_templates = sys_templates - gain_templates @ proj.T    # (Nf, 2)

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
        b_vec = None
        P_high = None
        if (atmos_templates is not None
                and atmos_templates.shape[-1] == ntod
                and z_scalar > 0):
            # Joint MAP fit of gain (1/f prior) and K systematics templates
            # (flat priors). Each template is one row of atmos_templates, with
            # a single scalar amplitude. The K-by-K normal equations weight the
            # template inner products by P_high(f) = (1/cov) / (z + 1/cov),
            # the "leak" of the prior past the Wiener filter.
            T_ft = np.fft.rfft(atmos_templates, axis=-1)  # (K, n_fft)
            n_fft = T_ft.shape[-1]

            wts = np.full(n_fft, 2.0)
            wts[0] = 1.0
            if ntod % 2 == 0:
                wts[-1] = 1.0

            P_high = np.where(np.isinf(inv_cov), 1.0,
                              inv_cov / (z_scalar + inv_cov))
            weighted = wts * P_high  # (n_fft,)

            M = np.real(np.einsum('kf,jf,f->kj', np.conj(T_ft), T_ft, weighted))
            r = np.real(np.einsum('kf,f,f->k', np.conj(T_ft), RHS[0], weighted)) / z_scalar

            # lstsq with a relative rcond is robust to near-singular M
            # (e.g. constant-elevation scans collapsing airmass to ~zero);
            # solve would otherwise return huge values that destructively
            # cancel between the gain and atmos solutions.
            try:
                b_vec, *_ = np.linalg.lstsq(M, r, rcond=1e-8)
                atmos_amplitude = b_vec @ atmos_templates
                b_T_ft = b_vec @ T_ft
                RHS_g = RHS - z_scalar * b_T_ft[np.newaxis, :]
            except np.linalg.LinAlgError:
                RHS_g = RHS
        else:
            RHS_g = RHS

        a_best_fit_ft = RHS_g / (z + 1./cov)
        a_best_fit    = np.fft.irfft(a_best_fit_ft, n=ntod)

        m_best_fit = np.linalg.pinv(sys_templates.T.dot(sys_templates)).dot(sys_templates.T).dot(data_normed - gain_templates.dot(a_best_fit))

        debug = {'a_best_fit_ft': a_best_fit_ft[0],
                 'RHS_ft': RHS[0],
                 'P_high': P_high,
                 'b_vec': b_vec}
        return a_best_fit, m_best_fit, gain_templates, sys_templates, atmos_amplitude, debug
