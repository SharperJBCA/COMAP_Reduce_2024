# temporal_model.py
#
# Pure functions for calibration temporal models: polynomial, nearest-neighbor, mean.
# These are independent of the pipeline framework and can be tested in isolation.

import numpy as np
from typing import Optional, Tuple

# Reference epoch for normalized time: t = (mjd - MJD_REF) / 365.25 (in years)
MJD_REF = 59000.0


def _normalized_time(mjd):
    """Convert MJD to normalized time in years relative to MJD_REF."""
    return (np.asarray(mjd, dtype=np.float64) - MJD_REF) / 365.25


# --------------- model evaluation ---------------

def evaluate_model(mjd: float, model_type: str, params: dict) -> float:
    """Evaluate a calibration model at a given MJD.

    Parameters
    ----------
    mjd : float
        Modified Julian Date of the observation.
    model_type : str
        One of "polynomial", "nearest", "mean".
    params : dict
        Model parameters. Keys depend on model_type:
        - "polynomial" / "mean": {"coeffs": list[float]}
        - "nearest": {"mjds": array, "gains": array}

    Returns
    -------
    float
        Calibration gain factor at the given MJD.
    """
    if model_type == "polynomial" or model_type == "mean":
        return polynomial_model(mjd, params["coeffs"])
    elif model_type == "nearest":
        return nearest_model(mjd, params["mjds"], params["gains"])
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def polynomial_model(mjd, coeffs) -> float:
    """Evaluate polynomial: gain = c0 + c1*t + c2*t^2 + ...

    Parameters
    ----------
    mjd : float or array
        MJD value(s).
    coeffs : list or array
        Polynomial coefficients [c0, c1, c2, ...].

    Returns
    -------
    float or array
        Gain value(s).
    """
    t = _normalized_time(mjd)
    return float(np.polyval(coeffs[::-1], t))


def nearest_model(mjd, ref_mjds, ref_gains) -> float:
    """Return gain of closest calibrator observation in time.

    Parameters
    ----------
    mjd : float
        MJD of the science observation.
    ref_mjds : array-like
        Sorted MJDs of calibrator observations.
    ref_gains : array-like
        Gain ratios at each calibrator observation.

    Returns
    -------
    float
        Gain from the nearest calibrator observation.
    """
    ref_mjds = np.asarray(ref_mjds, dtype=np.float64)
    ref_gains = np.asarray(ref_gains, dtype=np.float64)
    idx = np.argmin(np.abs(ref_mjds - mjd))
    return float(ref_gains[idx])


# --------------- model fitting ---------------

def fit_polynomial(mjds, gains, weights=None, degree=2) -> Tuple[list, float]:
    """Fit polynomial to gain vs time.

    Parameters
    ----------
    mjds : array-like
        MJDs of calibrator observations.
    gains : array-like
        Measured gain ratios.
    weights : array-like, optional
        Weights for each point (e.g., 1/error^2).
    degree : int
        Polynomial degree.

    Returns
    -------
    coeffs : list
        Polynomial coefficients [c0, c1, c2, ...] (constant first).
    rms : float
        RMS of residuals.
    """
    t = _normalized_time(mjds)
    gains = np.asarray(gains, dtype=np.float64)

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
    else:
        w = np.ones_like(gains)

    # numpy polyfit returns highest-degree-first; we store constant-first
    poly = np.polyfit(t, gains, degree, w=np.sqrt(w))
    coeffs = poly[::-1].tolist()  # [c0, c1, ..., c_deg]

    residuals = gains - np.polyval(poly, t)
    rms = float(np.sqrt(np.mean(residuals**2)))

    return coeffs, rms


def fit_mean(mjds, gains, weights=None) -> Tuple[float, float]:
    """Compute weighted mean gain.

    Returns
    -------
    mean_gain : float
    rms : float
        RMS of residuals from mean.
    """
    gains = np.asarray(gains, dtype=np.float64)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        mean_gain = float(np.average(gains, weights=w))
    else:
        mean_gain = float(np.mean(gains))

    rms = float(np.sqrt(np.mean((gains - mean_gain)**2)))
    return mean_gain, rms


def prepare_nearest(mjds, gains) -> Tuple[np.ndarray, np.ndarray]:
    """Sort calibrator observations by MJD for nearest-neighbor lookup.

    Returns
    -------
    sorted_mjds : ndarray
    sorted_gains : ndarray
    """
    mjds = np.asarray(mjds, dtype=np.float64)
    gains = np.asarray(gains, dtype=np.float64)
    order = np.argsort(mjds)
    return mjds[order], gains[order]


def compute_model_flux(source: str, mjd: float, frequencies_ghz: np.ndarray) -> np.ndarray:
    """Compute reference flux density (Jy) for a calibrator source.

    Uses the flux models from modules/utils/CaliModels.py.

    Parameters
    ----------
    source : str
        Calibrator name: "TauA", "CasA", "CygA", "Jupiter".
    mjd : float
        MJD of the observation.
    frequencies_ghz : array
        Frequencies in GHz, shape matching the flux array.

    Returns
    -------
    model_flux : ndarray
        Model flux density in Jy, same shape as frequencies_ghz.
    """
    from modules.utils.CaliModels import TauAFluxModel, CasAFluxModel, CygAFlux

    source_lower = source.lower().replace(" ", "")

    if source_lower == "taua":
        model = TauAFluxModel()
        return model(frequencies_ghz, mjd)
    elif source_lower == "casa":
        model = CasAFluxModel()
        return model(frequencies_ghz, mjd)
    elif source_lower == "cyga":
        return CygAFlux(frequencies_ghz, mjd=[mjd])
    else:
        raise ValueError(f"No flux model available for source: {source}")
