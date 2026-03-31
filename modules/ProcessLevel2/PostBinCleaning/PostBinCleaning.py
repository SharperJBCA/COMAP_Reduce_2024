# PostBinCleaning.py
#
# Second-pass cleaning on binned Level-2 data:
#   1. Re-fit and subtract the atmospheric model (offset + tau/sin(el)) from
#      the band-averaged binned data, where S/N is much higher than in the
#      original per-channel fit.
#   2. Subtract a median filter to remove residual large-timescale drifts.
#
# This module operates on the output of GainFilterAndBin and overwrites the
# binned_filtered_data dataset in-place.

import logging

import numpy as np
from scipy.linalg import solve
from tqdm import tqdm

from modules.pipeline_control.Pipeline import RetryH5PY, BaseCOMAPModule
from modules.SQLModule.SQLModule import COMAPData
from modules.utils.constants import NFEEDS, NBANDS, GROUND_FEED
from modules.utils.median_filter import medfilt


class PostBinCleaning(BaseCOMAPModule):
    """Re-clean binned filtered data: atmosphere re-fit + median filter."""

    def __init__(
        self,
        target_dataset: str = "level2/binned_filtered_data",
        medfilt_length: int = 50,
        overwrite: bool = False,
        output_group: str = "level2/post_bin_cleaning",
        outlier_sigma: float = 5.0,
    ) -> None:
        super().__init__()
        self.target_dataset = target_dataset
        self.medfilt_length = medfilt_length
        self.overwrite = overwrite
        self.output_group = output_group
        self.outlier_sigma = outlier_sigma

    # ------------------------------------------------------------------ fit
    @staticmethod
    def _fit_atmosphere_scan(data: np.ndarray, el: np.ndarray):
        """Fit offset + tau/sin(el) to a 1-D TOD segment.

        Parameters
        ----------
        data : (n_samples,) array
        el   : (n_samples,) array, elevation in **radians**

        Returns
        -------
        model : (n_samples,) array  —  the fitted atmospheric signal
        """
        mask = np.isfinite(data) & np.isfinite(el) & (el > 0)
        if mask.sum() < 3:
            return np.zeros_like(data)

        el_range = np.abs(np.nanmax(el[mask]) - np.nanmin(el[mask]))
        if el_range < np.radians(0.2):
            # Not enough elevation variation — just remove the median
            model = np.full_like(data, np.nanmedian(data[mask]))
            model[~mask] = 0.0
            return model

        A = np.ones((mask.sum(), 2), dtype=np.float64)
        A[:, 1] = 1.0 / np.sin(el[mask])
        try:
            coeffs = solve(A.T @ A, A.T @ data[mask])
        except np.linalg.LinAlgError:
            return np.full_like(data, np.nanmedian(data[mask]))

        model = np.zeros_like(data)
        model[mask] = coeffs[0] + coeffs[1] / np.sin(el[mask])
        return model

    # -------------------------------------------------------------- process
    @staticmethod
    def _auto_rms(data: np.ndarray) -> float:
        """Differenced RMS estimator (insensitive to slow drifts)."""
        N = len(data) // 2 * 2
        return float(np.nanstd(data[:N:2] - data[1:N:2]) / np.sqrt(2))

    def _clean_feed(self, data, elevation, scan_edges):
        """Clean one feed's binned data in-place.

        Parameters
        ----------
        data : (n_bands, n_channels, n_tod) array  — modified in-place
        elevation : (n_tod,) array in degrees
        scan_edges : (n_scans, 2) array
        """
        el_rad = np.radians(elevation)
        n_bands, n_channels, _ = data.shape

        for iband in range(n_bands):
            for ichan in range(n_channels):
                tod = data[iband, ichan, :]

                for scan_start, scan_end in scan_edges:
                    seg = tod[scan_start:scan_end].copy()
                    el_seg = el_rad[scan_start:scan_end]

                    if np.all(np.isnan(seg)):
                        continue

                    # 1) Median filter to get a smooth baseline
                    finite = np.isfinite(seg)
                    if finite.sum() <= self.medfilt_length:
                        continue
                    seg_for_med = seg.copy()
                    seg_for_med[~finite] = 0.0
                    med = np.array(
                        medfilt.medfilt(seg_for_med.astype(np.float64), self.medfilt_length)
                    ).astype(seg.dtype)

                    # 2) Flag outliers using residuals from the median filter
                    resid = seg - med
                    rms = self._auto_rms(resid[finite])
                    if rms <= 0 or not np.isfinite(rms):
                        continue
                    outlier = np.abs(resid) > self.outlier_sigma * rms
                    clean = finite & ~outlier

                    # 3) Atmosphere re-fit on clean (non-outlier) data
                    seg_for_atm = seg.copy()
                    seg_for_atm[~clean] = np.nan
                    atm_model = self._fit_atmosphere_scan(seg_for_atm, el_seg)
                    seg -= atm_model

                    # 4) Subtract median filter from atmosphere-corrected data
                    #    Recompute median after atmosphere removal
                    seg_for_med2 = seg.copy()
                    seg_for_med2[~finite] = 0.0
                    med2 = np.array(
                        medfilt.medfilt(seg_for_med2.astype(np.float64), self.medfilt_length)
                    ).astype(seg.dtype)
                    seg[finite] -= med2[finite]

                    tod[scan_start:scan_end] = seg

    # ------------------------------------------------------------------ run
    def run(self, file_info: COMAPData) -> None:
        if self.already_processed(file_info, self.output_group, self.overwrite):
            return

        logging.info("PostBinCleaning: processing %s", file_info.level2_path)

        with RetryH5PY(file_info.level2_path, "a") as f:
            if self.target_dataset not in f:
                logging.warning("PostBinCleaning: %s not found — skipping", self.target_dataset)
                return

            feeds = f["spectrometer/feeds"][:]
            scan_edges = f["level2/scan_edges"][...]
            data_full = f[self.target_dataset][...]  # (NFEEDS, NBANDS, n_ch, n_tod)

            for ifeed, feed in enumerate(tqdm(feeds, desc="PostBinCleaning")):
                if feed == GROUND_FEED:
                    continue

                elevation = f["spectrometer/pixel_pointing/pixel_el"][ifeed, :]
                self._clean_feed(data_full[feed - 1], elevation, scan_edges)

            # Overwrite dataset
            del f[self.target_dataset]
            f.create_dataset(self.target_dataset, data=data_full)

            # Mark as processed
            if self.output_group not in f:
                f.create_group(self.output_group)
            f[self.output_group].attrs["status"] = "complete"

        logging.info("PostBinCleaning: done")
