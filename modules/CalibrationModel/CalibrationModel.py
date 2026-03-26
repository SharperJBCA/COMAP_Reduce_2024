# CalibrationModel.py
#
# Top-level pipeline module that computes calibration gain factors from
# calibrator flux measurements and fits a temporal model for use in map-making.
#
# Runs between Level-2 processing and MapMaking in the pipeline.

import os
import json
import logging
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np

from modules.SQLModule.SQLModule import SQLModule, CalibrationFlux, CalibrationModelFit, db
from modules.CalibrationModel.temporal_model import (
    compute_model_flux, fit_polynomial, fit_mean, prepare_nearest,
)

# COMAP band center frequencies in GHz (shape [4, 2])
FREQUENCIES_GHZ = np.array([[26.5, 27.5],
                             [28.5, 29.5],
                             [30.5, 31.5],
                             [32.5, 33.5]])


class CalibrationModel:
    """Compute and store calibration gain factors from calibrator observations.

    This module:
    1. Reads per-element flux measurements from CalibrationFlux SQL table
    2. Computes reference (model) flux from CaliModels
    3. Computes gain_ratio = measured / model per feed/band/channel
    4. Fits a temporal model (polynomial, nearest-neighbor, or mean)
    5. Stores results in CalibrationModelFit SQL table
    6. Writes a backward-compatible .npy file
    """

    def __init__(self,
                 source='TauA',
                 min_obsid=7000,
                 max_obsid=1000000,
                 model_type='polynomial',
                 poly_degree=2,
                 output_npy_path='calibration/calibration_model.npy',
                 chi2_max=10.0,
                 gain_min=0.3,
                 gain_max=3.0,
                 min_observations=10,
                 overwrite=False):
        self.source = source
        self.min_obsid = int(min_obsid)
        self.max_obsid = int(max_obsid)
        self.model_type = model_type
        self.poly_degree = int(poly_degree)
        self.output_npy_path = output_npy_path
        self.chi2_max = float(chi2_max)
        self.gain_min = float(gain_min)
        self.gain_max = float(gain_max)
        self.min_observations = int(min_observations)
        self.overwrite = overwrite

        # Populated during run
        self._gain_data = None  # dict[(feed,band,channel)] -> (mjds, gains)

    def run(self):
        logging.info("CalibrationModel: starting (source=%s, model=%s)",
                     self.source, self.model_type)

        self._compute_gain_ratios()

        if self._gain_data is None or len(self._gain_data) == 0:
            logging.warning("CalibrationModel: no valid gain data found")
            return

        self._fit_model()
        self._write_npy()
        self._plot_diagnostics()

        logging.info("CalibrationModel: complete")

    def _compute_gain_ratios(self):
        """Query CalibrationFlux, compute model flux, derive gain ratios."""
        rows = db.query_calibration_flux(
            source=self.source,
            min_obsid=self.min_obsid,
            max_obsid=self.max_obsid,
        )
        logging.info("CalibrationModel: found %d flux measurements", len(rows))

        if not rows:
            self._gain_data = {}
            return

        # Group by (feed, band, channel)
        grouped = defaultdict(list)  # key -> list of (mjd, measured_flux, chi2)
        for r in rows:
            grouped[(r.feed, r.band, r.channel)].append(
                (r.mjd, r.measured_flux, r.chi2)
            )

        gain_data = {}
        for (feed, band, channel), measurements in grouped.items():
            mjds = []
            gains = []
            for mjd, meas_flux, chi2 in measurements:
                if meas_flux is None or not np.isfinite(meas_flux):
                    continue
                if chi2 is not None and chi2 > self.chi2_max:
                    continue

                # Compute reference flux from model
                freq = FREQUENCIES_GHZ[band, channel]
                try:
                    model_flux = compute_model_flux(self.source, mjd, freq)
                except (ValueError, RuntimeError) as e:
                    logging.debug("CalibrationModel: skipping mjd=%.2f: %s", mjd, e)
                    continue

                if not np.isfinite(model_flux) or model_flux <= 0:
                    continue

                gain = meas_flux / model_flux
                if not np.isfinite(gain):
                    continue
                if gain < self.gain_min or gain > self.gain_max:
                    continue

                mjds.append(mjd)
                gains.append(gain)

            if len(mjds) >= self.min_observations:
                gain_data[(feed, band, channel)] = (
                    np.array(mjds), np.array(gains)
                )

        self._gain_data = gain_data
        logging.info("CalibrationModel: %d elements with sufficient data (>=%d obs)",
                     len(gain_data), self.min_observations)

    def _fit_model(self):
        """Fit the temporal model for each feed/band/channel and store in SQL."""
        now = datetime.now(timezone.utc).isoformat()

        for (feed, band, channel), (mjds, gains) in self._gain_data.items():
            if self.model_type == "polynomial":
                coeffs, rms = fit_polynomial(mjds, gains, degree=self.poly_degree)
                db.upsert_calibration_model(
                    feed=feed, band=band, channel=channel,
                    source=self.source,
                    model_type="polynomial",
                    poly_coeffs=json.dumps(coeffs),
                    nearest_mjds=None,
                    nearest_gains=None,
                    fit_rms=rms,
                    n_observations=len(mjds),
                    updated_at=now,
                )
            elif self.model_type == "mean":
                mean_gain, rms = fit_mean(mjds, gains)
                db.upsert_calibration_model(
                    feed=feed, band=band, channel=channel,
                    source=self.source,
                    model_type="mean",
                    poly_coeffs=json.dumps([mean_gain]),
                    nearest_mjds=None,
                    nearest_gains=None,
                    fit_rms=rms,
                    n_observations=len(mjds),
                    updated_at=now,
                )
            elif self.model_type == "nearest":
                sorted_mjds, sorted_gains = prepare_nearest(mjds, gains)
                db.upsert_calibration_model(
                    feed=feed, band=band, channel=channel,
                    source=self.source,
                    model_type="nearest",
                    poly_coeffs=None,
                    nearest_mjds=json.dumps(sorted_mjds.tolist()),
                    nearest_gains=json.dumps(sorted_gains.tolist()),
                    fit_rms=None,
                    n_observations=len(mjds),
                    updated_at=now,
                )
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")

        logging.info("CalibrationModel: fitted %d elements", len(self._gain_data))

    def _write_npy(self):
        """Write backward-compatible .npy dict for ReadData consumption."""
        calib_dict = db.load_calibration_model()
        if calib_dict is None:
            logging.warning("CalibrationModel: no model to write")
            return

        os.makedirs(os.path.dirname(self.output_npy_path) or ".", exist_ok=True)
        np.save(self.output_npy_path, calib_dict, allow_pickle=True)
        logging.info("CalibrationModel: wrote %s", self.output_npy_path)

    def _plot_diagnostics(self):
        """Plot gain ratio vs MJD with model overlay, one panel per band."""
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        from modules.CalibrationModel.temporal_model import evaluate_model

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
        axes = axes.ravel()

        for iband in range(4):
            ax = axes[iband]
            has_data = False

            for (feed, band, channel), (mjds, gains) in self._gain_data.items():
                if band != iband:
                    continue
                has_data = True
                ax.scatter(mjds, gains, s=4, alpha=0.3, rasterized=True)

            if has_data and self.model_type in ("polynomial", "mean"):
                # Plot model curve for feed=1 as representative
                key = None
                for (feed, band, channel) in self._gain_data:
                    if band == iband:
                        key = (feed, band, channel)
                        break
                if key is not None:
                    mjd_range = np.linspace(
                        min(self._gain_data[key][0]),
                        max(self._gain_data[key][0]),
                        200
                    )
                    calib = db.load_calibration_model()
                    if calib is not None:
                        params = calib["model_params"][key[0]-1, key[1], key[2]]
                        if params is not None:
                            model_vals = [evaluate_model(m, calib["model_type"], params)
                                          for m in mjd_range]
                            ax.plot(mjd_range, model_vals, 'r-', lw=2, label='Model')
                            ax.legend(fontsize=8)

            ax.set_xlabel('MJD')
            ax.set_ylabel('Gain ratio (measured/model)')
            ax.set_title(f'Band {iband} ({FREQUENCIES_GHZ[iband, 0]:.1f}-{FREQUENCIES_GHZ[iband, 1]:.1f} GHz)')
            ax.grid(alpha=0.3)

        fig.suptitle(f'Calibration gain: {self.source} ({self.model_type})')

        out_dir = os.path.dirname(self.output_npy_path) or "."
        out_plot = os.path.join(out_dir, f'calibration_diagnostics_{self.source}.png')
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(out_plot, dpi=150)
        plt.close(fig)
        logging.info("CalibrationModel: saved diagnostic plot %s", out_plot)
