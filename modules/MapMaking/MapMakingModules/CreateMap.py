# CreateMap.py
#
# Description:
#  CreateMap class works with a ReadData class to create COMAP sky maps using
#  destriping.

import hashlib
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import toml


@dataclass
class MapJob:
    """Container for one map-making job configuration."""

    label: str
    files: List[str]
    feeds: List[int]
    output_filename: str


class CreateMap:
    """
    Unified map-making entry point.

    This class can run a single map, day/night splits, per-feed maps, alternate
    scan splits, and arbitrary user supplied split groups.
    """

    def __init__(
        self,
        map_name: str = "Unknown",
        band: int = 0,
        output_dir: str = "outputs/",
        source: str = "Unknown",
        source_group: str = "Unknown",
        wcs_def: str = None,
        wcs_def_file: str = None,
        offset_length: int = 100,
        feeds: list = [i for i in range(1, 20)],
        tod_data_name: str = "level2/binned_filtered_data",
        database_file: str = "databases/COMAP_manchester.db",
        file_list_name: str = "file_list.txt",
        sigma_red_cutoff: float = 0.4,
        calib_path: str = None,
        planck_30_path: str = None,
        lambda_ridge: float = 1e-4,
        jackknife_odd_even: str = None,
        n_processes: int = 1,
        split_mode: str = "none",
        sun_elevation_threshold: float = 0.0,
        split_groups: Dict[str, Sequence[str]] = None,
        config_dir: str = "modules/MapMaking/map_making_configs",
        **kwargs,
    ) -> None:
        logging.info("Initializing CreateMap")

        logger = logging.getLogger()
        log_file = None
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_file = handler.baseFilename

        self.file_list_name = file_list_name
        self.map_name = map_name
        self.tod_data_name = tod_data_name
        self.band = band
        self.output_dir = output_dir
        self.source = source
        self.source_group = source_group
        self.n_processes = n_processes
        self.split_mode = split_mode
        self.sun_elevation_threshold = sun_elevation_threshold
        self.split_groups = split_groups or {}

        self.feeds = list(feeds)
        self.mapmaking_dir = Path(__file__).parent.parent.absolute()
        self.working_dir = os.getcwd()
        self.config_dir = Path(self.working_dir) / config_dir
        self.config_dir.mkdir(parents=True, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        self.base_parameters = {
            "band": band,
            "wcs_def": wcs_def,
            "source": source,
            "source_group": source_group,
            "offset_length": offset_length,
            "feeds": self.feeds,
            "map_name": map_name,
            "tod_data_name": tod_data_name,
            "log_file_name": log_file,
            "sigma_red_cutoff": sigma_red_cutoff,
            "database": f"{self.working_dir}/{database_file}",
            "output_dir": output_dir,
            "lambda_ridge": lambda_ridge,
            "planck_30_path": planck_30_path,
            "calib_path": calib_path,
            "jackknife_odd_even": jackknife_odd_even,
            "split_mode": split_mode,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }

    def _job_token(self, label: str, file_list: Sequence[str], feeds: Sequence[int]) -> str:
        """Create deterministic token to keep map-making configs unique."""
        payload = {
            "label": label,
            "files": list(file_list),
            "feeds": list(feeds),
            "parameters": self.base_parameters,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return digest[:12]

    def _write_job_files(self, job: MapJob) -> Path:
        token = self._job_token(job.label, job.files, job.feeds)
        config_file = self.config_dir / f"config_{self.map_name}_{job.label}_{token}.toml"
        file_list_path = self.config_dir / f"file_list_{self.map_name}_{job.label}_{token}.txt"

        parameters = dict(self.base_parameters)
        parameters["feeds"] = list(job.feeds)
        parameters["output_filename"] = job.output_filename
        parameters["file_list"] = str(file_list_path)

        with open(config_file, "w") as f:
            toml.dump(parameters, f)
        with open(file_list_path, "w") as f:
            for file in job.files:
                f.write(file + "\n")

        logging.info("MapMaking config written: %s", config_file)
        logging.info("MapMaking file list written: %s", file_list_path)
        return config_file

    def _run_mapmaking_job(self, config_file: Path, n_files: int, label: str) -> None:
        command = [
            "mpirun",
            "-n",
            str(self.n_processes),
            "python",
            f"{self.mapmaking_dir}/run_map_making.py",
            str(config_file),
        ]
        file_list_str = f"CREATEMAP: MAPPING {n_files} files for {self.source} band {self.band} ({label})"
        logging.info(file_list_str)
        print(" ".join(command))

        result = subprocess.run(
            command,
            shell=False,
            cwd=self.working_dir,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        if result.returncode != 0:
            logging.error("Mapmaking command failed (%s): %s", label, result.stderr)
            raise RuntimeError(
                f"Map-making job '{label}' failed with return code {result.returncode}."
            )
        logging.info("Mapmaking command finished (%s)", label)

    def _split_day_night(self, file_list: Sequence[str]) -> Dict[str, List[str]]:
        day_list = []
        night_list = []

        import h5py
        from astropy.coordinates import AltAz, EarthLocation, get_sun
        from astropy.time import Time
        from modules.utils.Coordinates import comap_latitude, comap_longitude

        comap_loc = EarthLocation.from_geodetic(comap_longitude, comap_latitude, 1222.0)
        for filename in file_list:
            with h5py.File(filename, "r") as h:
                utc_start = h["comap"].attrs["utc_start"]
                time = Time.strptime(utc_start, "%Y-%m-%d-%H:%M:%S")
                sun_alt = get_sun(time).transform_to(AltAz(obstime=time, location=comap_loc)).alt.deg
                if sun_alt > self.sun_elevation_threshold:
                    day_list.append(filename)
                else:
                    night_list.append(filename)

        return {"day": day_list, "night": night_list}

    def _split_alternate_scans(self, file_list: Sequence[str]) -> Dict[str, List[str]]:
        return {
            "scan_a": [f for i, f in enumerate(file_list) if i % 2 == 0],
            "scan_b": [f for i, f in enumerate(file_list) if i % 2 == 1],
        }

    def _build_jobs(self, file_list: Sequence[str]) -> Iterable[MapJob]:
        feeds_str = "-".join([f"{i:02d}" for i in self.feeds])

        if self.split_mode == "none":
            yield MapJob(
                label="all",
                files=list(file_list),
                feeds=self.feeds,
                output_filename=f"{self.map_name}_band{self.band:02d}_feed{feeds_str}.fits",
            )
            return

        if self.split_mode == "per_feed":
            for feed in self.feeds:
                yield MapJob(
                    label=f"feed{feed:02d}",
                    files=list(file_list),
                    feeds=[feed],
                    output_filename=f"{self.map_name}_band{self.band:02d}_feed{feed:02d}.fits",
                )
            return

        if self.split_mode == "day_night":
            split = self._split_day_night(file_list)
        elif self.split_mode == "alternate_scans":
            split = self._split_alternate_scans(file_list)
        elif self.split_mode == "custom":
            split = {k: list(v) for k, v in self.split_groups.items()}
        else:
            raise ValueError(
                f"Unknown split_mode '{self.split_mode}'. Expected one of: "
                "none, per_feed, day_night, alternate_scans, custom"
            )

        for label, files in split.items():
            if not files:
                logging.warning("Skipping empty split '%s'.", label)
                continue
            yield MapJob(
                label=label,
                files=files,
                feeds=self.feeds,
                output_filename=f"{self.map_name}_band{self.band:02d}_feed{feeds_str}_{label}.fits",
            )

    def run(self, file_list: list) -> None:
        logging.info("Running CreateMap")
        jobs = list(self._build_jobs(file_list))
        for job in jobs:
            config_file = self._write_job_files(job)
            self._run_mapmaking_job(config_file, len(job.files), job.label)
