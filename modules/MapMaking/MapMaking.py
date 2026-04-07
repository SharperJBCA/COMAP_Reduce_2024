# MapMaking.py
#
# This module orchestrates map-making jobs by:
#   1. reading pipeline configuration,
#   2. resolving Level-2 file lists from DB or explicit obsid lists,
#   3. dispatching each configured map module with a consistent file list.

import importlib
import logging
import os

import h5py
import numpy as np
from tqdm import tqdm

from modules.SQLModule.SQLModule import db
from modules.parameter_files.parameter_files import read_parameter_file


def get_file_list(target_source_group=None, target_source=None, min_obs_id=7000, max_obs_id=100000, obsid_list=None):
    """Resolve and validate Level-2 files for continuum map making."""
    if obsid_list is not None:
        query_source_group_list = db.query_obsid_list(obsid_list, return_dict=False)
    else:
        query_source_group_list = db.query_source_group_list(
            target_source_group,
            source=target_source,
            min_obsid=min_obs_id,
            max_obsid=max_obs_id,
            return_dict=False,
        )

    n_no_level2 = sum(1 for _, f in query_source_group_list.items() if f.level2_path is None)
    source_file_list = [f.level2_path for _, f in query_source_group_list.items() if f.level2_path is not None]
    source_obsids = [f.obsid for _, f in query_source_group_list.items() if f.level2_path is not None]

    final_files = []
    excluded_bad_flags = 0
    excluded_missing_datasets = 0
    excluded_open_error = 0
    for filename, obsid in zip(tqdm(source_file_list, desc="Checking final file list"), source_obsids):
        quality_flags = db.get_quality_flags(obsid)
        if all([not v.is_good for _, v in quality_flags.items()]):
            logging.warning("Excluding obsid %d: all quality flags are bad", obsid)
            excluded_bad_flags += 1
            continue
        try:
            with h5py.File(filename, "r") as h5:
                if (
                    "level2/binned_filtered_data" in h5
                    and "level2_noise_stats/binned_filtered_data/auto_rms" in h5
                ):
                    final_files.append(filename)
                else:
                    logging.warning("Excluding obsid %d: missing required datasets in %s", obsid, filename)
                    excluded_missing_datasets += 1
        except OSError:
            logging.warning("Excluding obsid %d: could not open file %s", obsid, filename)
            excluded_open_error += 1
            continue

    # Summary report
    n_total = len(query_source_group_list)
    n_used = len(final_files)
    n_excluded = n_total - n_used
    logging.info("Map-making file selection: %d total observations, %d used, %d excluded "
                 "(%d no Level-2 path, %d all flags bad, %d missing datasets, %d file open errors)",
                 n_total, n_used, n_excluded, n_no_level2, excluded_bad_flags,
                 excluded_missing_datasets, excluded_open_error)
    return final_files


class MapMaking:
    def __init__(self, config_file="", line_mode=False) -> None:
        logging.info("Initializing MapMaking (line_mode=%s)", line_mode)

        self.config_file = config_file
        self.line_mode = line_mode
        self.parameters = read_parameter_file(self.config_file) if os.path.exists(config_file) else {}

    def _load_module(self, module_info):
        package = module_info["package"]
        module_name = module_info["module"]
        args = module_info["args"]
        module_cls = getattr(importlib.import_module(package), module_name)
        return module_cls(**args), args

    def _resolve_files(self, module, min_obs_id, max_obs_id, obsid_list):
        # Per-module obsid list file takes highest priority
        module_obsid_file = getattr(module, "obsid_list_file", None)
        if module_obsid_file is not None:
            logging.info("Loading obsid list from file: %s", module_obsid_file)
            obsid_list = np.loadtxt(module_obsid_file, dtype=int, ndmin=1).tolist()

        if obsid_list is not None:
            if isinstance(obsid_list, str):
                obsid_list = np.loadtxt(obsid_list, dtype=int).tolist()
            fileinfo = db.query_obsid_list(obsid_list, return_dict=False)
            return [entry.level2_path for _, entry in fileinfo.items() if entry.level2_path is not None]

        file_list_func = get_line_file_list if self.line_mode else get_file_list

        if isinstance(module.source, list):
            final_files = []
            for src in module.source:
                final_files += file_list_func(
                    target_source_group=module.source_group,
                    target_source=src,
                    min_obs_id=min_obs_id,
                    max_obs_id=max_obs_id,
                    obsid_list=obsid_list,
                )
                print(src, final_files)
            return final_files

        return file_list_func(
            target_source_group=module.source_group,
            target_source=module.source,
            min_obs_id=min_obs_id,
            max_obs_id=max_obs_id,
            obsid_list=obsid_list,
        )

    def run(self) -> None:
        logging.info("Running MapMaking (line_mode=%s)", self.line_mode)

        min_obs_id = self.parameters["Master"].get("min_obsid", 7000)
        max_obs_id = self.parameters["Master"].get("max_obsid", 100000)
        obsid_list = self.parameters["Master"].get("obsid_list", None)

        for module_info in self.parameters["Master"]["_pipeline"]:
            module, args = self._load_module(module_info)
            min_obs_id = args.get("min_obsid", min_obs_id)
            max_obs_id = args.get("max_obsid", max_obs_id)

            final_files = self._resolve_files(module, min_obs_id, max_obs_id, obsid_list)
            logging.info(
                "Executing module %s (%s files)", module.__class__.__name__, len(final_files)
            )
            module.run(final_files)


def get_line_file_list(target_source_group=None, target_source=None, min_obs_id=7000, max_obs_id=100000, obsid_list=None):
    # Get the filelist from the database
    if obsid_list is not None:
        query_source_group_list = db.query_obsid_list(obsid_list, return_dict=False)
    else:
        query_source_group_list = db.query_source_group_list(
            target_source_group,
            source=target_source,
            min_obsid=min_obs_id,
            max_obsid=max_obs_id,
            return_dict=False,
        )

    n_total = len(query_source_group_list)
    n_no_level2 = sum(1 for _, f in query_source_group_list.items() if f.level2_path is None)
    source_file_list = [
        (f.obsid, f.level1_path, f.level2_path) for _, f in query_source_group_list.items() if f.level2_path is not None
    ]

    # Check that level2/binned_filtered_data in each file
    final_files = []
    excluded_missing_datasets = 0
    excluded_open_error = 0
    for (obsid, level1_file, level2_file) in source_file_list:
        try:
            with h5py.File(level2_file, "r") as h5:
                if (
                    "level2/binned_filtered_data" in h5
                    and "level2_noise_stats/binned_filtered_data/auto_rms" in h5
                    and "level2/vane/gain" in h5
                ):
                    final_files.append([level1_file, level2_file])
                else:
                    logging.warning("Excluding obsid %d: missing required datasets in %s", obsid, level2_file)
                    excluded_missing_datasets += 1
        except OSError:
            logging.warning("Excluding obsid %d: could not open file %s", obsid, level2_file)
            excluded_open_error += 1
            continue

    n_used = len(final_files)
    logging.info("Line map-making file selection: %d total observations, %d used, %d excluded "
                 "(%d no Level-2 path, %d missing datasets, %d file open errors)",
                 n_total, n_used, n_total - n_used, n_no_level2,
                 excluded_missing_datasets, excluded_open_error)
    return final_files


class LineMapMaking(MapMaking):
    """Backward-compatible alias: MapMaking with line_mode=True."""

    def __init__(self, config_file="") -> None:
        super().__init__(config_file=config_file, line_mode=True)
