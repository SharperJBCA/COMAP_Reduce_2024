# COMAP Reduce 2024

Pipeline for COMAP data acquisition, Level-2 processing, and map making.

## Quick start

```bash
python run.py parameter_files/main_run.toml
```

The top-level pipeline is configured by TOML files in `parameter_files/` and module-specific config in `modules/MapMaking/map_making_configs/`.

---

## Repository layout

- `run.py`: main entry point.
- `parameter_files/`: pipeline configuration.
- `modules/DataAcquisition/`: discovery + ingest of Level-1 files into SQL.
- `modules/ProcessLevel2/`: Level-2 production pipeline.
- `modules/MapMaking/`: map making and line map products.
- `modules/SQLModule/`: database schema + API.
- `modules/scripts/`: helper scripts (querying, maintenance tasks).

---

## Environment setup

1. Create/activate a Python environment with required dependencies (`numpy`, `h5py`, `sqlalchemy`, `toml`, `pytest`, etc.).
2. Build utility extensions (if used in your workflow):

```bash
cd modules/utils
make
python setup_bin_funcs.py build_ext --inplace
python setup_median_filter.py build_ext --inplace
python setup_mean_filter.py build_ext --inplace
```

---

## How the pipeline is structured

The master runner (`run.py`) calls `run_pipeline(...)`, which executes the ordered module list defined in the `[Master] pipeline = [...]` section of your TOML.

Typical flow:

1. Data acquisition/update SQL with Level-1 metadata
2. Level-2 creation + processing
3. Map making

The Level-2 pipeline itself is modular and runs each configured module sequentially for each observation.

---

## Running common tasks

### 1) Produce / refresh Level-2 files

Configure `parameter_files/level2_processing.toml`, then run via `parameter_files/main_run.toml` (or directly through the configured module chain).

### 2) Build maps

Use the map-making configs:

- `parameter_files/level2_mapmaking.toml`
- `parameter_files/level2_line_mapmaking.toml`

### 3) Query targets

```bash
python modules/scripts/query_target.py --help
```

---

## Additional operations

## A) Update SQL `level2_path` values after a disk move

This is supported with the current framework.

A new maintenance script is provided:

```bash
python modules/scripts/remap_level2_paths.py \
  --database databases/COMAP_manchester.db \
  --old-prefix /old/disk/COMAP/level2 \
  --new-prefix /new/disk/COMAP/level2
```

By default this is a **dry-run**. Add `--write` to apply updates.

Optional safety check:

```bash
python modules/scripts/remap_level2_paths.py ... --write --require-new-path-exists
```

This only updates rows when the remapped file exists.

## B) Keep Level-2 files self-contained with SQL metadata snapshots

A new optional post-processing module is provided:

- `modules/ProcessLevel2/FinalizeLevel2Metadata/FinalizeLevel2Metadata.py`

It writes a SQL snapshot into each Level-2 file under:

- `level2/sql_snapshot`

Included metadata:

- scalar observation fields from `comap_data`
- per `(pixel, frequency_band)` quality flag status
- associated quality statistics and comments

To enable, place it at the **end** of your Level-2 module sequence in the processing TOML.

---

## Extending the pipeline

To add a new module:

1. Create a module class with a `run(self, file_info)` entry point.
2. Place it under an appropriate package in `modules/`.
3. Expose the class via package `__init__.py` where needed.
4. Add it to the TOML pipeline list with constructor kwargs.

For Level-2 steps, prefer inheriting from `BaseCOMAPModule` and using `RetryH5PY` for robust HDF5 access.

---

## Notes and operational guidance

- Use SQL as the source of truth for observation indexing and file discovery.
- Treat Level-2 HDF5 as the portable science product; the SQL snapshot module helps portability/reproducibility.
- If you run multiprocessing Level-2 jobs on shared storage, monitor I/O pressure first before deep compute optimization.
