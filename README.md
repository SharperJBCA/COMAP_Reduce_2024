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

### 4) Reconcile Level-2 files against SQL after disk moves / missing files

```bash
python modules/scripts/reconcile_level2_inventory.py \
  --database databases/COMAP_manchester.db \
  --search-root /mnt/disk1/COMAP/level2 /mnt/disk2/COMAP/level2 \
  --output-dir reconcile_reports
```

Default mode is dry-run. Add `--write` to:

- clear `level2_path` rows when a Level-2 file cannot be found,
- remap stale `level2_path` values to discovered on-disk paths.

The script writes:

- `missing_level2_rows.tsv` (obsids with DB entry but no file found)
- `missing_level2_obsids.txt` (ready for Level-2 pipeline obsid list input)
- `missing_level2_level1_paths.txt` (Level-1 paths for reprocessing workflows)
- `remapped_level2_rows.tsv`
- `level2_files_not_in_db.tsv`
- `duplicate_level2_files.tsv`

### 5) Quick SQL visibility tooling

Use `db_overview.py` for quick operational summaries:

```bash
python modules/scripts/db_overview.py --database databases/COMAP_manchester.db summary
python modules/scripts/db_overview.py --database databases/COMAP_manchester.db missing-level2 --limit 500
python modules/scripts/db_overview.py --database databases/COMAP_manchester.db path-search /new/disk/path
```

### 6) Analyze chunked map integration depth

Run destriped map-making on ordered file chunks and track cumulative RMS-vs-hits:

```bash
python modules/scripts/analyze_chunked_integration.py \
  --database databases/COMAP_manchester.db \
  --source fg6 \
  --min-obsid 12000 \
  --max-obsid 18000 \
  --chunk-size 25 \
  --band 0 \
  --feeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
  --wcs-def Galactic \
  --offset-length 100 \
  --sigma-red-cutoff 0.4 \
  --center-lon 120.0 \
  --center-lat 0.0 \
  --center-radius-pix 10 \
  --annulus-rin-pix 20 \
  --annulus-rout-pix 35 \
  --output-dir outputs/chunked_depth \
  --plot-name rms_vs_hits.png \
  --csv-name rms_vs_hits.csv
```

Outputs include a per-stage CSV (`n_chunks`, `n_files`, center RMS stats, annulus mean hits) and a two-panel (linear + log-log) PNG plot with an `A/sqrt(hits)` reference curve.

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

## I/O Benchmarking

Before tuning `nprocess` for Level-2 processing, measure how your storage scales with concurrent HDF5 readers using the benchmark script:

```bash
# Using the pipeline database to find files automatically
python modules/scripts/benchmark_hdf5_io.py \
    --database databases/COMAP_manchester.db \
    --concurrency 1,2,4,8,16 \
    --num-files 20 \
    --level 1 \
    --output benchmark_level1.png

# Or point at a directory of HDF5 files directly
python modules/scripts/benchmark_hdf5_io.py \
    --directory /scratch/nas_comap1/level1/ \
    --concurrency 1,2,4,8 \
    --num-files 30 \
    --output benchmark_nas1.png

# Quick latency test (reads a single slice per file instead of the full dataset)
python modules/scripts/benchmark_hdf5_io.py \
    --directory /path/to/files \
    --concurrency 1,2,4,8 \
    --num-files 10 \
    --slice-only \
    --output benchmark_latency.png
```

The script produces a two-panel plot:
- **Left panel**: total read throughput (MB/s) vs number of concurrent readers
- **Right panel**: per-file read latency distribution (box plot) at each concurrency level

Use `--repeats N` for more stable measurements. The warm-up round (skippable with `--skip-warmup`) pre-populates NFS caches so subsequent measurements reflect steady-state performance.

If throughput plateaus or degrades beyond a certain concurrency level, set `nprocess` in your processing TOML to that value or below.

---

## Notes and operational guidance

- Use SQL as the source of truth for observation indexing and file discovery.
- Treat Level-2 HDF5 as the portable science product; the SQL snapshot module helps portability/reproducibility.
- If you run multiprocessing Level-2 jobs on shared storage, run the I/O benchmark first to find the optimal concurrency level before tuning compute parallelism.
