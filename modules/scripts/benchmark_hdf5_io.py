#!/usr/bin/env python
"""Benchmark HDF5 read throughput at varying concurrency levels.

Measures how NFS/disk I/O scales with the number of concurrent readers,
helping determine the optimal nprocess setting for the pipeline.

Usage examples:

    # Using the pipeline database to find files
    python modules/scripts/benchmark_hdf5_io.py \
        --database databases/COMAP_manchester.db \
        --concurrency 1,2,4,8 --num-files 20 --level 1

    # Using a directory of HDF5 files
    python modules/scripts/benchmark_hdf5_io.py \
        --directory /scratch/nas_comap1/level1/ \
        --concurrency 1,2,4,8,16 --num-files 30

    # Quick test with small slice reads
    python modules/scripts/benchmark_hdf5_io.py \
        --directory /path/to/files --num-files 5 --concurrency 1,2,4 \
        --slice-only --output quick_bench.png
"""

import argparse
import glob
import multiprocessing
import os
import random
import sys
import time

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Default datasets per level
DEFAULT_DATASETS = {
    1: "spectrometer/tod",
    2: "level2/binned_filtered_data",
}


def discover_files_from_database(db_path, level, num_files):
    """Query the pipeline database for existing file paths."""
    sys.path.insert(0, os.getcwd())
    from modules.SQLModule.SQLModule import db as pipeline_db

    pipeline_db.connect(db_path)

    from modules.SQLModule.SQLModule import COMAPData
    import sqlalchemy as sa

    col = COMAPData.level1_path if level == 1 else COMAPData.level2_path
    with pipeline_db.Session() as session:
        rows = (
            session.query(col)
            .filter(col.isnot(None))
            .all()
        )
    pipeline_db.disconnect()

    paths = [r[0] for r in rows if r[0] and os.path.exists(r[0])]
    if not paths:
        raise RuntimeError(f"No existing level-{level} files found in database")

    random.shuffle(paths)
    return paths[:num_files]


def discover_files_from_directory(directory, num_files):
    """Find HDF5 files in a directory."""
    patterns = ["*.hdf5", "*.h5", "*.hdf"]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(directory, pat)))
        paths.extend(glob.glob(os.path.join(directory, "**", pat), recursive=True))

    paths = [p for p in set(paths) if os.path.isfile(p)]
    if not paths:
        raise RuntimeError(f"No HDF5 files found in {directory}")

    random.shuffle(paths)
    return paths[:num_files]


def _read_file(args):
    """Worker function: open one HDF5 file, read a dataset, return timing info."""
    filepath, dataset_path, slice_only = args
    t0 = time.monotonic()
    nbytes = 0
    try:
        with h5py.File(filepath, "r") as f:
            if dataset_path and dataset_path in f:
                dset = f[dataset_path]
                if slice_only:
                    # Read only first element along first axis
                    sl = tuple([slice(0, 1)] + [slice(None)] * (len(dset.shape) - 1))
                    data = dset[sl]
                else:
                    data = dset[:]
                nbytes = data.nbytes
            else:
                # Fallback: find first dataset and read it
                def _find_first_dataset(group):
                    for key in group:
                        item = group[key]
                        if isinstance(item, h5py.Dataset):
                            return item
                        elif isinstance(item, h5py.Group):
                            result = _find_first_dataset(item)
                            if result is not None:
                                return result
                    return None

                dset = _find_first_dataset(f)
                if dset is not None:
                    if slice_only:
                        sl = tuple(
                            [slice(0, 1)] + [slice(None)] * (len(dset.shape) - 1)
                        )
                        data = dset[sl]
                    else:
                        data = dset[:]
                    nbytes = data.nbytes

    except Exception as e:
        elapsed = time.monotonic() - t0
        return filepath, 0, elapsed, str(e)

    elapsed = time.monotonic() - t0
    return filepath, nbytes, elapsed, None


def run_benchmark(files, concurrency_levels, dataset_path, slice_only, repeats=1):
    """Run the benchmark sweep across concurrency levels.

    Returns dict mapping concurrency -> {
        'wall_time': float,
        'per_file_times': list[float],
        'total_bytes': int,
        'throughput_mbps': float,
        'errors': list[str],
    }
    """
    results = {}
    work_items = [(f, dataset_path, slice_only) for f in files]

    for n_workers in concurrency_levels:
        all_per_file_times = []
        all_bytes = 0
        all_errors = []
        total_wall = 0

        for _ in range(repeats):
            t_wall_start = time.monotonic()

            if n_workers == 1:
                # Sequential — no pool overhead
                outcomes = [_read_file(item) for item in work_items]
            else:
                with multiprocessing.Pool(n_workers) as pool:
                    outcomes = pool.map(_read_file, work_items)

            t_wall = time.monotonic() - t_wall_start
            total_wall += t_wall

            for _, nbytes, elapsed, err in outcomes:
                all_per_file_times.append(elapsed)
                all_bytes += nbytes
                if err:
                    all_errors.append(err)

        avg_wall = total_wall / repeats
        throughput = (all_bytes / repeats) / (1024 * 1024) / avg_wall if avg_wall > 0 else 0

        results[n_workers] = {
            "wall_time": avg_wall,
            "per_file_times": all_per_file_times,
            "total_bytes": all_bytes // repeats,
            "throughput_mbps": throughput,
            "errors": all_errors,
        }

        n_err = len(all_errors)
        err_str = f" ({n_err} errors)" if n_err else ""
        print(
            f"  workers={n_workers:2d}  wall={avg_wall:6.1f}s  "
            f"throughput={throughput:7.1f} MB/s  "
            f"median_per_file={np.median(all_per_file_times):.2f}s{err_str}"
        )

    return results


def plot_results(results, output_path, num_files):
    """Create a two-panel diagnostic plot."""
    concurrency = sorted(results.keys())
    throughputs = [results[n]["throughput_mbps"] for n in concurrency]
    per_file_data = [results[n]["per_file_times"] for n in concurrency]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: throughput vs concurrency
    ax1.plot(concurrency, throughputs, "o-", color="steelblue", linewidth=2, markersize=8)
    ax1.set_xlabel("Concurrent readers")
    ax1.set_ylabel("Throughput (MB/s)")
    ax1.set_title(f"Total Read Throughput ({num_files} files)")
    ax1.set_xticks(concurrency)
    ax1.grid(True, alpha=0.3)

    # Right: per-file time distribution
    ax2.boxplot(
        per_file_data,
        labels=[str(n) for n in concurrency],
        showfliers=True,
        flierprops=dict(marker=".", markersize=3, alpha=0.5),
    )
    ax2.set_xlabel("Concurrent readers")
    ax2.set_ylabel("Time per file (s)")
    ax2.set_title("Per-File Read Latency")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")


def print_summary(results):
    """Print a text summary table."""
    print("\n" + "=" * 70)
    print(f"{'Workers':>8} {'Wall (s)':>10} {'MB/s':>10} {'Med (s)':>10} "
          f"{'P95 (s)':>10} {'Errors':>8}")
    print("-" * 70)
    for n in sorted(results.keys()):
        r = results[n]
        times = r["per_file_times"]
        print(
            f"{n:>8d} {r['wall_time']:>10.1f} {r['throughput_mbps']:>10.1f} "
            f"{np.median(times):>10.2f} {np.percentile(times, 95):>10.2f} "
            f"{len(r['errors']):>8d}"
        )
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark HDF5 read throughput at varying concurrency levels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--database", type=str, help="Pipeline SQLite database path")
    source.add_argument("--directory", type=str, help="Directory containing HDF5 files")

    parser.add_argument(
        "--concurrency", type=str, default="1,2,4,8",
        help="Comma-separated concurrency levels (default: 1,2,4,8)",
    )
    parser.add_argument(
        "--num-files", type=int, default=20,
        help="Number of files to benchmark (default: 20)",
    )
    parser.add_argument(
        "--level", type=int, choices=[1, 2], default=1,
        help="File level: 1=level1, 2=level2 (default: 1)",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="HDF5 dataset path to read (default: auto based on --level)",
    )
    parser.add_argument(
        "--output", type=str, default="benchmark_hdf5_io.png",
        help="Output plot filename (default: benchmark_hdf5_io.png)",
    )
    parser.add_argument(
        "--slice-only", action="store_true",
        help="Read only a single slice per dataset (faster, tests latency not throughput)",
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Number of times to repeat each concurrency level (default: 1)",
    )
    parser.add_argument(
        "--skip-warmup", action="store_true",
        help="Skip the cache warm-up round",
    )

    args = parser.parse_args()

    concurrency_levels = [int(x) for x in args.concurrency.split(",")]
    dataset_path = args.dataset or DEFAULT_DATASETS.get(args.level)

    # Discover files
    print(f"Discovering level-{args.level} files...")
    if args.database:
        files = discover_files_from_database(args.database, args.level, args.num_files)
    else:
        files = discover_files_from_directory(args.directory, args.num_files)

    print(f"Found {len(files)} files for benchmarking")
    if len(files) < args.num_files:
        print(f"  (requested {args.num_files}, but only {len(files)} available)")

    print(f"Dataset: {dataset_path}")
    print(f"Concurrency levels: {concurrency_levels}")
    print(f"Slice-only: {args.slice_only}")
    print()

    # Warm-up round
    if not args.skip_warmup:
        print("Warm-up round (populating caches)...")
        run_benchmark(files, [1], dataset_path, args.slice_only)
        print()

    # Main benchmark
    print("Running benchmark...")
    results = run_benchmark(
        files, concurrency_levels, dataset_path, args.slice_only, repeats=args.repeats
    )

    print_summary(results)
    plot_results(results, args.output, len(files))


if __name__ == "__main__":
    main()
