#!/usr/bin/env python3
"""
Aggregate Statistics Script for CNN Experiments

This script analyzes log files from centralized and decentralized training runs,
and outputs only the aggregate statistics for a given experiment run.

Usage:
    python aggregate_stats.py EXPERIMENT [SPECIFIER]

Arguments:
    EXPERIMENT  Name of the experiment directory (e.g., 'mac_b64')
    SPECIFIER   Name of the decentralized folder to compare against (default: 'decentralized')

Examples:
    python aggregate_stats.py mac_b64
    python aggregate_stats.py mac_b64 decentralized_alt

The script expects the following directory structure:
    EXPERIMENT/
    ├── centralized/
    │   ├── run1/
    │   │   └── log.txt
    │   ├── run2/
    │   │   └── log.txt
    │   └── run3/
    │       └── log.txt
    └── [SPECIFIER]/
        ├── run1/
        │   └── log.txt
        ├── run2/
        │   └── log.txt
        └── run3/
            └── log.txt
"""

import argparse
import os
import re
import statistics
from pathlib import Path


def parse_log_file(filepath):
    """
    Parse a log file and extract timing information.

    Args:
        filepath: Path to the log file

    Returns:
        List of time values for each batch
    """
    times = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            # Match centralized format: epoch 1 | step 0/62 | loss = 1.4084 | 1 / 62 | 3.3706s
            centralized_match = re.search(r'loss = ([\d.]+).*?([\d.]+)s$', line)
            if centralized_match:
                time_val = float(centralized_match.group(2))
                times.append(time_val)
                continue

            # Match decentralized format: epoch 1 | step 0/62 | loss = 1.4261 | global_step = 1/62 | 1.782144546508789
            decentralized_match = re.search(r'loss = ([\d.]+).*?([\d.]+)$', line)
            if decentralized_match:
                # Check that the last part is actually a time value (not step count)
                time_candidate = float(decentralized_match.group(2))

                # Heuristic: if the value is reasonably small (less than 100), treat as time
                # Otherwise, it might be a step counter or other metric
                if time_candidate < 100:  # Adjust threshold as needed based on your data
                    times.append(time_candidate)

    return times


def analyze_single_run(experiment, specifier, run_num):
    """
    Analyze logs from centralized and decentralized runs of an experiment.

    Args:
        experiment: Name of the experiment directory
        specifier: Name of the decentralized folder to compare against
        run_num: Run number (1-3)

    Returns:
        Dictionary with timing statistics for both centralized and decentralized runs
    """
    base_dir = Path(__file__).parent

    # Define paths to log files
    centralized_log = base_dir / experiment / "centralized" / f"run{run_num}" / "log.txt"
    decentralized_log = base_dir / experiment / specifier / f"run{run_num}" / "log.txt"

    result = {
        'run': run_num,
        'centralized': {'times': [], 'total_time': None, 'avg_time_per_batch': None},
        'decentralized': {'times': [], 'total_time': None, 'avg_time_per_batch': None}
    }

    # Check if files exist and parse them
    if centralized_log.exists():
        result['centralized']['times'] = parse_log_file(centralized_log)
        if result['centralized']['times']:
            result['centralized']['total_time'] = sum(result['centralized']['times'])
            result['centralized']['avg_time_per_batch'] = statistics.mean(result['centralized']['times'])

    if decentralized_log.exists():
        result['decentralized']['times'] = parse_log_file(decentralized_log)
        if result['decentralized']['times']:
            result['decentralized']['total_time'] = sum(result['decentralized']['times'])
            result['decentralized']['avg_time_per_batch'] = statistics.mean(result['decentralized']['times'])

    return result


def calculate_aggregate_stats(all_runs_data):
    """
    Calculate aggregate statistics across all runs.

    Args:
        all_runs_data: List of dictionaries containing timing data for all runs

    Returns:
        Dictionary with aggregate statistics
    """
    # Collect all centralized and decentralized times
    all_cent_total_times = []
    all_cent_avg_times_per_batch = []
    
    all_decen_total_times = []
    all_decen_avg_times_per_batch = []

    for run_data in all_runs_data:
        if run_data['centralized']['total_time'] is not None:
            all_cent_total_times.append(run_data['centralized']['total_time'])
            all_cent_avg_times_per_batch.append(run_data['centralized']['avg_time_per_batch'])

        if run_data['decentralized']['total_time'] is not None:
            all_decen_total_times.append(run_data['decentralized']['total_time'])
            all_decen_avg_times_per_batch.append(run_data['decentralized']['avg_time_per_batch'])

    # Calculate aggregate statistics
    result = {
        'centralized': {},
        'decentralized': {},
        'speedup': {}
    }

    # Centralized stats
    if all_cent_total_times:
        result['centralized']['avg_total_time'] = statistics.mean(all_cent_total_times)
        result['centralized']['std_dev_total_time'] = statistics.stdev(all_cent_total_times) if len(all_cent_total_times) > 1 else 0.0
        
        result['centralized']['avg_time_per_batch'] = statistics.mean(all_cent_avg_times_per_batch)
        result['centralized']['std_dev_time_per_batch'] = statistics.stdev(all_cent_avg_times_per_batch) if len(all_cent_avg_times_per_batch) > 1 else 0.0
    else:
        result['centralized'] = None

    # Decentralized stats
    if all_decen_total_times:
        result['decentralized']['avg_total_time'] = statistics.mean(all_decen_total_times)
        result['decentralized']['std_dev_total_time'] = statistics.stdev(all_decen_total_times) if len(all_decen_total_times) > 1 else 0.0
        
        result['decentralized']['avg_time_per_batch'] = statistics.mean(all_decen_avg_times_per_batch)
        result['decentralized']['std_dev_time_per_batch'] = statistics.stdev(all_decen_avg_times_per_batch) if len(all_decen_avg_times_per_batch) > 1 else 0.0
    else:
        result['decentralized'] = None

    # Speedup stats
    if result['centralized'] and result['decentralized']:
        avg_cent_total = result['centralized']['avg_total_time']
        avg_decen_total = result['decentralized']['avg_total_time']
        
        time_saved = avg_cent_total - avg_decen_total
        percent_speedup = (time_saved / avg_cent_total) * 100 if avg_cent_total != 0 else 0
        
        result['speedup']['avg_percent_speedup'] = percent_speedup
        result['speedup']['time_saved'] = time_saved
        result['speedup']['std_dev_time_saved'] = ((result['centralized']['std_dev_total_time'] ** 2 + 
                                                   result['decentralized']['std_dev_total_time'] ** 2) ** 0.5) if result['centralized']['std_dev_total_time'] and result['decentralized']['std_dev_total_time'] else 0.0

    return result


def print_aggregate_stats(experiment, specifier, stats):
    """
    Print the aggregate statistics in the required format.

    Args:
        experiment: Name of the experiment
        specifier: Name of the decentralized folder
        stats: Dictionary with aggregate statistics
    """
    print(f"{experiment}/{specifier}")

    # Centralized total time
    if stats['centralized']:
        avg_total = stats['centralized']['avg_total_time']
        std_dev_total = stats['centralized']['std_dev_total_time']
        std_dev_pct_total = (std_dev_total / avg_total) * 100 if avg_total != 0 else 0
        print(f"C_total_time {avg_total:.3f}s ± {std_dev_total:.3f}s ({std_dev_pct_total:.2f}%)")

        # Centralized time per batch
        avg_batch = stats['centralized']['avg_time_per_batch']
        std_dev_batch = stats['centralized']['std_dev_time_per_batch']
        std_dev_pct_batch = (std_dev_batch / avg_batch) * 100 if avg_batch != 0 else 0
        print(f"C_time_per_batch {avg_batch:.3f}s ± {std_dev_batch:.3f}s ({std_dev_pct_batch:.2f}%)")
    else:
        print("C_total_time N/A ± N/A (N/A%)")
        print("C_time_per_batch N/A ± N/A (N/A%)")

    # Decentralized total time
    if stats['decentralized']:
        avg_total = stats['decentralized']['avg_total_time']
        std_dev_total = stats['decentralized']['std_dev_total_time']
        std_dev_pct_total = (std_dev_total / avg_total) * 100 if avg_total != 0 else 0
        print(f"D_total_time {avg_total:.3f}s ± {std_dev_total:.3f}s ({std_dev_pct_total:.2f}%)")

        # Decentralized time per batch
        avg_batch = stats['decentralized']['avg_time_per_batch']
        std_dev_batch = stats['decentralized']['std_dev_time_per_batch']
        std_dev_pct_batch = (std_dev_batch / avg_batch) * 100 if avg_batch != 0 else 0
        print(f"D_time_per_batch {avg_batch:.3f}s ± {std_dev_batch:.3f}s ({std_dev_pct_batch:.2f}%)")
    else:
        print("D_total_time N/A ± N/A (N/A%)")
        print("D_time_per_batch N/A ± N/A (N/A%)")

    # Speedup stats
    if stats['speedup']:
        avg_speedup = stats['speedup']['avg_percent_speedup']
        time_saved = stats['speedup']['time_saved']
        std_dev_time_saved = stats['speedup']['std_dev_time_saved']
        std_dev_pct_saved = (std_dev_time_saved / abs(time_saved)) * 100 if time_saved != 0 else 0
        print(f"% Total Time Decrease: {avg_speedup:.3f}% ({time_saved:.3f}s ± {std_dev_time_saved:.3f}s ({std_dev_pct_saved:.2f}%))")
    else:
        print("% Total Time Decrease: N/A% (N/A ± N/A (N/A%))")


def analyze_experiment(experiment, specifier):
    """
    Analyze logs from centralized and decentralized runs of an experiment.
    Aggregates across all runs (1-3).

    Args:
        experiment: Name of the experiment directory
        specifier: Name of the decentralized folder to compare against
    """
    all_runs_data = []

    # Analyze each run
    for run in range(1, 4):
        run_data = analyze_single_run(experiment, specifier, run)
        all_runs_data.append(run_data)

    # Calculate and print aggregate statistics
    stats = calculate_aggregate_stats(all_runs_data)
    print_aggregate_stats(experiment, specifier, stats)


def main():
    parser = argparse.ArgumentParser(description="Output aggregate statistics for experiment logs")
    parser.add_argument("experiment", help="Name of the experiment directory")
    parser.add_argument("specifier", nargs="?", default="decentralized", 
                        help="Name of the decentralized folder to compare against (default: 'decentralized')")

    args = parser.parse_args()

    # Validate experiment directory exists
    exp_path = Path(__file__).parent / args.experiment
    if not exp_path.exists():
        print(f"Error: Experiment directory does not exist: {exp_path}")
        return

    # Validate decentralized directory exists
    decen_path = exp_path / args.specifier
    if not decen_path.exists():
        print(f"Error: Decentralized directory does not exist: {decen_path}")
        return

    analyze_experiment(args.experiment, args.specifier)


if __name__ == "__main__":
    main()