#!/usr/bin/env python3
"""
Log Analysis Script for CNN Experiments

This script analyzes log files from centralized and decentralized training runs,
calculating the average time per batch and total training time for each run,
as well as aggregated statistics across all runs.

Usage:
    python analyze_logs.py EXPERIMENT

Arguments:
    EXPERIMENT  Name of the experiment directory (e.g., 'linux_b32')

Examples:
    python analyze_logs.py linux_b32

The script expects the following directory structure:
    EXPERIMENT/
    ├── centralized/
    │   ├── run1/
    │   │   └── log.txt
    │   ├── run2/
    │   │   └── log.txt
    │   └── run3/
    │       └── log.txt
    └── decentralized/
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


def analyze_single_run(experiment, run_num):
    """
    Analyze logs from centralized and decentralized runs of an experiment.

    Args:
        experiment: Name of the experiment directory
        run_num: Run number (1-3)

    Returns:
        Dictionary with timing statistics for both centralized and decentralized runs
    """
    base_dir = Path(__file__).parent

    # Define paths to log files
    centralized_log = base_dir / experiment / "centralized" / f"run{run_num}" / "log.txt"
    decentralized_log = base_dir / experiment / "decentralized" / f"run{run_num}" / "log.txt"

    result = {
        'run': run_num,
        'centralized': {'times': [], 'avg_time': None, 'std_dev': None},
        'decentralized': {'times': [], 'avg_time': None, 'std_dev': None}
    }

    # Check if files exist and parse them
    if centralized_log.exists():
        result['centralized']['times'] = parse_log_file(centralized_log)
        if result['centralized']['times']:
            result['centralized']['avg_time'] = statistics.mean(result['centralized']['times'])
            if len(result['centralized']['times']) > 1:
                result['centralized']['std_dev'] = statistics.stdev(result['centralized']['times'])
            else:
                result['centralized']['std_dev'] = 0.0

    if decentralized_log.exists():
        result['decentralized']['times'] = parse_log_file(decentralized_log)
        if result['decentralized']['times']:
            result['decentralized']['avg_time'] = statistics.mean(result['decentralized']['times'])
            if len(result['decentralized']['times']) > 1:
                result['decentralized']['std_dev'] = statistics.stdev(result['decentralized']['times'])
            else:
                result['decentralized']['std_dev'] = 0.0

    return result


def calculate_speedup_stats(cent_total_time, decen_total_time):
    """
    Calculate speedup statistics between centralized and decentralized runs.
    
    Args:
        cent_total_time: Total time for centralized run
        decen_total_time: Total time for decentralized run
        
    Returns:
        Dictionary with speedup statistics
    """
    if cent_total_time is None or decen_total_time is None:
        return {'percent_speedup': None, 'time_saved': None}
    
    time_saved = cent_total_time - decen_total_time
    percent_speedup = (time_saved / cent_total_time) * 100 if cent_total_time != 0 else 0
    
    return {
        'percent_speedup': percent_speedup,
        'time_saved': time_saved
    }


def print_run_analysis(run_data):
    """
    Print the analysis results for a single run.

    Args:
        run_data: Dictionary containing timing data for a single run
    """
    run_num = run_data['run']
    
    print(f"\n--- Run {run_num} ---")
    
    # Centralized stats
    cent = run_data['centralized']
    if cent['avg_time'] is not None:
        std_dev_pct = (cent['std_dev'] / cent['avg_time']) * 100 if cent['avg_time'] != 0 else 0
        print(f"Centralized:")
        print(f"  Average time per batch: {cent['avg_time']:.6f}s ± {cent['std_dev']:.6f}s ({std_dev_pct:.2f}%)")
        print(f"  Total training time: {sum(cent['times']):.6f}s")
    else:
        print("Centralized: Log file not found or could not be parsed")
    
    # Decentralized stats
    decen = run_data['decentralized']
    if decen['avg_time'] is not None:
        std_dev_pct = (decen['std_dev'] / decen['avg_time']) * 100 if decen['avg_time'] != 0 else 0
        print(f"Decentralized:")
        print(f"  Average time per batch: {decen['avg_time']:.6f}s ± {decen['std_dev']:.6f}s ({std_dev_pct:.2f}%)")
        print(f"  Total training time: {sum(decen['times']):.6f}s")
        
        # Calculate and show speedup if both runs exist
        if cent['avg_time'] is not None:
            speedup_stats = calculate_speedup_stats(sum(cent['times']), sum(decen['times']))
            if speedup_stats['percent_speedup'] is not None:
                print(f"  Percent speedup: {speedup_stats['percent_speedup']:.2f}% "
                      f"(time saved: {speedup_stats['time_saved']:.6f}s)")
    else:
        print("Decentralized: Log file not found or could not be parsed")


def print_aggregate_analysis(all_runs_data):
    """
    Print the aggregated analysis results across all runs.

    Args:
        all_runs_data: List of dictionaries containing timing data for all runs
    """
    print(f"\n{'='*20} AGGREGATE STATISTICS ACROSS ALL RUNS {'='*20}")
    
    # Collect all centralized and decentralized times
    all_cent_avg_times = []
    all_cent_std_devs = []
    all_cent_total_times = []
    
    all_decen_avg_times = []
    all_decen_std_devs = []
    all_decen_total_times = []
    
    for run_data in all_runs_data:
        if run_data['centralized']['avg_time'] is not None:
            all_cent_avg_times.append(run_data['centralized']['avg_time'])
            all_cent_std_devs.append(run_data['centralized']['std_dev'])
            all_cent_total_times.append(sum(run_data['centralized']['times']))
            
        if run_data['decentralized']['avg_time'] is not None:
            all_decen_avg_times.append(run_data['decentralized']['avg_time'])
            all_decen_std_devs.append(run_data['decentralized']['std_dev'])
            all_decen_total_times.append(sum(run_data['decentralized']['times']))
    
    # Print centralized aggregate stats
    if all_cent_avg_times:
        avg_cent_avg_time = statistics.mean(all_cent_avg_times)
        std_cent_avg_time = statistics.stdev(all_cent_avg_times) if len(all_cent_avg_times) > 1 else 0.0
        std_cent_avg_time_pct = (std_cent_avg_time / avg_cent_avg_time) * 100 if avg_cent_avg_time != 0 else 0
        
        avg_cent_total_time = statistics.mean(all_cent_total_times)
        std_cent_total_time = statistics.stdev(all_cent_total_times) if len(all_cent_total_times) > 1 else 0.0
        
        print(f"\nCentralized (averaged across runs):")
        print(f"  Average time per batch: {avg_cent_avg_time:.6f}s ± {std_cent_avg_time:.6f}s ({std_cent_avg_time_pct:.2f}%)")
        print(f"  Total training time: {avg_cent_total_time:.6f}s ± {std_cent_total_time:.6f}s")
    else:
        print("\nCentralized: No valid runs found")
    
    # Print decentralized aggregate stats
    if all_decen_avg_times:
        avg_decen_avg_time = statistics.mean(all_decen_avg_times)
        std_decen_avg_time = statistics.stdev(all_decen_avg_times) if len(all_decen_avg_times) > 1 else 0.0
        std_decen_avg_time_pct = (std_decen_avg_time / avg_decen_avg_time) * 100 if avg_decen_avg_time != 0 else 0
        
        avg_decen_total_time = statistics.mean(all_decen_total_times)
        std_decen_total_time = statistics.stdev(all_decen_total_times) if len(all_decen_total_times) > 1 else 0.0
        
        print(f"\nDecentralized (averaged across runs):")
        print(f"  Average time per batch: {avg_decen_avg_time:.6f}s ± {std_decen_avg_time:.6f}s ({std_decen_avg_time_pct:.2f}%)")
        print(f"  Total training time: {avg_decen_total_time:.6f}s ± {std_decen_total_time:.6f}s")
        
        # Calculate aggregate speedup
        if all_cent_total_times:
            avg_cent_total = statistics.mean(all_cent_total_times)
            avg_decen_total = statistics.mean(all_decen_total_times)
            speedup_stats = calculate_speedup_stats(avg_cent_total, avg_decen_total)
            if speedup_stats['percent_speedup'] is not None:
                print(f"  Percent speedup: {speedup_stats['percent_speedup']:.2f}% "
                      f"(time saved: {speedup_stats['time_saved']:.6f}s)")
    else:
        print("\nDecentralized: No valid runs found")


def analyze_experiment(experiment):
    """
    Analyze logs from centralized and decentralized runs of an experiment.
    Aggregates across all runs (1-3).

    Args:
        experiment: Name of the experiment directory
    """
    print(f"Analyzing experiment '{experiment}'...")
    
    all_runs_data = []
    
    # Analyze each run individually first
    for run in range(1, 4):
        print(f"\nProcessing run {run}...")
        run_data = analyze_single_run(experiment, run)
        all_runs_data.append(run_data)
        print_run_analysis(run_data)
    
    # Print aggregate statistics
    print_aggregate_analysis(all_runs_data)


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment logs for timing statistics")
    parser.add_argument("experiment", help="Name of the experiment directory")

    args = parser.parse_args()

    # Validate experiment directory exists
    exp_path = Path(__file__).parent / args.experiment
    if not exp_path.exists():
        print(f"Error: Experiment directory does not exist: {exp_path}")
        return

    analyze_experiment(args.experiment)


if __name__ == "__main__":
    main()

