#!/usr/bin/env python3
"""
Decentralized Configurations Comparison Script for CNN Experiments

This script analyzes and compares log files from all decentralized configurations 
within a single experiment (mac_b64 or mac_b128), calculating the average time per 
batch, total training time, standard deviations, and other statistics for each 
configuration.

Usage:
    python compare_decentralized.py EXPERIMENT

Arguments:
    EXPERIMENT  Name of the experiment directory (e.g., 'mac_b64' or 'mac_b128')

Examples:
    python compare_decentralized.py mac_b64
    python compare_decentralized.py mac_b128

The script expects the following directory structure:
    EXPERIMENT/
    ├── centralized/
    │   ├── run1/
    │   │   └── log.txt
    │   ├── run2/
    │   │   └── log.txt
    │   └── run3/
    │       └── log.txt
    └── decentralized_*/ (various memory configs)
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
        Tuple of (List of time values for each batch, selected pipe split config)
    """
    times = []
    selected_config = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            # Look for selected pipe split configuration
            if line.lower().startswith("selected:"):
                selected_config = line.split(":", 1)[1].strip()
                continue

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

    return times, selected_config


def analyze_single_run(experiment, run_num, config_type):
    """
    Analyze logs from a specific run of an experiment.

    Args:
        experiment: Name of the experiment directory
        run_num: Run number (1-3)
        config_type: Type of configuration ('centralized' or 'decentralized_*')

    Returns:
        Dictionary with timing statistics for the run
    """
    base_dir = Path(__file__).parent

    # Define path to log file
    log_path = base_dir / experiment / config_type / f"run{run_num}" / "log.txt"

    result = {
        'run': run_num,
        'config_type': config_type,
        'times': [],
        'avg_time': None,
        'std_dev': None,
        'selected_config': None
    }

    # Check if file exists and parse it
    if log_path.exists():
        result['times'], result['selected_config'] = parse_log_file(log_path)
        if result['times']:
            result['avg_time'] = statistics.mean(result['times'])
            if len(result['times']) > 1:
                result['std_dev'] = statistics.stdev(result['times'])
            else:
                result['std_dev'] = 0.0

    return result


def get_decentralized_configs(experiment):
    """
    Get all decentralized configuration types for an experiment.

    Args:
        experiment: Name of the experiment directory

    Returns:
        List of decentralized configuration types
    """
    base_dir = Path(__file__).parent
    exp_path = base_dir / experiment
    
    config_types = []
    for item in exp_path.iterdir():
        if item.is_dir() and item.name.startswith('decentralized'):
            config_types.append(item.name)
    
    return sorted(config_types)


def print_detailed_run_info(config_data, config_name):
    """
    Print detailed information for each run of a configuration.
    
    Args:
        config_data: List of run data for a configuration
        config_name: Name of the configuration
    """
    print(f"\n  Detailed Run Information for {config_name}:")
    print(f"    {'Run':<6} {'Avg Time (s)':<15} {'Std Dev (s)':<15} {'Total Time (s)':<15} {'Num Batches':<12}")
    print(f"    {'-'*6} {'-'*15} {'-'*15} {'-'*15} {'-'*12}")
    
    for run_data in config_data:
        avg_time = f"{run_data['avg_time']:.6f}" if run_data['avg_time'] is not None else "N/A"
        std_dev = f"{run_data['std_dev']:.6f}" if run_data['std_dev'] is not None else "N/A"
        total_time = f"{sum(run_data['times']):.6f}" if run_data['times'] else "N/A"
        num_batches = len(run_data['times']) if run_data['times'] else 0
        
        print(f"    {run_data['run']:<6} {avg_time:<15} {std_dev:<15} {total_time:<15} {num_batches:<12}")


def print_aggregated_stats(config_data, config_name):
    """
    Print aggregated statistics for a configuration.
    
    Args:
        config_data: List of run data for a configuration
        config_name: Name of the configuration
    """
    # Extract valid data for aggregation
    valid_runs = [run for run in config_data if run['avg_time'] is not None]
    
    if not valid_runs:
        print(f"\n  {config_name}: No valid runs found")
        return
    
    avg_times = [run['avg_time'] for run in valid_runs]
    std_devs = [run['std_dev'] for run in valid_runs]
    total_times = [sum(run['times']) for run in valid_runs]
    num_batches_list = [len(run['times']) for run in valid_runs]
    
    # Calculate aggregates
    avg_of_avgs = statistics.mean(avg_times)
    std_of_avgs = statistics.stdev(avg_times) if len(avg_times) > 1 else 0.0
    avg_std_dev = statistics.mean(std_devs)
    avg_total_time = statistics.mean(total_times)
    std_total_time = statistics.stdev(total_times) if len(total_times) > 1 else 0.0
    avg_num_batches = statistics.mean(num_batches_list)
    
    print(f"\n  Aggregated Statistics for {config_name}:")
    print(f"    Average of averages: {avg_of_avgs:.6f}s ± {std_of_avgs:.6f}s")
    print(f"    Average std deviation: {avg_std_dev:.6f}s")
    print(f"    Average total time: {avg_total_time:.6f}s ± {std_total_time:.6f}s")
    print(f"    Average number of batches: {avg_num_batches:.2f}")
    
    # Show min/max for context
    print(f"    Min/Max average time: {min(avg_times):.6f}s / {max(avg_times):.6f}s")
    print(f"    Min/Max total time: {min(total_times):.6f}s / {max(total_times):.6f}s")


def analyze_centralized_runs(experiment):
    """
    Analyze logs from centralized runs of an experiment.

    Args:
        experiment: Name of the experiment directory

    Returns:
        Dictionary with timing statistics for centralized runs
    """
    base_dir = Path(__file__).parent

    # Analyze each run
    centralized_runs_data = []
    for run in range(1, 4):
        run_data = analyze_single_run(experiment, run, "centralized")
        centralized_runs_data.append(run_data)

    return centralized_runs_data


def calculate_speedup(centralized_data, decentralized_data):
    """
    Calculate speedup percentage between centralized and decentralized configurations.

    Args:
        centralized_data: List of run data for centralized configuration
        decentralized_data: List of run data for decentralized configuration

    Returns:
        Dictionary with speedup statistics
    """
    # Extract valid centralized total times
    cent_valid_runs = [run for run in centralized_data if run['avg_time'] is not None]
    if not cent_valid_runs:
        return {'avg_percent_speedup': None, 'time_saved': None, 'std_dev_time_saved': None}

    cent_total_times = [sum(run['times']) for run in cent_valid_runs]
    avg_cent_total = statistics.mean(cent_total_times)
    std_cent_total = statistics.stdev(cent_total_times) if len(cent_total_times) > 1 else 0.0

    # Extract valid decentralized total times
    decen_valid_runs = [run for run in decentralized_data if run['avg_time'] is not None]
    if not decen_valid_runs:
        return {'avg_percent_speedup': None, 'time_saved': None, 'std_dev_time_saved': None}

    decen_total_times = [sum(run['times']) for run in decen_valid_runs]
    avg_decen_total = statistics.mean(decen_total_times)
    std_decen_total = statistics.stdev(decen_total_times) if len(decen_total_times) > 1 else 0.0

    # Calculate speedup
    time_saved = avg_cent_total - avg_decen_total
    percent_speedup = (time_saved / avg_cent_total) * 100 if avg_cent_total != 0 else 0

    # Calculate std dev of time saved
    std_dev_time_saved = ((std_cent_total ** 2 + std_decen_total ** 2) ** 0.5) if std_cent_total and std_decen_total else 0.0

    return {
        'avg_percent_speedup': percent_speedup,
        'time_saved': time_saved,
        'std_dev_time_saved': std_dev_time_saved
    }


def analyze_and_compare_decentralized(experiment):
    """
    Compare all decentralized configurations within a single experiment.

    Args:
        experiment: Name of the experiment directory
    """
    print(f"Analyzing decentralized configurations in '{experiment}'...")

    # Get all decentralized configuration types
    decentralized_configs = get_decentralized_configs(experiment)

    if not decentralized_configs:
        print(f"No decentralized configurations found in '{experiment}'")
        return

    print(f"\nFound {len(decentralized_configs)} decentralized configurations:")
    for config in decentralized_configs:
        print(f"  - {config}")

    # Analyze centralized runs to get baseline
    centralized_runs_data = analyze_centralized_runs(experiment)

    # Check if centralized data exists
    cent_valid_runs = [run for run in centralized_runs_data if run['avg_time'] is not None]
    if not cent_valid_runs:
        print("Warning: No valid centralized data found for comparison")
        baseline_available = False
    else:
        baseline_available = True
        cent_total_times = [sum(run['times']) for run in cent_valid_runs]
        avg_cent_total = statistics.mean(cent_total_times)
        print(f"\nCentralized baseline: {avg_cent_total:.3f}s total training time")

    # Analyze each configuration
    all_config_data = {}

    for config_type in decentralized_configs:
        print(f"\n{'='*60}")
        print(f"CONFIGURATION: {config_type}")
        print(f"{'='*60}")

        # Analyze all runs for this config
        config_runs_data = []
        for run in range(1, 4):
            run_data = analyze_single_run(experiment, run, config_type)
            config_runs_data.append(run_data)

        # Store for later comparison
        all_config_data[config_type] = config_runs_data

        # Print detailed run info
        print_detailed_run_info(config_runs_data, config_type)

        # Print aggregated stats
        print_aggregated_stats(config_runs_data, config_type)

    # Print cross-configuration comparison
    print(f"\n{'='*80}")
    print("CROSS-CONFIGURATION COMPARISON")
    print(f"{'='*80}")

    print(f"{'Config Type':<20} {'Avg Time (s)':<15} {'Std Dev (s)':<15} {'Total Time (s)':<18} {'Num Batches':<12}")
    print(f"{'-'*20} {'-'*15} {'-'*15} {'-'*18} {'-'*12}")

    for config_type, config_runs_data in all_config_data.items():
        # Calculate aggregated stats for this config
        valid_runs = [run for run in config_runs_data if run['avg_time'] is not None]

        if valid_runs:
            avg_times = [run['avg_time'] for run in valid_runs]
            total_times = [sum(run['times']) for run in valid_runs]
            num_batches_list = [len(run['times']) for run in valid_runs]

            avg_of_avgs = statistics.mean(avg_times)
            std_of_avgs = statistics.stdev(avg_times) if len(avg_times) > 1 else 0.0
            avg_total_time = statistics.mean(total_times)
            avg_num_batches = statistics.mean(num_batches_list)

            print(f"{config_type:<20} {avg_of_avgs:.6f}±{std_of_avgs:.6f}  {'N/A':<15} {avg_total_time:.6f}±{'N/A':<10} {avg_num_batches:<12.2f}")
        else:
            print(f"{config_type:<20} {'N/A':<15} {'N/A':<15} {'N/A':<18} {'N/A':<12}")

    # Print ranking by performance
    print(f"\nRANKING BY AVERAGE TIME PER BATCH (fastest to slowest):")
    perf_rankings = []

    for config_type, config_runs_data in all_config_data.items():
        valid_runs = [run for run in config_runs_data if run['avg_time'] is not None]
        if valid_runs:
            avg_times = [run['avg_time'] for run in valid_runs]
            avg_of_avgs = statistics.mean(avg_times)
            perf_rankings.append((config_type, avg_of_avgs))

    # Sort by average time (ascending)
    perf_rankings.sort(key=lambda x: x[1])

    for i, (config_type, avg_time) in enumerate(perf_rankings, 1):
        print(f"  {i}. {config_type}: {avg_time:.6f}s")

    # Print ranking by total time
    print(f"\nRANKING BY TOTAL TRAINING TIME (fastest to slowest):")
    total_rankings = []

    for config_type, config_runs_data in all_config_data.items():
        valid_runs = [run for run in config_runs_data if run['avg_time'] is not None]
        if valid_runs:
            total_times = [sum(run['times']) for run in valid_runs]
            avg_total_time = statistics.mean(total_times)
            total_rankings.append((config_type, avg_total_time))

    # Sort by total time (ascending)
    total_rankings.sort(key=lambda x: x[1])

    for i, (config_type, avg_total_time) in enumerate(total_rankings, 1):
        print(f"  {i}. {config_type}: {avg_total_time:.6f}s")

    # Print the requested format for each decentralized configuration
    print(f"\n{'='*80}")
    print("DETAILED RESULTS IN REQUESTED FORMAT")
    print(f"{'='*80}")

    for config_type, config_runs_data in all_config_data.items():
        valid_runs = [run for run in config_runs_data if run['avg_time'] is not None]

        if valid_runs:
            # Calculate total times for each run
            total_times = [sum(run['times']) for run in valid_runs]

            avg_total_time = statistics.mean(total_times)

            # Calculate the standard deviation of the total times
            std_of_totals = statistics.stdev(total_times) if len(total_times) > 1 else 0.0

            # Calculate the percentage that std dev is of the total time
            if avg_total_time != 0:
                percent_std = (std_of_totals / avg_total_time) * 100
            else:
                percent_std = 0.0

            # Get the selected configuration from the first valid run (assuming it's consistent)
            selected_config = valid_runs[0]['selected_config'] if valid_runs and valid_runs[0]['selected_config'] else "N/A"

            # Calculate speedup if baseline is available
            if baseline_available:
                speedup_stats = calculate_speedup(centralized_runs_data, config_runs_data)
                avg_percent_speedup = speedup_stats['avg_percent_speedup']
                time_saved = speedup_stats['time_saved']
                std_dev_time_saved = speedup_stats['std_dev_time_saved']

                # Calculate the percentage that std dev is of the time saved
                if time_saved != 0 and time_saved != None:
                    std_dev_pct_saved = (abs(std_dev_time_saved) / abs(time_saved)) * 100
                else:
                    std_dev_pct_saved = 0.0

                print(f"{config_type}: {avg_total_time:.3f}s ± {std_of_totals:.3f}s ({percent_std:.2f}%) - selected: {selected_config} - % Total Time Decrease: {avg_percent_speedup:.3f}% ({time_saved:.3f}s ± {std_dev_time_saved:.3f}s ({std_dev_pct_saved:.2f}%))")
            else:
                print(f"{config_type}: {avg_total_time:.3f}s ± {std_of_totals:.3f}s ({percent_std:.2f}%) - selected: {selected_config}")
        else:
            print(f"{config_type}: No valid data")


def main():
    parser = argparse.ArgumentParser(description="Compare decentralized configurations within an experiment")
    parser.add_argument("experiment", help="Name of the experiment directory (e.g., 'mac_b64' or 'mac_b128')")
    
    args = parser.parse_args()
    
    # Validate experiment directory exists
    base_dir = Path(__file__).parent
    exp_path = base_dir / args.experiment
    if not exp_path.exists():
        print(f"Error: Experiment directory does not exist: {exp_path}")
        return
    
    # Validate it's one of the expected experiments
    if args.experiment not in ['mac_b64', 'mac_b128']:
        print(f"Warning: '{args.experiment}' is not one of the expected experiments (mac_b64, mac_b128)")
    
    analyze_and_compare_decentralized(args.experiment)


if __name__ == "__main__":
    main()

