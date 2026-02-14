#!/usr/bin/env python3
"""
Batch Time Plotting Script for CNN Experiments

This script plots the average time per batch for each corresponding batch 
across all 3 runs in an experiment, with decentralized in green and 
centralized in red.

Usage:
    python plot_batch_times.py EXPERIMENT

Arguments:
    EXPERIMENT  Name of the experiment directory (e.g., 'linux_b32')

Examples:
    python plot_batch_times.py linux_b32

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
import matplotlib.pyplot as plt
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


def get_batch_times_for_experiment(experiment):
    """
    Extract batch times for all runs in an experiment.

    Args:
        experiment: Name of the experiment directory

    Returns:
        Dictionary with batch times for centralized and decentralized runs
    """
    base_dir = Path(__file__).parent

    result = {
        'centralized': [],
        'decentralized': []
    }

    # Process each run (1-3)
    for run_num in range(1, 4):
        # Define paths to log files
        centralized_log = base_dir / experiment / "centralized" / f"run{run_num}" / "log.txt"
        decentralized_log = base_dir / experiment / "decentralized" / f"run{run_num}" / "log.txt"

        # Parse centralized log if it exists
        if centralized_log.exists():
            centralized_times = parse_log_file(centralized_log)
            result['centralized'].append(centralized_times)

        # Parse decentralized log if it exists
        if decentralized_log.exists():
            decentralized_times = parse_log_file(decentralized_log)
            result['decentralized'].append(decentralized_times)

    return result


def plot_batch_times(experiment):
    """
    Plot the average time per batch for each corresponding batch across all 3 runs.

    Args:
        experiment: Name of the experiment directory
    """
    print(f"Plotting batch times for experiment '{experiment}'...")

    # Get batch times for the experiment
    batch_times = get_batch_times_for_experiment(experiment)

    # Determine the maximum number of batches across all runs
    max_batches_cent = 0
    max_batches_decen = 0
    
    if batch_times['centralized']:
        max_batches_cent = max(len(times) for times in batch_times['centralized']) if batch_times['centralized'] else 0
    
    if batch_times['decentralized']:
        max_batches_decen = max(len(times) for times in batch_times['decentralized']) if batch_times['decentralized'] else 0
    
    max_batches = max(max_batches_cent, max_batches_decen)

    if max_batches == 0:
        print(f"No valid log files found for experiment '{experiment}'")
        return

    # Prepare x-axis (batch numbers)
    batch_numbers = list(range(1, max_batches + 1))

    # Calculate average times per batch for centralized and decentralized
    avg_centralized_times = []
    avg_decentralized_times = []

    # Calculate average centralized times per batch
    for batch_idx in range(max_batches):
        batch_times_list = []
        for run_times in batch_times['centralized']:
            if batch_idx < len(run_times):
                batch_times_list.append(run_times[batch_idx])
        
        if batch_times_list:
            avg_time = sum(batch_times_list) / len(batch_times_list)
            avg_centralized_times.append(avg_time)
        else:
            avg_centralized_times.append(None)  # Placeholder if no data for this batch

    # Calculate average decentralized times per batch
    for batch_idx in range(max_batches):
        batch_times_list = []
        for run_times in batch_times['decentralized']:
            if batch_idx < len(run_times):
                batch_times_list.append(run_times[batch_idx])
        
        if batch_times_list:
            avg_time = sum(batch_times_list) / len(batch_times_list)
            avg_decentralized_times.append(avg_time)
        else:
            avg_decentralized_times.append(None)  # Placeholder if no data for this batch

    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot centralized times in red
    if avg_centralized_times:
        valid_indices = [i for i, val in enumerate(avg_centralized_times) if val is not None]
        valid_values = [val for val in avg_centralized_times if val is not None]
        if valid_values:
            plt.plot([batch_numbers[i] for i in valid_indices], valid_values, 
                     color='red', label='Centralized', marker='o', markersize=3, linewidth=1)

    # Plot decentralized times in blue
    if avg_decentralized_times:
        valid_indices = [i for i, val in enumerate(avg_decentralized_times) if val is not None]
        valid_values = [val for val in avg_decentralized_times if val is not None]
        if valid_values:
            plt.plot([batch_numbers[i] for i in valid_indices], valid_values,
                     color='blue', label='Decentralized', marker='o', markersize=3, linewidth=1)

    plt.xlabel('Batch Number')
    plt.ylabel('Average Time per Batch (seconds)')
    plt.title(f'Average Time per Batch Comparison - {experiment}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)

    # Show the plot
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot average time per batch for experiment runs")
    parser.add_argument("experiment", help="Name of the experiment directory")

    args = parser.parse_args()

    # Validate experiment directory exists
    exp_path = Path(__file__).parent / args.experiment
    if not exp_path.exists():
        print(f"Error: Experiment directory does not exist: {exp_path}")
        return

    plot_batch_times(args.experiment)


if __name__ == "__main__":
    main()

