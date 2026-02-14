#!/usr/bin/env python3
"""
Batch Time Bar Plotting Script for CNN Experiments

This script plots the average time per batch for each corresponding batch
across all 3 runs in an experiment using bar charts, with decentralized 
in green and centralized in red.

Usage:
    python plot_bar_batch_times.py EXPERIMENT

Arguments:
    EXPERIMENT  Name of the experiment directory (e.g., 'linux_b32')

Examples:
    python plot_bar_batch_times.py linux_b32

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
import numpy as np
from pathlib import Path
from scipy.interpolate import make_interp_spline


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
                # Check that the value is actually a time value (not step count)
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


def plot_bar_batch_times(experiment):
    """
    Plot the average time per batch for each corresponding batch across all 3 runs using bar charts.

    Args:
        experiment: Name of the experiment directory
    """
    print(f"Plotting batch times as bars for experiment '{experiment}'...")

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

    display_batches = max_batches

    # Calculate average times per batch for centralized and decentralized
    avg_centralized_times = []
    avg_decentralized_times = []

    # Calculate average centralized times per batch
    for batch_idx in range(display_batches):
        batch_times_list = []
        for run_times in batch_times['centralized']:
            if batch_idx < len(run_times):
                batch_times_list.append(run_times[batch_idx])

        if batch_times_list:
            avg_time = sum(batch_times_list) / len(batch_times_list)
            avg_centralized_times.append(avg_time)
        else:
            avg_centralized_times.append(0)  # Use 0 if no data for this batch

    # Calculate average decentralized times per batch
    for batch_idx in range(display_batches):
        batch_times_list = []
        for run_times in batch_times['decentralized']:
            if batch_idx < len(run_times):
                batch_times_list.append(run_times[batch_idx])

        if batch_times_list:
            avg_time = sum(batch_times_list) / len(batch_times_list)
            avg_decentralized_times.append(avg_time)
        else:
            avg_decentralized_times.append(0)  # Use 0 if no data for this batch

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # Set the positions for the bars
    x_pos = np.arange(display_batches)

    # Create smooth trend lines across the tops of the bars with lower z-order
    # Only create smooth lines if we have enough points for interpolation
    if len(x_pos) >= 3:  # Need at least 3 points for smoothing
        # Create more points for smoother curve
        x_smooth = np.linspace(x_pos.min(), x_pos.max(), 300)

        # Create smooth centralized trend line (lighter red)
        if avg_centralized_times and any(t is not None and t != 0 for t in avg_centralized_times):
            # Filter out None values and create arrays without None
            valid_indices = [i for i, val in enumerate(avg_centralized_times) if val is not None and val != 0]
            if len(valid_indices) >= 3:  # Need at least 3 valid points for interpolation
                x_valid = [x_pos[i] for i in valid_indices]
                y_valid = [avg_centralized_times[i] for i in valid_indices]

                # Create spline interpolation
                spl = make_interp_spline(x_valid, y_valid, k=2)
                y_smooth_cent = spl(x_smooth)
                ax.plot(x_smooth, y_smooth_cent, color='pink', linestyle='-', linewidth=1.5, label='_Centralized_Trend', zorder=1)
            else:
                # Fallback to regular line if not enough points for smoothing
                ax.plot(x_pos, avg_centralized_times, color='pink', linestyle='-', linewidth=1.5, marker='', label='_Centralized_Trend', zorder=1)
        else:
            # Fallback to regular line if no valid data
            ax.plot(x_pos, avg_centralized_times, color='pink', linestyle='-', linewidth=1.5, marker='', label='_Centralized_Trend', zorder=1)

        # Create smooth decentralized trend line (lighter blue)
        if avg_decentralized_times and any(t is not None and t != 0 for t in avg_decentralized_times):
            # Filter out None values and create arrays without None
            valid_indices = [i for i, val in enumerate(avg_decentralized_times) if val is not None and val != 0]
            if len(valid_indices) >= 3:  # Need at least 3 valid points for interpolation
                x_valid = [x_pos[i] for i in valid_indices]
                y_valid = [avg_decentralized_times[i] for i in valid_indices]

                # Create spline interpolation
                spl = make_interp_spline(x_valid, y_valid, k=2)
                y_smooth_decent = spl(x_smooth)
                ax.plot(x_smooth, y_smooth_decent, color='lightblue', linestyle='-', linewidth=1.5, label='_Decentralized_Trend', zorder=1)
            else:
                # Fallback to regular line if not enough points for smoothing
                ax.plot(x_pos, avg_decentralized_times, color='lightblue', linestyle='-', linewidth=1.5, marker='', label='_Decentralized_Trend', zorder=1)
        else:
            # Fallback to regular line if no valid data
            ax.plot(x_pos, avg_decentralized_times, color='lightblue', linestyle='-', linewidth=1.5, marker='', label='_Decentralized_Trend', zorder=1)
    else:
        # Fallback to regular lines if not enough points for smoothing
        ax.plot(x_pos, avg_centralized_times, color='pink', linestyle='-', linewidth=1.5, marker='', label='_Centralized_Trend', zorder=1)
        ax.plot(x_pos, avg_decentralized_times, color='lightblue', linestyle='-', linewidth=1.5, marker='', label='_Decentralized_Trend', zorder=1)

    # Create bars that are shifted just slightly to avoid complete overlap while staying close
    # Bars are drawn with higher z-order so they appear on top
    bar_width = 0.4
    bars_centralized = ax.bar(x_pos - bar_width/8, avg_centralized_times,
                              width=bar_width*0.8, label='Centralized', color='red', alpha=0.8, zorder=2)
    bars_decentralized = ax.bar(x_pos + bar_width/8, avg_decentralized_times,
                                width=bar_width*0.8, label='Decentralized', color='blue', alpha=0.8, zorder=2)

    # Add labels and title
    ax.set_xlabel('Batch Number')
    ax.set_ylabel('Average Time per Batch (seconds)')
    ax.set_title(f'Average Time per Batch Comparison - {experiment}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(i+1) for i in x_pos])

    # Add legend (filter out the hidden trend line labels)
    handles, labels = ax.get_legend_handles_labels()
    # Remove the hidden trend line labels (those starting with '_')
    filtered_handles_labels = [(h, l) for h, l in zip(handles, labels) if not l.startswith('_')]
    final_handles, final_labels = zip(*filtered_handles_labels) if filtered_handles_labels else ([], [])
    ax.legend(final_handles, final_labels)

    # Add grid for better readability
    ax.grid(True, axis='y', alpha=0.3)

    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)

    # Rotate x-axis labels if there are many batches
    if display_batches > 10:
        plt.xticks(rotation=45, ha="right")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Show the plot
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot average time per batch for experiment runs using bar charts")
    parser.add_argument("experiment", help="Name of the experiment directory")

    args = parser.parse_args()

    # Validate experiment directory exists
    exp_path = Path(__file__).parent / args.experiment
    if not exp_path.exists():
        print(f"Error: Experiment directory does not exist: {exp_path}")
        return

    plot_bar_batch_times(args.experiment)


if __name__ == "__main__":
    main()

