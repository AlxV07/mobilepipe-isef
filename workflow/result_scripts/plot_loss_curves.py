#!/usr/bin/env python3
"""
Loss Curve Plotting Tool for CNN Experiments

This script plots loss curves from centralized and decentralized training runs,
allowing for visual comparison of training progress.

Usage:
    python plot_loss_curves.py EXPERIMENT [RUN_NUM]

Arguments:
    EXPERIMENT  Name of the experiment directory (e.g., 'linux_b32')
    RUN_NUM     Run number (1-3). If not provided, averages across all runs.

Examples:
    python plot_loss_curves.py linux_b32 1      # Plot run 1 only
    python plot_loss_curves.py linux_b32        # Average across all runs

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
    Parse a log file and extract loss values.

    Args:
        filepath: Path to the log file

    Returns:
        List of loss values for each log entry
    """
    losses = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            # Match centralized format: epoch 1 | step 0/62 | loss = 1.4084 | 1 / 62 | 3.3706s
            centralized_match = re.search(r'loss = ([\d.]+)', line)
            if centralized_match:
                loss = float(centralized_match.group(1))
                losses.append(loss)
                continue

            # Match decentralized format: epoch 1 | step 0/62 | loss = 1.4261 | global_step = 1/62 | 1.782144546508789
            decentralized_match = re.search(r'loss = ([\d.]+)', line)
            if decentralized_match:
                loss = float(decentralized_match.group(1))
                losses.append(loss)

    return losses


def get_loss_data_for_run(experiment, run_num):
    """
    Get loss data for a single run (both centralized and decentralized).

    Args:
        experiment: Name of the experiment directory
        run_num: Run number (1-3)

    Returns:
        Tuple of (centralized_losses, decentralized_losses) or (None, None) if error
    """
    base_dir = Path(__file__).parent

    # Define paths to log files
    centralized_log = base_dir / experiment / "centralized" / f"run{run_num}" / "log.txt"
    decentralized_log = base_dir / experiment / "decentralized" / f"run{run_num}" / "log.txt"

    # Check if files exist
    if not centralized_log.exists():
        print(f"Error: Centralized log file does not exist: {centralized_log}")
        return None, None
    if not decentralized_log.exists():
        print(f"Error: Decentralized log file does not exist: {decentralized_log}")
        return None, None

    # Parse both log files
    cent_losses = parse_log_file(centralized_log)
    decen_losses = parse_log_file(decentralized_log)

    return cent_losses, decen_losses


def plot_single_run(experiment, run_num):
    """
    Plot loss curves for a single run.

    Args:
        experiment: Name of the experiment directory
        run_num: Run number (1-3)
    """
    cent_losses, decen_losses = get_loss_data_for_run(experiment, run_num)

    if cent_losses is None or decen_losses is None:
        return

    # Determine the minimum length to ensure both curves align properly
    min_len = min(len(cent_losses), len(decen_losses))
    cent_losses = cent_losses[:min_len]
    decen_losses = decen_losses[:min_len]

    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot centralized (blue) and decentralized (red) loss curves
    plt.plot(range(len(cent_losses)), cent_losses, label='Centralized', color='red', linewidth=1.5)
    plt.plot(range(len(decen_losses)), decen_losses, label='Decentralized', color='blue', linewidth=1.5)
    
    plt.title(f'Loss Curves Comparison - {experiment} Run {run_num}')
    plt.xlabel('Global Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_average_curves(experiment):
    """
    Plot averaged loss curves across all runs.

    Args:
        experiment: Name of the experiment directory
    """
    all_cent_losses = []
    all_decen_losses = []
    
    # Collect data from all runs
    for run in range(1, 4):
        cent_losses, decen_losses = get_loss_data_for_run(experiment, run)
        
        if cent_losses is not None and decen_losses is not None:
            all_cent_losses.append(cent_losses)
            all_decen_losses.append(decen_losses)
        else:
            print(f"Warning: Could not load data for run {run}, skipping...")
    
    if not all_cent_losses or not all_decen_losses:
        print("Error: No valid data found for any run.")
        return
    
    # Find the minimum length across all runs to ensure alignment
    min_len = min(min(len(run_losses) for run_losses in all_cent_losses),
                  min(len(run_losses) for run_losses in all_decen_losses))
    
    # Trim all runs to the same length
    all_cent_losses = [run_losses[:min_len] for run_losses in all_cent_losses]
    all_decen_losses = [run_losses[:min_len] for run_losses in all_decen_losses]
    
    # Calculate averages at each step
    avg_cent_losses = []
    avg_decen_losses = []
    
    for i in range(min_len):
        cent_vals = [run[i] for run in all_cent_losses]
        decen_vals = [run[i] for run in all_decen_losses]
        
        avg_cent = sum(cent_vals) / len(cent_vals)
        avg_decen = sum(decen_vals) / len(decen_vals)
        
        avg_cent_losses.append(avg_cent)
        avg_decen_losses.append(avg_decen)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    plt.plot(range(len(avg_cent_losses)), avg_cent_losses, label='Centralized (Avg)', color='red', linewidth=2)
    plt.plot(range(len(avg_decen_losses)), avg_decen_losses, label='Decentralized (Avg)', color='blue', linewidth=2)
    
    plt.title(f'Average Loss Curves Comparison - {experiment} (All Runs)')
    plt.xlabel('Global Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot loss curves from experiment logs")
    parser.add_argument("experiment", help="Name of the experiment directory")
    parser.add_argument("run", type=int, nargs='?', choices=range(1, 4),
                        help="Run number (1-3). If not provided, averages across all runs.")

    args = parser.parse_args()

    # Validate experiment directory exists
    exp_path = Path(__file__).parent / args.experiment
    if not exp_path.exists():
        print(f"Error: Experiment directory does not exist: {exp_path}")
        return

    if args.run is not None:
        plot_single_run(args.experiment, args.run)
    else:
        plot_average_curves(args.experiment)


if __name__ == "__main__":
    main()

