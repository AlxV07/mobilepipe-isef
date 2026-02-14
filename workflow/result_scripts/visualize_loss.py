#!/usr/bin/env python3
"""
Loss Visualization Script for CNN Experiments

This script visualizes loss values from log files of centralized and decentralized training runs.

Usage:
    python visualize_loss.py EXPERIMENT [RUN_NUM]

Arguments:
    EXPERIMENT  Name of the experiment directory (e.g., 'linux_b32')
    RUN_NUM     Run number (1-3). If not provided, defaults to 1.

Examples:
    python visualize_loss.py linux_b32 1      # Visualize run 1 only
    python visualize_loss.py linux_b32        # Visualize run 1 by default

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
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_log_file(log_file_path):
    """
    Parse the log file and extract loss values along with epoch and step information.

    Args:
        log_file_path (str): Path to the log file

    Returns:
        dict: Dictionary containing epochs, steps, losses, and global step counts
    """
    epochs = []
    steps = []
    losses = []
    global_steps = []

    # Regular expression to match the log format
    # Looking for patterns like: epoch X | step Y/Z | loss = W.WWWW | A / B | C.s
    pattern = r'epoch (\d+) \| step (\d+)/\d+ \| loss = (\d+\.?\d*) \| (\d+) / \d+ \| \d+\.?\d*s'

    with open(log_file_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line.strip())
            if match:
                epoch = int(match.group(1))
                step = int(match.group(2))
                loss = float(match.group(3))
                global_step = int(match.group(4))

                epochs.append(epoch)
                steps.append(step)
                losses.append(loss)
                global_steps.append(global_step)

    return {
        'epochs': epochs,
        'steps': steps,
        'losses': losses,
        'global_steps': global_steps
    }


def plot_losses(data, experiment, run_num, approach):
    """
    Create plots for the extracted loss data.

    Args:
        data (dict): Dictionary containing epochs, steps, losses, and global step counts
        experiment (str): Name of the experiment
        run_num (int): Run number
        approach (str): Training approach ('centralized' or 'decentralized')
    """
    epochs = data['epochs']
    steps = data['steps']
    losses = data['losses']
    global_steps = data['global_steps']

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Loss Visualization - {experiment} - {approach} - Run {run_num}', fontsize=16)

    # Plot 1: Loss vs Global Step
    axes[0, 0].plot(global_steps, losses, linewidth=0.8)
    axes[0, 0].set_title('Loss vs Global Step')
    axes[0, 0].set_xlabel('Global Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Loss vs Epoch (averaged per epoch)
    unique_epochs = sorted(set(epochs))
    avg_losses_per_epoch = []
    for epoch in unique_epochs:
        epoch_losses = [loss for i, loss in enumerate(losses) if epochs[i] == epoch]
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_losses_per_epoch.append(avg_loss)

    axes[0, 1].plot(unique_epochs, avg_losses_per_epoch, marker='o', linewidth=2, markersize=8)
    axes[0, 1].set_title('Average Loss per Epoch')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Average Loss')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Loss vs Step (for each epoch separately)
    for epoch in unique_epochs:
        epoch_indices = [i for i, e in enumerate(epochs) if e == epoch]
        epoch_steps = [steps[i] for i in epoch_indices]
        epoch_losses = [losses[i] for i in epoch_indices]

        # Only plot first 3 epochs to avoid overcrowding
        if epoch <= 3:
            axes[1, 0].plot(epoch_steps, epoch_losses, label=f'Epoch {epoch}', alpha=0.7)

    axes[1, 0].set_title('Loss vs Step (First 3 Epochs)')
    axes[1, 0].set_xlabel('Step within Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Moving average of loss to smooth out fluctuations
    window_size = min(50, len(losses) // 10)  # Use 5% of data points for smoothing
    if window_size > 1:
        # Calculate moving average using numpy's convolve function for better alignment
        weights = np.ones(window_size) / window_size
        ma = np.convolve(losses, weights, mode='valid')

        # The moving average starts at index window_size-1
        start_idx = window_size - 1
        x_ma = range(start_idx, start_idx + len(ma))

        axes[1, 1].plot(range(len(losses)), losses, alpha=0.3, label='Original', linewidth=0.5)
        axes[1, 1].plot(x_ma, ma, label=f'Moving Average (window={window_size})', linewidth=2)
        axes[1, 1].set_title(f'Loss with Moving Average (window={window_size})')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Not enough data\nfor moving average',
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Moving Average Plot')

    plt.tight_layout()
    plt.show()


def visualize_experiment(experiment, run_num, approach='centralized'):
    """
    Visualize loss data from a specific experiment run.

    Args:
        experiment: Name of the experiment directory
        run_num: Run number (1-3)
        approach: Training approach ('centralized' or 'decentralized')
    """
    base_dir = Path(__file__).parent

    # Define path to log file
    log_file_path = base_dir / experiment / approach / f"run{run_num}" / "log.txt"

    # Check if file exists
    if not log_file_path.exists():
        print(f"Error: Log file does not exist: {log_file_path}")
        return

    print(f"Parsing log file: {log_file_path}")
    data = parse_log_file(log_file_path)

    if not data['losses']:
        print("No loss values found in the log file.")
        return

    print(f"Found {len(data['losses'])} loss values across {len(set(data['epochs']))} epochs.")
    print(f"Loss range: {min(data['losses']):.4f} to {max(data['losses']):.4f}")

    print("Creating visualizations...")
    plot_losses(data, experiment, run_num, approach)


def main():
    parser = argparse.ArgumentParser(description="Visualize loss values from experiment logs")
    parser.add_argument("experiment", help="Name of the experiment directory")
    parser.add_argument("run", type=int, nargs='?', default=1, choices=range(1, 4),
                        help="Run number (1-3). Defaults to 1 if not provided.")

    args = parser.parse_args()

    # Validate experiment directory exists
    exp_path = Path(__file__).parent / args.experiment
    if not exp_path.exists():
        print(f"Error: Experiment directory does not exist: {exp_path}")
        return

    # Visualize both centralized and decentralized approaches
    for approach in ['centralized', 'decentralized']:
        print(f"\nVisualizing {approach} approach...")
        visualize_experiment(args.experiment, args.run, approach)


if __name__ == "__main__":
    main()
		
