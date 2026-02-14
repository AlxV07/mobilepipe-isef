#!/usr/bin/env python3
"""
Idle Time Calculation Script for CNN Experiments

This script analyzes JSON log files from the selected decentralized configuration
to calculate the idle time for host and client devices based on the
time spent waiting for gradients (host) and waiting for input (client).

Usage:
    python calculate_idle_times.py EXPERIMENT

Arguments:
    EXPERIMENT  Name of the experiment directory (e.g., 'mac_b64', 'mac_b128', etc.)

Examples:
    python calculate_idle_times.py mac_b64
    python calculate_idle_times.py mac_b128

The script expects the following directory structure:
    EXPERIMENT/
    └── decentralized_*/ (various memory configs)
        ├── run1/
        │   ├── MobilePipe_C1.json
        │   ├── MobilePipe_C2.json
        │   ├── MobilePipe_C3.json
        │   ├── MobilePipe_training.json
        │   └── log.txt
        ├── run2/
        │   ├── MobilePipe_C1.json
        │   ├── MobilePipe_C2.json
        │   ├── MobilePipe_C3.json
        │   ├── MobilePipe_training.json
        │   └── log.txt
        └── run3/
            ├── MobilePipe_C1.json
            ├── MobilePipe_C2.json
            ├── MobilePipe_C3.json
            ├── MobilePipe_training.json
            └── log.txt
"""

import argparse
import json
import os
import re
import statistics
from pathlib import Path


def calculate_idle_time_from_intervals(intervals):
    """
    Calculate total idle time from a list of time intervals.

    Args:
        intervals: List of [start_time, end_time] pairs

    Returns:
        Total idle time in seconds
    """
    total_idle_time = 0.0

    for interval in intervals:
        if len(interval) >= 2:
            start_time, end_time = interval[0], interval[1]
            total_idle_time += (end_time - start_time)

    return total_idle_time


def calculate_additional_client_idle_time(ios_s2_intervals, host_full_batch_intervals):
    """
    Calculate additional client idle time: time between end of last unified compute batch
    and end of last host batch in each batch set.

    Args:
        ios_s2_intervals: List of [start_time, end_time] pairs for unified compute batches
        host_full_batch_intervals: List of [start_time, end_time] pairs for host batches

    Returns:
        Additional client idle time in seconds
    """
    additional_idle_time = 0.0

    if not ios_s2_intervals or not host_full_batch_intervals:
        return additional_idle_time

    # Sort intervals by start time to ensure proper ordering
    ios_s2_sorted = sorted(ios_s2_intervals, key=lambda x: x[0])
    host_full_batch_sorted = sorted(host_full_batch_intervals, key=lambda x: x[0])

    # Assuming there are 3 batch sets based on the architecture
    # We need to group the ios_s2 intervals according to the host batch sets
    for host_batch in host_full_batch_sorted:
        host_start, host_end = host_batch[0], host_batch[1]

        # Find ios_s2 intervals that fall within this host batch timeframe
        relevant_ios_s2 = [interval for interval in ios_s2_sorted
                           if interval[0] >= host_start and interval[1] <= host_end]

        if relevant_ios_s2:
            # Find the last ios_s2 interval in this batch set
            last_ios_s2_end = max(interval[1] for interval in relevant_ios_s2)

            # The additional idle time is the time between the end of the last ios_s2
            # and the end of the host batch
            additional_idle_time += (host_end - last_ios_s2_end)

    return additional_idle_time


def analyze_json_file(filepath):
    """
    Analyze a JSON log file and extract idle time information.

    Args:
        filepath: Path to the JSON file

    Returns:
        Dictionary with host_idle_time and client_idle_time
    """
    if not os.path.exists(filepath):
        return {'host_idle_time': 0.0, 'client_idle_time': 0.0}

    with open(filepath, 'r') as f:
        data = json.load(f)

    # Calculate host idle time from HOST_WAITING_FOR_GRAD intervals
    host_waiting_for_grad_intervals = data.get('HOST_WAITING_FOR_GRAD', [])
    host_idle_time = calculate_idle_time_from_intervals(host_waiting_for_grad_intervals)

    # Calculate client idle time from IOS_WAITING_FOR_INPUT intervals
    ios_waiting_for_input_intervals = data.get('IOS_WAITING_FOR_INPUT', [])
    client_idle_time = calculate_idle_time_from_intervals(ios_waiting_for_input_intervals)

    # Add additional client idle time: time between end of last unified compute batch
    # and end of last host batch in each batch set
    ios_s2_intervals = data.get('IOS_S2_COMBINED_MICROBATCH', [])  # Unified compute batches
    host_full_batch_intervals = data.get('HOST_FULL_BATCH', [])  # Host batches

    # Split the intervals into 3 batch sets and calculate additional idle time for each
    additional_client_idle_time = calculate_additional_client_idle_time(ios_s2_intervals, host_full_batch_intervals)
    client_idle_time += additional_client_idle_time

    return {
        'host_idle_time': host_idle_time,
        'client_idle_time': client_idle_time
    }


def find_selected_config_in_log(log_file_path):
    """
    Find the selected configuration number in the log file.

    Args:
        log_file_path: Path to the log.txt file

    Returns:
        Number of the selected configuration (as string) or None if not found
    """
    if not os.path.exists(log_file_path):
        return None

    with open(log_file_path, 'r') as f:
        for line in f:
            # Look for lines like "Selected: 2"
            match = re.search(r'Selected:\s*(\d+)', line.strip())
            if match:
                return match.group(1).strip()

    return None


def analyze_single_run(experiment, run_num, config_type):
    """
    Analyze all JSON files for a specific run of an experiment.

    Args:
        experiment: Name of the experiment directory
        run_num: Run number (1-3)
        config_type: Type of configuration ('decentralized_*')

    Returns:
        Dictionary with idle time statistics for the run
    """
    base_dir = Path(__file__).parent

    # Define paths to JSON files
    c1_path = base_dir / experiment / config_type / f"run{run_num}" / "MobilePipe_C1.json"
    c2_path = base_dir / experiment / config_type / f"run{run_num}" / "MobilePipe_C2.json"
    c3_path = base_dir / experiment / config_type / f"run{run_num}" / "MobilePipe_C3.json"
    training_path = base_dir / experiment / config_type / f"run{run_num}" / "MobilePipe_training.json"

    result = {
        'run': run_num,
        'config_type': config_type,
        'host_idle_time': 0.0,
        'client_idle_time': 0.0,
        'total_client_idle_time': 0.0  # Sum of all client idle times
    }

    # Analyze each device's JSON file
    for path in [c1_path, c2_path, c3_path]:
        if path.exists():
            device_data = analyze_json_file(path)
            result['host_idle_time'] += device_data['host_idle_time']
            result['total_client_idle_time'] += device_data['client_idle_time']

    # Also analyze the training file for any additional host idle time
    if training_path.exists():
        training_data = analyze_json_file(training_path)
        result['host_idle_time'] += training_data['host_idle_time']
        result['total_client_idle_time'] += training_data['client_idle_time']

    return result


def analyze_and_calculate_idle_times(experiment):
    """
    Calculate idle times for each decentralized configuration within a single experiment.
    For each decentralized configuration folder, find the selected config from run1,
    then calculate idle times across run1, run2, and run3 using that selected config.

    Args:
        experiment: Name of the experiment directory
    """
    base_dir = Path(__file__).parent
    exp_path = base_dir / experiment

    # Get all decentralized configuration types
    decentralized_configs = []
    for item in exp_path.iterdir():
        if item.is_dir() and item.name.startswith('decentralized'):
            decentralized_configs.append(item.name)

    decentralized_configs.sort()  # Sort to ensure consistent ordering

    # Determine the maximum width needed for the left side (config name + type description)
    max_width = 0
    for config_name in decentralized_configs:
        # Calculate width for each type: config_name + " - Host Idle Time", etc.
        widths = [
            len(config_name) + len(" - Host Idle Time"),
            len(config_name) + len(" - Client Idle Time"),
            len(config_name) + len(":")  # for the total line
        ]
        max_width = max(max_width, max(widths))

    # Process each decentralized configuration folder separately
    for config_folder in decentralized_configs:
        # Find the selected configuration in run1 of this folder
        run1_log_path = exp_path / config_folder / "run1" / "log.txt"
        selected_config_number_str = find_selected_config_in_log(run1_log_path)

        if not selected_config_number_str:
            continue

        # The selected config is the one in this folder itself since each folder represents a different experiment
        selected_config_name = config_folder
        selected_config_number = int(selected_config_number_str)

        # Analyze all runs (1, 2, 3) for this selected config
        config_runs_data = []
        for run in range(1, 4):
            run_data = analyze_single_run(experiment, run, selected_config_name)
            config_runs_data.append(run_data)

        # Calculate statistics across runs
        host_idle_times = [run['host_idle_time'] for run in config_runs_data]
        client_idle_times = [run['total_client_idle_time'] for run in config_runs_data]

        # Calculate averages and standard deviations
        avg_host_idle_time = statistics.mean(host_idle_times) if host_idle_times else 0.0
        avg_client_idle_time = statistics.mean(client_idle_times) if client_idle_times else 0.0

        std_host_idle_time = statistics.stdev(host_idle_times) if len(host_idle_times) > 1 else 0.0
        std_client_idle_time = statistics.stdev(client_idle_times) if len(client_idle_times) > 1 else 0.0

        # Calculate percentages
        percent_host_std = (std_host_idle_time / avg_host_idle_time * 100) if avg_host_idle_time != 0 else 0.0
        percent_client_std = (std_client_idle_time / avg_client_idle_time * 100) if avg_client_idle_time != 0 else 0.0

        # Print separate lines for host, client, and total idle times with consistent padding
        padded_config_host = f"{selected_config_name} - Host Idle Time"
        padded_config_client = f"{selected_config_name} - Client Idle Time"
        padded_config_total = f"{selected_config_name}"

        print(f"{padded_config_host:<{max_width}}: {avg_host_idle_time:>8.3f}s ± {std_host_idle_time:>7.3f}s ({percent_host_std:>6.2f}%)")
        print(f"{padded_config_client:<{max_width}}: {avg_client_idle_time:>8.3f}s ± {std_client_idle_time:>7.3f}s ({percent_client_std:>6.2f}%)")

        # Print combined format similar to compare_decentralized.py
        total_idle_time = avg_host_idle_time + avg_client_idle_time
        combined_std = ((std_host_idle_time ** 2 + std_client_idle_time ** 2) ** 0.5) if std_host_idle_time > 0 or std_client_idle_time > 0 else 0.0
        percent_combined_std = (combined_std / total_idle_time * 100) if total_idle_time != 0 else 0.0

        print(f"{padded_config_total:<{max_width}}: {total_idle_time:>8.3f}s ± {combined_std:>7.3f}s ({percent_combined_std:>6.2f}%) - selected: {selected_config_number}")
        print()  # Add a blank line between experiments


def main():
    parser = argparse.ArgumentParser(description="Calculate idle times for host and client in the selected decentralized configuration")
    parser.add_argument("experiment", help="Name of the experiment directory (e.g., 'mac_b64', 'mac_b128')")

    args = parser.parse_args()

    # Validate experiment directory exists
    base_dir = Path(__file__).parent
    exp_path = base_dir / args.experiment
    if not exp_path.exists():
        print(f"Error: Experiment directory does not exist: {exp_path}")
        return

    analyze_and_calculate_idle_times(args.experiment)


if __name__ == "__main__":
    main()

