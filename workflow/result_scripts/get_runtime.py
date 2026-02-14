#!/usr/bin/env python3
"""
Runtime Extraction Script for CNN Experiments

This script extracts the total runtime from JSON files in both default and mobilepipe training configurations.

Usage:
    python get_runtime.py EXPERIMENT RUN_NUM

Arguments:
    EXPERIMENT  Name of the experiment directory (e.g., 'linux_b32')
    RUN_NUM     Run number (1-3)

Examples:
    python get_runtime.py linux_b32 1

The script expects the following directory structure:
    EXPERIMENT/
    ├── centralized/
    │   └── run{RUN_NUM}/
    │       └── Default_training.json
    └── decentralized/
        └── run{RUN_NUM}/
            └── MobilePipe_training.json
"""

import argparse
import json
import os
from pathlib import Path


def get_total_runtime_from_json(json_path):
    """
    Extract the total runtime from a JSON file.

    Args:
        json_path: Path to the JSON file

    Returns:
        Total runtime in seconds, or None if not found
    """
    if not os.path.exists(json_path):
        return None

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Look for HOST_TRAINING_RUN which contains the overall training time
    if 'HOST_TRAINING_RUN' in data and data['HOST_TRAINING_RUN']:
        # Usually there's one entry in HOST_TRAINING_RUN with [start_time, end_time]
        for start_end_pair in data['HOST_TRAINING_RUN']:
            if len(start_end_pair) >= 2:
                start_time, end_time = start_end_pair[0], start_end_pair[1]
                return end_time - start_time  # Return the duration

    # If HOST_TRAINING_RUN is not present or empty, calculate from all events
    # Find the earliest start time and latest end time across all events
    min_start = float('inf')
    max_end = float('-inf')

    for key, value in data.items():
        if isinstance(value, list) and key != 'HOST_INIT_MODEL_OFFLOAD':  # Skip empty arrays
            for event in value:
                if isinstance(event, list) and len(event) >= 2:
                    start_time, end_time = event[0], event[1]
                    min_start = min(min_start, start_time)
                    max_end = max(max_end, end_time)

    if min_start != float('inf') and max_end != float('-inf'):
        total_runtime = max_end - min_start
        return total_runtime

    return None


def main():
    parser = argparse.ArgumentParser(description="Extract total runtime from training JSON files")
    parser.add_argument("experiment", help="Name of the experiment directory")
    parser.add_argument("run", type=int, choices=range(1, 4), 
                        help="Run number (1-3)")
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    
    # Define paths to JSON files
    default_json_path = base_dir / args.experiment / "centralized" / f"run{args.run}" / "Default_training.json"
    mobilepipe_json_path = base_dir / args.experiment / "decentralized" / f"run{args.run}" / "MobilePipe_training.json"
    
    # Extract runtimes
    default_runtime = get_total_runtime_from_json(default_json_path)
    mobilepipe_runtime = get_total_runtime_from_json(mobilepipe_json_path)
    
    # Print results
    print(f"Experiment: {args.experiment}, Run: {args.run}")
    print("-" * 50)
    
    if default_runtime is not None:
        print(f"Default Training Runtime: {default_runtime:.4f} seconds")
    else:
        print(f"Default Training Runtime: Not found or unavailable")
    
    if mobilepipe_runtime is not None:
        print(f"MobilePipe Training Runtime: {mobilepipe_runtime:.4f} seconds")
    else:
        print(f"MobilePipe Training Runtime: Not found or unavailable")


if __name__ == "__main__":
    main()

