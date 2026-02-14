#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches


COLOR_MAP = {
    "HOST_S1_FORWARD_MICROBATCH": "#4C72B0",
    "HOST_S1_BACKWARD_MICROBATCH": "#DD8452",
    "HOST_WAITING_FOR_GRAD": "#C44E52",
    "HOST_OPTIMIZATION_STEP": "#8172B2",
    "IOS_S2_COMBINED_MICROBATCH": "#55A868",
    "IOS_WAITING_FOR_INPUT": "#CCB974",
}
IGNORE = {
    'HOST_FULL_BATCH',
    'HOST_INTERMEDIATE_CONFIG_SYNC',
    'HOST_INIT_MODEL_OFFLOAD',
}


def load_trace(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def collect_intervals(trace: dict, prefix: str):
    """
    Collect all (start, duration, label) tuples for keys starting with prefix.
    """
    intervals = []
    for key, spans in trace.items():
        if not key.startswith(prefix):
            continue
        for start, end in spans:
            intervals.append((start, end - start, key))
    return intervals


def plot_gantt(host_intervals, ios_intervals):
    fig, ax = plt.subplots(figsize=(14, 4))

    # Normalize time to first event
    all_starts = [s for s, _, _ in host_intervals + ios_intervals]
    t0 = min(all_starts)

    def draw_row(intervals, y, height):
        for start, dur, label in intervals:
            if label in IGNORE:
                continue
            ax.barh(
                y,
                dur,
                left=start - t0,
                height=height,
                color=COLOR_MAP.get(label, "gray"),
                edgecolor="black",
            )
            print(label)
            handles = [
                mpatches.Patch(color=color, label=label)
                for label, color in COLOR_MAP.items()
            ]

            ax.legend(
                handles=handles,
                title="Event type",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
            )

    # Two pipeline stages
    draw_row(host_intervals, y=0.5, height=0.4)
    draw_row(ios_intervals, y=0, height=0.4)

    ax.set_yticks([0, 0.5, 1.3])
    ax.set_yticklabels(["Stage 2: IOS", "Stage 1: HOST", " "])
    ax.set_xlabel("Time since start (seconds)")
    ax.set_title("Pipeline Parallel Execution Gantt Chart")

    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot pipeline Gantt chart")
    # Add mutually exclusive groups for arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--json-file", "-j", type=Path, help="Direct path to JSON trace file")
    group.add_argument("--experiment", type=str, help="Experiment name (requires --run and --config)")

    parser.add_argument("--run", type=int, choices=[1, 2, 3], help="Run number (requires --experiment and --config)")
    parser.add_argument("--config", type=int, choices=[1, 2, 3], help="Config number (requires --experiment and --run)")

    args = parser.parse_args()

    # Check if using experiment/run/config combination
    if args.experiment:
        if not args.run or not args.config:
            parser.error("--experiment requires both --run and --config")

        json_path = (
            Path(args.experiment)
            / "decentralized"
            / f"run{args.run}"
            / f"MobilePipe_C{args.config}.json"
        )
    else:
        # Using direct JSON file path
        json_path = args.json_file

    if not json_path.exists():
        raise FileNotFoundError(f"Missing file: {json_path}")

    trace = load_trace(json_path)

    host_intervals = collect_intervals(trace, "HOST_")
    ios_intervals = collect_intervals(trace, "IOS_")

    if not host_intervals or not ios_intervals:
        raise ValueError("Expected both HOST_ and IOS_ events in trace")

    plot_gantt(host_intervals, ios_intervals)


if __name__ == "__main__":
    main()

