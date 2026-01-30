#!/usr/bin/env python3
"""
Walk through results/ and aggregate all eval data into a single CSV.
"""

import os
import json
import csv
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_CSV = Path(__file__).parent / "aggregated_results.csv"


def main():
    rows = []

    # Walk through each model directory
    for model_dir in sorted(RESULTS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue

        # Walk through each system prompt subdirectory
        for prompt_dir in sorted(model_dir.iterdir()):
            if not prompt_dir.is_dir():
                continue

            summary_path = prompt_dir / "summary.json"
            if not summary_path.exists():
                continue

            try:
                with open(summary_path, "r") as f:
                    data = json.load(f)

                row = {
                    "model": data["model"]["name"],
                    "system_prompt": data["system_prompt"]["key"],
                    "medical_avg_aligned": data["results"]["medical"]["avg_aligned"],
                    "medical_avg_coherent": data["results"]["medical"]["avg_coherent"],
                    "financial_avg_aligned": data["results"]["financial"]["avg_aligned"],
                    "financial_avg_coherent": data["results"]["financial"]["avg_coherent"],
                }
                rows.append(row)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not process {summary_path}: {e}")
                continue

    # Write to CSV
    fieldnames = [
        "model",
        "system_prompt",
        "medical_avg_aligned",
        "medical_avg_coherent",
        "financial_avg_aligned",
        "financial_avg_coherent",
    ]

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Aggregated {len(rows)} results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
