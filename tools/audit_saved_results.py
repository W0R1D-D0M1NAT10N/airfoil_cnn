#!/usr/bin/env python3
"""Audit committed CSV evidence without loading models or running simulations.

Run from any working directory: python /path/to/repo/tools/audit_saved_results.py
The affine fit below is deliberately labeled IN-SAMPLE, not held-out validation.
Only Python's standard library is required. No repository files are modified.
"""

import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]


def read_csv(relative_path):
    with (ROOT / relative_path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def metrics(truth, predictions):
    if len(truth) != len(predictions) or not truth:
        raise ValueError("Metrics need equally sized, nonempty paired values")
    if not all(math.isfinite(value) for value in truth + predictions):
        raise ValueError("Metrics need finite numeric values")
    errors = [actual - predicted for actual, predicted in zip(truth, predictions)]
    mse = mean(error * error for error in errors)
    truth_mean = mean(truth)
    total_variation = sum((value - truth_mean) ** 2 for value in truth)
    if total_variation == 0:
        raise ValueError("R-squared is undefined for constant truth values")
    return {
        "n": len(truth),
        "mae": mean(abs(error) for error in errors),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "r_squared": 1 - sum(error * error for error in errors) / total_variation,
    }


def main():
    labels = read_csv("airfoil_data_clean.csv")
    predictions = read_csv("airfoil_predictions_vs_true.csv")
    flight_rows = read_csv("3DModel/flight_data.csv")
    truth = [float(row["true_values"]) for row in predictions]
    raw = [float(row["pred_values"]) for row in predictions]
    raw_metrics = metrics(truth, raw)

    # Least-squares correction fitted AND scored on the same saved rows.
    # This diagnoses bias; it is not an independently validated calibration.
    raw_mean, truth_mean = mean(raw), mean(truth)
    raw_variation = sum((value - raw_mean) ** 2 for value in raw)
    if raw_variation == 0:
        raise ValueError("Cannot fit an affine correction to constant predictions")
    slope = sum(
        (predicted - raw_mean) * (actual - truth_mean)
        for actual, predicted in zip(truth, raw)
    ) / raw_variation
    intercept = truth_mean - slope * raw_mean
    adjusted = [slope * value + intercept for value in raw]

    # Keep sets of labels so conflicting duplicate source rows remain visible.
    source_labels = {}
    for row in labels:
        key = (row["file_path"], float(row["aoa"]))
        source_labels.setdefault(key, set()).add(float(row["cl"]))
    unmatched_predictions = sum(
        float(row["true_values"]) not in source_labels.get(
            (row["airfoil_name"], float(row["AoA"])), set()
        )
        for row in predictions
    )

    mesh_names = sorted({row["data"] for row in flight_rows})
    mesh_counts = Counter(row["data"] for row in flight_rows)
    mesh_summary = {}
    for name in mesh_names:
        angles = [float(row["aoa"]) for row in flight_rows if row["data"] == name]
        mesh_summary[name] = {
            "rows": mesh_counts[name],
            "aoa_min_degrees": min(angles),
            "aoa_max_degrees": max(angles),
            "mesh_present": (ROOT / "3DModel/data" / name).is_file(),
        }

    expected_artifacts = [
        "airfoil_cnn.pth",
        "images",
        "train_image_paths.csv",
        "val_image_paths.csv",
        "aoa_scaler.joblib",
        "3DModel/saved_models/model_weights.pth",
        "3DModel/lift_predictions.csv",
    ]
    report = {
        "scope": "Saved CSV audit only; no training, inference, or solver execution",
        "coordinate_dat_files": len(list((ROOT / "airfoils").glob("*.dat"))),
        "clean_2d_data": {
            "rows": len(labels),
            "profiles": len({row["file_path"] for row in labels}),
            "conflicting_profile_angle_labels": sum(
                len(values) > 1 for values in source_labels.values()
            ),
        },
        "saved_2d_predictions": {
            "profiles": len({row["airfoil_name"] for row in predictions}),
            "unmatched_reference_labels": unmatched_predictions,
            "raw_metrics": raw_metrics,
        },
        "IN_SAMPLE_affine_diagnostic_NOT_held_out": {
            "note": "Fit and score use the same rows; independent split unavailable",
            "slope": slope,
            "intercept": intercept,
            "metrics": metrics(truth, adjusted),
        },
        "saved_3d_reference_data": {
            "rows": len(flight_rows),
            "meshes": mesh_summary,
            "rows_with_aoa_0_through_12": sum(
                0 <= float(row["aoa"]) <= 12 for row in flight_rows
            ),
            "note": "Reference data only; no 3D model-performance metrics computed",
        },
        "artifact_presence_at_expected_paths": {
            name: (ROOT / name).exists() for name in expected_artifacts
        },
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
