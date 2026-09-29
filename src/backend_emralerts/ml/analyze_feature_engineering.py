#!/usr/bin/env python3
"""Profile a CSV dataset and produce a model-oriented feature-engineering plan."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from statistics import median
from typing import Any


ID_PATTERN = re.compile(r"(^|_)(id|identifier|key)($|_)")
DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y")


def parse_number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def parse_date(value: str) -> date | None:
    for format_string in DATE_FORMATS:
        try:
            return datetime.strptime(value, format_string).date()
        except ValueError:
            continue
    return None


def quantile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def infer_type(values: list[str]) -> str:
    populated = [value for value in values if value.strip()]
    if not populated:
        return "empty"
    if all(parse_date(value) is not None for value in populated):
        return "date"
    if all(parse_number(value) is not None for value in populated):
        return "numeric"
    return "categorical"


def analyze_column(name: str, values: list[str], row_count: int) -> dict[str, Any]:
    populated = [value.strip() for value in values if value.strip()]
    missing_count = row_count - len(populated)
    column_type = infer_type(values)
    unique_count = len(set(populated))
    profile: dict[str, Any] = {
        "name": name,
        "detected_type": column_type,
        "missing_count": missing_count,
        "missing_rate": round(missing_count / row_count, 4) if row_count else 0,
        "unique_count": unique_count,
        "unique_rate": round(unique_count / len(populated), 4) if populated else 0,
    }

    if ID_PATTERN.search(name.lower()):
        profile["recommendation"] = {
            "include_in_model": False,
            "reason": "Identifier values can cause memorization and do not generalize to unseen patients or physicians.",
            "optional_derived_features": ["physician encounter volume", "historical patient aggregates"],
            "note": "Only derive these from training folds to prevent target leakage.",
        }
        return profile

    if column_type == "date":
        dates = [parse_date(value) for value in populated]
        valid_dates = [item for item in dates if item is not None]
        profile["date_range"] = {
            "minimum": min(valid_dates).isoformat(),
            "maximum": max(valid_dates).isoformat(),
        }
        profile["recommendation"] = {
            "include_in_model": True,
            "transformations": [
                "Extract year, month, quarter, day of week, and day of year.",
                "Encode cyclical calendar features with sine and cosine for month and day of week.",
                "Create recency only when a prediction cutoff date is defined.",
            ],
        }
    elif column_type == "numeric":
        numbers = [parse_number(value) for value in populated]
        valid_numbers = [item for item in numbers if item is not None]
        lower_quartile = quantile(valid_numbers, 0.25)
        upper_quartile = quantile(valid_numbers, 0.75)
        interquartile_range = upper_quartile - lower_quartile
        lower_fence = lower_quartile - 1.5 * interquartile_range
        upper_fence = upper_quartile + 1.5 * interquartile_range
        profile["statistics"] = {
            "minimum": min(valid_numbers),
            "median": median(valid_numbers),
            "maximum": max(valid_numbers),
            "outliers_iqr": sum(number < lower_fence or number > upper_fence for number in valid_numbers),
        }
        profile["recommendation"] = {
            "include_in_model": True,
            "transformations": [
                "Impute missing values with the training-set median and add a missing-value indicator.",
                "Use robust scaling for linear or distance-based models.",
                "Consider log1p transformation for non-negative, right-skewed count features.",
                "Winsorize extreme values only after validating the effect within cross-validation.",
            ],
        }
    else:
        counts = Counter(populated)
        top_values = [{"value": value, "count": count} for value, count in counts.most_common(5)]
        profile["top_values"] = top_values
        encoding = (
            "One-hot encode."
            if unique_count <= 20
            else "Group infrequent categories into Other, then one-hot encode; use fold-safe target encoding only if needed."
        )
        profile["recommendation"] = {
            "include_in_model": True,
            "transformations": [
                "Impute missing values with an explicit Unknown category.",
                encoding,
                "Normalize case and surrounding whitespace before encoding.",
            ],
        }
    return profile


def analyze_dataset(input_path: Path, target_column: str) -> dict[str, Any]:
    with input_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError("The CSV file must include a header row.")
        fieldnames = reader.fieldnames
        if target_column not in fieldnames:
            raise ValueError(f"Target column '{target_column}' was not found. Available columns: {', '.join(fieldnames)}")
        rows = list(reader)

    columns = {name: [row.get(name, "") for row in rows] for name in fieldnames}
    target_values = [value.strip() for value in columns[target_column] if value.strip()]
    target_counts = Counter(target_values)
    if not target_counts:
        raise ValueError(f"Target column '{target_column}' contains no values.")

    feature_profiles = [
        analyze_column(name, values, len(rows))
        for name, values in columns.items()
        if name != target_column
    ]
    included_features = [
        profile["name"]
        for profile in feature_profiles
        if profile["recommendation"]["include_in_model"]
    ]
    excluded_features = [
        profile["name"]
        for profile in feature_profiles
        if not profile["recommendation"]["include_in_model"]
    ]

    return {
        "input_file": str(input_path),
        "row_count": len(rows),
        "target": {
            "name": target_column,
            "classes": dict(sorted(target_counts.items())),
            "class_balance": {
                value: round(count / len(target_values), 4)
                for value, count in sorted(target_counts.items())
            },
            "recommendation": "Use stratified cross-validation and report precision, recall, PR-AUC, and ROC-AUC for this binary target.",
        },
        "model_feature_summary": {
            "recommended_raw_features": included_features,
            "excluded_raw_features": excluded_features,
            "pipeline_guidance": [
                "Fit imputers, scalers, encoders, and any identifier-derived aggregates on each training fold only.",
                "Use a ColumnTransformer or equivalent preprocessing pipeline so transformations are reproduced during inference.",
                "Evaluate treatment effectiveness predictions for calibration in addition to discrimination.",
            ],
        },
        "columns": feature_profiles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a feature-engineering report for a tabular machine-learning CSV dataset."
    )
    parser.add_argument("input_csv", type=Path, help="Path to the input CSV file.")
    parser.add_argument(
        "--target",
        default="TARGET",
        help="Name of the target column (default: TARGET).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("feature_engineering_report.json"),
        help="Path for the JSON report (default: feature_engineering_report.json).",
    )
    args = parser.parse_args()

    report = analyze_dataset(args.input_csv, args.target)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Feature-engineering report written to {args.output}")
    print(f"Rows analyzed: {report['row_count']}")
    print(f"Recommended raw features: {len(report['model_feature_summary']['recommended_raw_features'])}")
    print(f"Excluded identifiers: {', '.join(report['model_feature_summary']['excluded_raw_features']) or 'none'}")


if __name__ == "__main__":
    main()
