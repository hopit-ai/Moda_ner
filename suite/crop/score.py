#!/usr/bin/env python3
"""Create community-facing metrics for the Fashionpedia -> MODA benchmark."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

from suite._eval import normalize_value
from suite._eval import evaluate_image_attributes
from suite._eval import attributes_from_entry

CATEGORY_FIELDS = ("master_category", "category")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def _row_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("record_id") or row.get("id") or f"row:{index}")


def _values(field: str, value: Any) -> set[str]:
    if isinstance(value, str):
        raw = [value]
    elif isinstance(value, list):
        raw = [item for item in value if isinstance(item, str)]
    else:
        raw = []
    return {normalize_value(field, item) for item in raw if item.strip()}


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _point_metrics(group_stats: list[dict[str, Any]]) -> dict[str, float]:
    category_correct = sum(item["category_correct"] for item in group_stats)
    master_correct = sum(item["master_correct"] for item in group_stats)
    rows = sum(item["rows"] for item in group_stats)
    tp = sum(item["tp"] for item in group_stats)
    fp = sum(item["fp"] for item in group_stats)
    fn = sum(item["fn"] for item in group_stats)
    field_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for item in group_stats:
        for field, counts in item["fields"].items():
            for index, count in enumerate(counts):
                field_totals[field][index] += count
    supported_field_f1 = [
        _prf(counts[0], counts[1], counts[2])["f1"]
        for counts in field_totals.values()
        if counts[3] > 0
    ]
    micro = _prf(tp, fp, fn)
    return {
        "category_accuracy": category_correct / rows if rows else 0.0,
        "master_category_accuracy": master_correct / rows if rows else 0.0,
        "attribute_micro_precision": micro["precision"],
        "attribute_micro_recall": micro["recall"],
        "attribute_micro_f1": micro["f1"],
        "attribute_field_macro_f1": (
            sum(supported_field_f1) / len(supported_field_f1) if supported_field_f1 else 0.0
        ),
    }


def _build_group_stats(
    gold_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Aggregate row-level outcomes by source image for clustered inference."""
    pred_by_id = {_row_id(row, index): row for index, row in enumerate(predictions)}
    groups: dict[str, dict[str, Any]] = {}
    attribute_fields = sorted(
        {
            str(field)
            for row in gold_rows
            for field in row.get("evaluated_attributes", [])
            if field not in CATEGORY_FIELDS
        }
    )
    for index, gold in enumerate(gold_rows):
        prediction = pred_by_id.get(_row_id(gold, index), {})
        gold_attrs = attributes_from_entry(gold)
        pred_attrs = prediction.get("attributes") or {}
        image = gold.get("image") or {}
        group_id = str(image.get("image_group_id") or _row_id(gold, index))
        stats = groups.setdefault(
            group_id,
            {
                "rows": 0,
                "category_correct": 0,
                "master_correct": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "fields": defaultdict(lambda: [0, 0, 0, 0]),
            },
        )
        stats["rows"] += 1
        for field, correct_key in (
            ("category", "category_correct"),
            ("master_category", "master_correct"),
        ):
            stats[correct_key] += int(
                _values(field, gold_attrs.get(field)) == _values(field, pred_attrs.get(field))
            )
        for field in attribute_fields:
            gold_values = _values(field, gold_attrs.get(field))
            pred_values = _values(field, pred_attrs.get(field))
            tp = len(gold_values & pred_values)
            fp = len(pred_values - gold_values)
            fn = len(gold_values - pred_values)
            stats["tp"] += tp
            stats["fp"] += fp
            stats["fn"] += fn
            counts = stats["fields"][field]
            counts[0] += tp
            counts[1] += fp
            counts[2] += fn
            counts[3] += len(gold_values)
    return groups, attribute_fields


def summarize(
    gold_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    groups, attribute_fields = _build_group_stats(gold_rows, predictions)
    group_stats = list(groups.values())
    point = _point_metrics(group_stats)
    intervals: dict[str, dict[str, float]] = {}
    if bootstrap_samples > 0 and group_stats:
        rng = random.Random(seed)
        distributions = {key: [] for key in point}
        for _ in range(bootstrap_samples):
            sample = [group_stats[rng.randrange(len(group_stats))] for _ in group_stats]
            metrics = _point_metrics(sample)
            for key, value in metrics.items():
                distributions[key].append(value)
        intervals = {
            key: {
                "low": round(_percentile(values, 0.025), 4),
                "high": round(_percentile(values, 0.975), 4),
            }
            for key, values in distributions.items()
        }

    supervision = evaluate_image_attributes(predictions, gold_rows)
    return {
        "protocol": "Fashionpedia 2020 val; official boxes; MODA mapping; oracle crops",
        "rows": len(gold_rows),
        "unique_image_groups": len(groups),
        "attribute_fields": attribute_fields,
        "point_metrics": {key: round(value, 4) for key, value in point.items()},
        "cluster_bootstrap_95ci": intervals,
        "bootstrap_samples": bootstrap_samples,
        "structure": {
            "raw_structure_validity": supervision["raw_structure_validity"],
            "schema_compliance": supervision["schema_compliance"],
            "exact_match_on_all_mapped_fields": supervision[
                "exact_match_on_judged_fields"
            ],
        },
        "per_field": supervision["per_attribute"],
    }


def compare(
    gold_rows: list[dict[str, Any]],
    baseline_predictions: list[dict[str, Any]],
    candidate_predictions: list[dict[str, Any]],
    *,
    baseline_name: str = "baseline",
    candidate_name: str = "candidate",
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare two checkpoints with an image-clustered paired bootstrap."""
    baseline_groups, baseline_fields = _build_group_stats(
        gold_rows, baseline_predictions
    )
    candidate_groups, candidate_fields = _build_group_stats(
        gold_rows, candidate_predictions
    )
    if baseline_fields != candidate_fields:
        raise ValueError("Baseline and candidate fields do not match")
    if baseline_groups.keys() != candidate_groups.keys():
        raise ValueError("Baseline and candidate image groups do not match")

    group_ids = sorted(baseline_groups)
    baseline_point = _point_metrics([baseline_groups[key] for key in group_ids])
    candidate_point = _point_metrics([candidate_groups[key] for key in group_ids])
    point_delta = {
        key: candidate_point[key] - baseline_point[key] for key in baseline_point
    }
    delta_distributions = {key: [] for key in point_delta}
    if bootstrap_samples > 0 and group_ids:
        rng = random.Random(seed)
        for _ in range(bootstrap_samples):
            sample_ids = [group_ids[rng.randrange(len(group_ids))] for _ in group_ids]
            baseline_metrics = _point_metrics(
                [baseline_groups[key] for key in sample_ids]
            )
            candidate_metrics = _point_metrics(
                [candidate_groups[key] for key in sample_ids]
            )
            for key in point_delta:
                delta_distributions[key].append(
                    candidate_metrics[key] - baseline_metrics[key]
                )

    delta_intervals = {
        key: {
            "low": round(_percentile(values, 0.025), 4),
            "high": round(_percentile(values, 0.975), 4),
        }
        for key, values in delta_distributions.items()
        if values
    }
    probability_candidate_better = {
        key: round(sum(value > 0 for value in values) / len(values), 4)
        for key, values in delta_distributions.items()
        if values
    }

    baseline_eval = evaluate_image_attributes(baseline_predictions, gold_rows)
    candidate_eval = evaluate_image_attributes(candidate_predictions, gold_rows)
    per_field_delta = {
        field: round(
            candidate_eval["per_attribute"][field]["value_f1"]
            - baseline_eval["per_attribute"][field]["value_f1"],
            4,
        )
        for field in sorted(baseline_eval["per_attribute"])
    }
    return {
        "protocol": "Fashionpedia 2020 val; official boxes; MODA mapping; oracle crops",
        "rows": len(gold_rows),
        "unique_image_groups": len(group_ids),
        "baseline_name": baseline_name,
        "candidate_name": candidate_name,
        "baseline_point_metrics": {
            key: round(value, 4) for key, value in baseline_point.items()
        },
        "candidate_point_metrics": {
            key: round(value, 4) for key, value in candidate_point.items()
        },
        "candidate_minus_baseline": {
            key: round(value, 4) for key, value in point_delta.items()
        },
        "paired_cluster_bootstrap_delta_95ci": delta_intervals,
        "bootstrap_probability_candidate_better": probability_candidate_better,
        "bootstrap_samples": bootstrap_samples,
        "per_field_value_f1_delta": per_field_delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path)
    parser.add_argument("--comparison-output", type=Path)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = summarize(
        _load_jsonl(args.gold),
        _load_jsonl(args.predictions),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "per_field"}, indent=2))
    if args.baseline_predictions:
        comparison_output = args.comparison_output or args.output.with_name(
            "comparison_metrics.json"
        )
        comparison = compare(
            _load_jsonl(args.gold),
            _load_jsonl(args.baseline_predictions),
            _load_jsonl(args.predictions),
            baseline_name=args.baseline_name,
            candidate_name=args.candidate_name,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        comparison_output.parent.mkdir(parents=True, exist_ok=True)
        comparison_output.write_text(json.dumps(comparison, indent=2) + "\n")
        print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
