"""Leakage-safe threshold calibration and structured decoding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .contract import (
    CATEGORY_FIELDS,
    AttributeVocabulary,
)


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": _f1(tp, fp, fn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": tp + fn,
        "predicted": tp + fp,
    }


def _is_supervised(row: Mapping[str, Any], field: str) -> bool:
    fields = row.get("supervised_fields")
    return fields is None or field in fields


def _predicted_indices(
    probabilities: Sequence[float],
    *,
    applicable_probability: float,
    applicability_threshold: float,
    value_threshold: float | Sequence[float],
    multi_label: bool,
    max_values: int | None = None,
) -> set[int]:
    if applicable_probability < applicability_threshold or not probabilities:
        return set()
    if not multi_label:
        return {max(range(len(probabilities)), key=probabilities.__getitem__)}
    value_thresholds = (
        [float(value_threshold)] * len(probabilities)
        if isinstance(value_threshold, int | float)
        else [float(value) for value in value_threshold]
    )
    if len(value_thresholds) != len(probabilities):
        raise ValueError(
            "Multi-label value thresholds must match the probability vector"
        )
    predicted = {
        index
        for index, probability in enumerate(probabilities)
        if probability >= value_thresholds[index]
    }
    if max_values is not None and len(predicted) > max_values:
        predicted = set(
            sorted(predicted, key=lambda index: probabilities[index], reverse=True)[
                :max_values
            ]
        )
    return predicted


def calibrate_thresholds(
    rows: Sequence[Mapping[str, Any]],
    vocabulary: AttributeVocabulary,
    *,
    grid: Sequence[float] = tuple(index / 20 for index in range(1, 20)),
    multilabel_strategy: str = "per_value",
    max_values_by_field: Mapping[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    """Maximize exact value micro-F1 independently per conditional field."""
    multi_fields = set(vocabulary.multi_label_fields)
    thresholds: dict[str, dict[str, Any]] = {}
    for field in vocabulary.fields:
        if field in CATEGORY_FIELDS:
            continue
        field_rows = [row for row in rows if _is_supervised(row, field)]
        if not field_rows:
            thresholds[field] = {
                "applicability": 0.5,
                "value": 0.5,
                "calibration_f1": 0.0,
                "supervised_rows": 0,
            }
            continue
        if field in multi_fields and multilabel_strategy == "per_value":
            class_count = len(vocabulary.values[field])
            applicability = torch.tensor(
                [row["applicability_probabilities"][field] for row in field_rows],
                dtype=torch.float32,
            )
            value_probabilities = torch.tensor(
                [row["value_probabilities"][field] for row in field_rows],
                dtype=torch.float32,
            )
            gold = torch.zeros((len(field_rows), class_count), dtype=torch.bool)
            for row_index, row in enumerate(field_rows):
                indices = row["gold_value_indices"].get(field, [])
                if indices:
                    gold[row_index, indices] = True
            best_multi: tuple[float, float, tuple[float, ...]] = (
                -1.0,
                0.5,
                tuple(0.5 for _ in range(class_count)),
            )
            for applicability_threshold in grid:
                app_mask = applicability >= applicability_threshold
                best_scores = torch.full((class_count,), -1.0)
                chosen_thresholds = torch.full((class_count,), 0.5)
                chosen_tp = torch.zeros(class_count, dtype=torch.long)
                chosen_fp = torch.zeros(class_count, dtype=torch.long)
                chosen_fn = torch.zeros(class_count, dtype=torch.long)
                for value_threshold in grid:
                    predicted = app_mask[:, None] & (
                        value_probabilities >= value_threshold
                    )
                    tp = (predicted & gold).sum(dim=0)
                    fp = (predicted & ~gold).sum(dim=0)
                    fn = (~predicted & gold).sum(dim=0)
                    denominator = 2 * tp + fp + fn
                    scores = torch.where(
                        denominator > 0,
                        (2 * tp).float() / denominator.float(),
                        torch.zeros_like(denominator, dtype=torch.float32),
                    )
                    better = scores >= best_scores
                    best_scores[better] = scores[better]
                    chosen_thresholds[better] = value_threshold
                    chosen_tp[better] = tp[better]
                    chosen_fp[better] = fp[better]
                    chosen_fn[better] = fn[better]
                totals = (
                    int(chosen_tp.sum()),
                    int(chosen_fp.sum()),
                    int(chosen_fn.sum()),
                )
                candidate_multi = (
                    _f1(totals[0], totals[1], totals[2]),
                    applicability_threshold,
                    tuple(round(float(value), 6) for value in chosen_thresholds),
                )
                if candidate_multi > best_multi:
                    best_multi = candidate_multi
            thresholds[field] = {
                "applicability": best_multi[1],
                "value": 0.5,
                "value_by_index": list(best_multi[2]),
                "calibration_f1": best_multi[0],
                "supervised_rows": len(field_rows),
            }
            continue
        if field in multi_fields:
            if multilabel_strategy != "global":
                raise ValueError(
                    "multilabel_strategy must be either 'per_value' or 'global'"
                )
            max_values = (max_values_by_field or {}).get(field)
            best_global = (-1.0, 0.5, 0.5)
            for applicability_threshold in grid:
                for value_threshold in grid:
                    tp = fp = fn = 0
                    for row in field_rows:
                        predicted = _predicted_indices(
                            row["value_probabilities"][field],
                            applicable_probability=row["applicability_probabilities"][field],
                            applicability_threshold=applicability_threshold,
                            value_threshold=value_threshold,
                            multi_label=True,
                            max_values=max_values,
                        )
                        gold = set(row["gold_value_indices"].get(field, []))
                        tp += len(predicted & gold)
                        fp += len(predicted - gold)
                        fn += len(gold - predicted)
                    candidate = (
                        _f1(tp, fp, fn),
                        applicability_threshold,
                        value_threshold,
                    )
                    if candidate > best_global:
                        best_global = candidate
            thresholds[field] = {
                "applicability": best_global[1],
                "value": best_global[2],
                "calibration_f1": best_global[0],
                "multilabel_strategy": "global",
                "max_values": max_values,
                "supervised_rows": len(field_rows),
            }
            continue
        best = (-1.0, 0.5, 0.5)
        value_grid = (0.0,)
        for applicability_threshold in grid:
            for value_threshold in value_grid:
                tp = fp = fn = 0
                for row in field_rows:
                    predicted = _predicted_indices(
                        row["value_probabilities"][field],
                        applicable_probability=row["applicability_probabilities"][field],
                        applicability_threshold=applicability_threshold,
                        value_threshold=value_threshold,
                        multi_label=field in multi_fields,
                    )
                    gold = set(row["gold_value_indices"].get(field, []))
                    tp += len(predicted & gold)
                    fp += len(predicted - gold)
                    fn += len(gold - predicted)
                score = _f1(tp, fp, fn)
                candidate = (score, applicability_threshold, value_threshold)
                if candidate > best:
                    best = candidate
        thresholds[field] = {
            "applicability": best[1],
            "value": best[2],
            "calibration_f1": best[0],
            "supervised_rows": len(field_rows),
        }
    return thresholds


def decode_probabilities(
    category_probabilities: Mapping[str, Sequence[float]],
    applicability_probabilities: Mapping[str, float],
    value_probabilities: Mapping[str, Sequence[float]],
    vocabulary: AttributeVocabulary,
    thresholds: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Decode calibrated field predictions into sparse MODA attributes."""
    attributes: dict[str, Any] = {}
    for field in CATEGORY_FIELDS:
        probabilities = category_probabilities.get(field, [])
        if probabilities:
            index = max(range(len(probabilities)), key=probabilities.__getitem__)
            attributes[field] = vocabulary.values[field][index]
    multi_fields = set(vocabulary.multi_label_fields)
    for field in vocabulary.fields:
        if field in CATEGORY_FIELDS:
            continue
        field_thresholds = thresholds.get(field, {})
        indices = _predicted_indices(
            value_probabilities.get(field, []),
            applicable_probability=applicability_probabilities.get(field, 0.0),
            applicability_threshold=float(field_thresholds.get("applicability", 0.5)),
            value_threshold=field_thresholds.get(
                "value_by_index", float(field_thresholds.get("value", 0.5))
            ),
            multi_label=field in multi_fields,
            max_values=(
                int(field_thresholds["max_values"])
                if field_thresholds.get("max_values") is not None
                else None
            ),
        )
        values = [vocabulary.values[field][index] for index in sorted(indices)]
        if values:
            attributes[field] = values if field in multi_fields else values[0]
    return attributes


def score_probability_rows(
    rows: Sequence[Mapping[str, Any]],
    vocabulary: AttributeVocabulary,
    thresholds: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Score held-out rows after applying thresholds learned elsewhere."""
    category_correct = {field: 0 for field in CATEGORY_FIELDS}
    category_rows = {field: 0 for field in CATEGORY_FIELDS}
    field_counts = {
        field: {"tp": 0, "fp": 0, "fn": 0}
        for field in vocabulary.fields
        if field not in CATEGORY_FIELDS
    }
    per_value_counts = {
        field: [
            {"tp": 0, "fp": 0, "fn": 0}
            for _value in vocabulary.values[field]
        ]
        for field in field_counts
    }
    applicability_counts = {
        field: {"tp": 0, "fp": 0, "fn": 0}
        for field in field_counts
    }
    oracle_value_counts = {
        field: {"tp": 0, "fp": 0, "fn": 0}
        for field in field_counts
    }
    exact_rows = 0
    exact_evaluated_rows = 0
    for row in rows:
        decoded = decode_probabilities(
            row["category_probabilities"],
            row["applicability_probabilities"],
            row["value_probabilities"],
            vocabulary,
            thresholds,
        )
        exact = True
        evaluated = False
        for field in CATEGORY_FIELDS:
            if not _is_supervised(row, field):
                continue
            probabilities = row["category_probabilities"][field]
            predicted = max(range(len(probabilities)), key=probabilities.__getitem__)
            gold = int(row["gold_category_indices"][field])
            category_correct[field] += int(predicted == gold)
            category_rows[field] += 1
            exact &= predicted == gold
            evaluated = True
        for field, counts in field_counts.items():
            if not _is_supervised(row, field):
                continue
            predicted_values = decoded.get(field, [])
            if isinstance(predicted_values, str):
                predicted_values = [predicted_values]
            value_indices = vocabulary.value_to_index(field)
            predicted = {
                value_indices[value] for value in predicted_values if value in value_indices
            }
            gold = set(row["gold_value_indices"].get(field, []))
            counts["tp"] += len(predicted & gold)
            counts["fp"] += len(predicted - gold)
            counts["fn"] += len(gold - predicted)
            for value_index, value_count in enumerate(per_value_counts[field]):
                is_predicted = value_index in predicted
                is_gold = value_index in gold
                value_count["tp"] += int(is_predicted and is_gold)
                value_count["fp"] += int(is_predicted and not is_gold)
                value_count["fn"] += int(not is_predicted and is_gold)
            threshold = thresholds.get(field, {})
            predicted_applicable = row["applicability_probabilities"][field] >= float(
                threshold.get("applicability", 0.5)
            )
            gold_applicable = bool(gold)
            app_counts = applicability_counts[field]
            app_counts["tp"] += int(predicted_applicable and gold_applicable)
            app_counts["fp"] += int(predicted_applicable and not gold_applicable)
            app_counts["fn"] += int(not predicted_applicable and gold_applicable)
            probabilities = row["value_probabilities"][field]
            oracle_predicted = _predicted_indices(
                probabilities,
                applicable_probability=1.0,
                applicability_threshold=0.0,
                value_threshold=threshold.get(
                    "value_by_index", float(threshold.get("value", 0.5))
                ),
                multi_label=field in set(vocabulary.multi_label_fields),
                max_values=(
                    int(threshold["max_values"])
                    if threshold.get("max_values") is not None
                    else None
                ),
            )
            if gold_applicable:
                value_counts = oracle_value_counts[field]
                value_counts["tp"] += len(oracle_predicted & gold)
                value_counts["fp"] += len(oracle_predicted - gold)
                value_counts["fn"] += len(gold - oracle_predicted)
            exact &= predicted == gold
            evaluated = True
        exact_rows += int(evaluated and exact)
        exact_evaluated_rows += int(evaluated)

    total_tp = sum(item["tp"] for item in field_counts.values())
    total_fp = sum(item["fp"] for item in field_counts.values())
    total_fn = sum(item["fn"] for item in field_counts.values())
    per_field_f1 = {
        field: _f1(counts["tp"], counts["fp"], counts["fn"])
        for field, counts in field_counts.items()
    }
    per_field = {
        field: {
            **_prf(counts["tp"], counts["fp"], counts["fn"]),
            "applicability": _prf(
                applicability_counts[field]["tp"],
                applicability_counts[field]["fp"],
                applicability_counts[field]["fn"],
            ),
            "oracle_applicability_value": _prf(
                oracle_value_counts[field]["tp"],
                oracle_value_counts[field]["fp"],
                oracle_value_counts[field]["fn"],
            ),
        }
        for field, counts in field_counts.items()
    }
    overall = _prf(total_tp, total_fp, total_fn)
    return {
        "rows": len(rows),
        "category_accuracy": {
            field: correct / category_rows[field] if category_rows[field] else 0.0
            for field, correct in category_correct.items()
        },
        "attribute_micro_precision": overall["precision"],
        "attribute_micro_recall": overall["recall"],
        "attribute_micro_f1": overall["f1"],
        "attribute_field_macro_f1": (
            sum(per_field_f1.values()) / len(per_field_f1) if per_field_f1 else 0.0
        ),
        "exact_all_fields": (
            exact_rows / exact_evaluated_rows if exact_evaluated_rows else 0.0
        ),
        "per_field_f1": per_field_f1,
        "per_field": per_field,
        "per_value": {
            field: {
                value: _prf(
                    per_value_counts[field][index]["tp"],
                    per_value_counts[field][index]["fp"],
                    per_value_counts[field][index]["fn"],
                )
                for index, value in enumerate(vocabulary.values[field])
            }
            for field in field_counts
        },
        "counts": {"tp": total_tp, "fp": total_fp, "fn": total_fn},
    }
