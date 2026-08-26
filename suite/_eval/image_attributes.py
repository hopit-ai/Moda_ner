"""Evaluation for sparse v3 fashion-image attribute predictions.

The public Fashionpedia validation rows contain positive annotations only. Their
``supervised_attributes`` mask therefore prevents unlabelled fields from being
counted as false positives. A production human benchmark should instead include
``evaluated_attributes`` for every field a reviewer judged, including fields
that were judged absent; that enables true applicability precision and recall.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from .conditional import normalize_value
from .schema import (
    MULTI_LABEL_ATTRIBUTES,
    SINGLE_LABEL_ATTRIBUTES,
)
from .serialization import (
    INDEXED_JUDGMENT_END_MARKER,
    INDEXED_JUDGMENT_START_MARKER,
    JUDGMENT_END_MARKER,
    JUDGMENT_START_MARKER,
    parse_attribute_judgment_sequence,
    parse_attribute_sequence,
)
from .entries import attributes_from_entry

ALL_ATTRIBUTES = frozenset(SINGLE_LABEL_ATTRIBUTES + MULTI_LABEL_ATTRIBUTES)
MULTI_ATTRIBUTES = frozenset(MULTI_LABEL_ATTRIBUTES)

BUSINESS_CRITICAL_ATTRIBUTES = frozenset(
    {
        "master_category",
        "category",
        "sub_category",
        "color_palette_primary",
        "material",
        "pattern",
        "fit",
        "silhouette",
        "sleeve_length",
        "neckline",
        "collar_presence",
        "closure_type",
    }
)


def _row_id(row: dict[str, Any], fallback_index: int) -> str:
    return str(row.get("record_id") or row.get("id") or f"row:{fallback_index}")


def _prediction_attributes(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("attributes", row.get("attributes_json", {}))
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return raw if isinstance(raw, dict) else {}


def _values(field: str, value: Any) -> set[str]:
    raw_values: list[str] = []
    if isinstance(value, str):
        raw_values = [value]
    elif isinstance(value, list):
        raw_values = [item for item in value if isinstance(item, str)]
    elif isinstance(value, dict) and isinstance(value.get("value"), str):
        # Legacy compatibility. Schema compliance still rejects confidence wrappers.
        raw_values = [value["value"]]
    return {normalize_value(field, item) for item in raw_values if item.strip()}


def is_schema_compliant(attributes: dict[str, Any]) -> bool:
    """Validate the sparse, confidence-free v3 output contract."""
    if not isinstance(attributes, dict) or not attributes:
        return False
    if set(attributes) - ALL_ATTRIBUTES:
        return False
    for field, value in attributes.items():
        if field in MULTI_ATTRIBUTES:
            if not isinstance(value, list) or not value:
                return False
            if any(not isinstance(item, str) or not item.strip() for item in value):
                return False
            normalized = [item.strip().lower() for item in value]
            if len(normalized) != len(set(normalized)):
                return False
            if any(item == "unknown" for item in normalized):
                return False
        else:
            if not isinstance(value, str) or not value.strip():
                return False
            if value.strip().lower() == "unknown":
                return False
    if "collar_style" in attributes and attributes.get("collar_presence") != "present":
        return False
    return True


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate_image_attributes(
    predictions: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score aligned predictions with per-row supervision masks."""
    pred_by_id = {_row_id(row, index): row for index, row in enumerate(predictions)}
    value_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0, "judged": 0}
    )
    app_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    state_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "judged": 0})

    aligned = 0
    exact_rows = 0
    raw_json_valid = 0
    raw_structure_valid = 0
    raw_json_available = 0
    judgment_raw_available = 0
    complete_judgment_raw = 0
    schema_valid = 0
    complete_judgement_rows = 0
    state_correct = 0
    state_judged = 0

    for index, gold_row in enumerate(gold_rows):
        record_id = _row_id(gold_row, index)
        pred_row = pred_by_id.get(record_id)
        if pred_row is None and index < len(predictions):
            candidate = predictions[index]
            if not candidate.get("record_id") and not candidate.get("id"):
                pred_row = candidate
        if pred_row is None:
            pred_row = {"attributes": {}}
        else:
            aligned += 1

        pred_attrs = _prediction_attributes(pred_row)
        gold_attrs = attributes_from_entry(gold_row)
        schema_valid += int(is_schema_compliant(pred_attrs))

        raw_output = pred_row.get("raw_output")
        explicit_state_output = False
        if isinstance(raw_output, str):
            raw_json_available += 1
            if JUDGMENT_START_MARKER in raw_output or INDEXED_JUDGMENT_START_MARKER in raw_output:
                explicit_state_output = True
                judgment_raw_available += 1
                _raw_attributes, raw_states = parse_attribute_judgment_sequence(raw_output)
                is_complete_judgment = (
                    JUDGMENT_END_MARKER in raw_output or INDEXED_JUDGMENT_END_MARKER in raw_output
                ) and set(raw_states) == ALL_ATTRIBUTES
                complete_judgment_raw += int(is_complete_judgment)
            try:
                parsed_raw = json.loads(raw_output.strip())
                is_json_object = isinstance(parsed_raw, dict)
                raw_json_valid += int(is_json_object)
                raw_structure_valid += int(is_json_object)
            except json.JSONDecodeError:
                raw_structure_valid += int(bool(parse_attribute_sequence(raw_output)))

        if isinstance(gold_row.get("evaluated_attributes"), list):
            evaluated_fields = {
                str(field) for field in gold_row["evaluated_attributes"] if field in ALL_ATTRIBUTES
            }
            complete_judgement_rows += 1
        elif isinstance(gold_row.get("supervised_attributes"), list):
            evaluated_fields = {
                str(field) for field in gold_row["supervised_attributes"] if field in ALL_ATTRIBUTES
            }
        else:
            evaluated_fields = set(gold_attrs) & ALL_ATTRIBUTES

        row_exact = True
        gold_states = gold_row.get("attribute_states")
        pred_states = pred_row.get("attribute_states")
        if not isinstance(pred_states, dict) and isinstance(raw_output, str):
            _raw_attributes, pred_states = parse_attribute_judgment_sequence(raw_output)
        for field in evaluated_fields:
            pred_values = _values(field, pred_attrs.get(field))
            gold_values = _values(field, gold_attrs.get(field))
            shared = pred_values & gold_values
            stats = value_stats[field]
            stats["tp"] += len(shared)
            stats["fp"] += len(pred_values - gold_values)
            stats["fn"] += len(gold_values - pred_values)
            stats["support"] += len(gold_values)
            stats["judged"] += 1

            pred_has = bool(pred_values)
            gold_has = bool(gold_values)
            if pred_has and gold_has:
                app_stats[field]["tp"] += 1
            elif pred_has:
                app_stats[field]["fp"] += 1
            elif gold_has:
                app_stats[field]["fn"] += 1
            row_exact &= pred_values == gold_values
            if explicit_state_output and isinstance(gold_states, dict) and field in gold_states:
                correct_state = isinstance(pred_states, dict) and pred_states.get(
                    field
                ) == gold_states.get(field)
                state_stats[field]["correct"] += int(correct_state)
                state_stats[field]["judged"] += 1
                state_correct += int(correct_state)
                state_judged += 1
        exact_rows += int(row_exact and bool(evaluated_fields))

    per_attribute: dict[str, dict[str, float | int]] = {}
    for field in sorted(value_stats):
        values = value_stats[field]
        value_precision = _safe_ratio(values["tp"], values["tp"] + values["fp"])
        value_recall = _safe_ratio(values["tp"], values["tp"] + values["fn"])
        app = app_stats[field]
        app_precision = _safe_ratio(app["tp"], app["tp"] + app["fp"])
        app_recall = _safe_ratio(app["tp"], app["tp"] + app["fn"])
        per_attribute[field] = {
            "value_precision": round(value_precision, 4),
            "value_recall": round(value_recall, 4),
            "value_f1": round(_f1(value_precision, value_recall), 4),
            "applicability_precision": round(app_precision, 4),
            "applicability_recall": round(app_recall, 4),
            "applicability_f1": round(_f1(app_precision, app_recall), 4),
            "support": values["support"],
            "judged_rows": values["judged"],
            "state_accuracy": round(
                _safe_ratio(state_stats[field]["correct"], state_stats[field]["judged"]),
                4,
            )
            if state_stats[field]["judged"]
            else None,
        }

    supported_f1 = [
        float(metrics["value_f1"])
        for metrics in per_attribute.values()
        if int(metrics["support"]) > 0
    ]
    critical = {
        field: per_attribute[field]
        for field in sorted(BUSINESS_CRITICAL_ATTRIBUTES)
        if field in per_attribute and int(per_attribute[field]["support"]) > 0
    }
    critical_min_f1 = min(
        (float(metrics["value_f1"]) for metrics in critical.values()),
        default=0.0,
    )
    gold_count = len(gold_rows)
    return {
        "gold_rows": gold_count,
        "prediction_rows": len(predictions),
        "aligned_rows": aligned,
        "supervision_mode": (
            "complete_applicability"
            if complete_judgement_rows == gold_count and gold_count
            else "positive_only_masked"
        ),
        "raw_json_validity": round(_safe_ratio(raw_json_valid, raw_json_available), 4),
        "raw_json_rows": raw_json_available,
        "raw_structure_validity": round(_safe_ratio(raw_structure_valid, raw_json_available), 4),
        "raw_structure_rows": raw_json_available,
        "complete_judgment_structure_validity": round(
            _safe_ratio(complete_judgment_raw, judgment_raw_available), 4
        ),
        "complete_judgment_output_rows": judgment_raw_available,
        "schema_compliance": round(_safe_ratio(schema_valid, gold_count), 4),
        "judgment_state_accuracy": (
            round(_safe_ratio(state_correct, state_judged), 4) if state_judged else None
        ),
        "judgment_state_decisions": state_judged,
        "exact_match_on_judged_fields": round(_safe_ratio(exact_rows, gold_count), 4),
        "macro_value_f1": round(
            sum(supported_f1) / len(supported_f1) if supported_f1 else 0.0,
            4,
        ),
        "critical_min_value_f1": round(critical_min_f1, 4),
        "critical_attributes": critical,
        "per_attribute": per_attribute,
    }
