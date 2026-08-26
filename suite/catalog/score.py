#!/usr/bin/env python3
"""Score image-only MODA-schema predictions on frozen Shopping100k labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


class ScoringError(ValueError):
    """Raised when predictions violate the frozen Shopping100k scoring contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ScoringError(f"{path}:{line_number} is not an object")
        rows.append(value)
    return rows


def index_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        record_id = str(row.get("record_id") or row.get("custom_id") or "")
        if not record_id or record_id in result:
            raise ScoringError(f"Missing or duplicate record ID: {record_id!r}")
        result[record_id] = dict(row)
    return result


def raw_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ScoringError("Prediction attributes must contain strings or string lists")


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower().replace("_", " ")).strip()


def canonical(value: str, spec: Mapping[str, Any]) -> str:
    normalized = normalize(value)
    aliases = {normalize(key): normalize(mapped) for key, mapped in spec.get("aliases", {}).items()}
    return aliases.get(normalized, normalized)


def predicted_values(row: Mapping[str, Any], spec: Mapping[str, Any]) -> set[str]:
    attributes = row.get("attributes")
    if not isinstance(attributes, Mapping):
        raise ScoringError("Prediction row has no attributes object")
    values: set[str] = set()
    for field in spec["prediction_fields"]:
        values.update(canonical(value, spec) for value in raw_values(attributes.get(field)))
    return values


def f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 0.0 if not denominator else 2 * tp / denominator


def score(
    labels: list[dict[str, Any]], predictions: list[dict[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    label_index = index_rows(labels)
    prediction_index = index_rows(predictions)
    if set(label_index) != set(prediction_index):
        raise ScoringError(
            f"Prediction ID mismatch: missing={len(set(label_index) - set(prediction_index))}, "
            f"extra={len(set(prediction_index) - set(label_index))}"
        )
    per_attribute: dict[str, Any] = {}
    total_tp = total_fp = total_fn = total_exact = total_eligible = 0
    for attribute, spec in config["attributes"].items():
        classes = [normalize(value) for value in spec["values"]]
        class_counts = {value: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for value in classes}
        tp = fp = fn = exact = eligible = 0
        for record_id, row in label_index.items():
            labels_object = row.get("labels")
            if not isinstance(labels_object, Mapping):
                raise ScoringError("Release row has no labels object")
            raw_gold = labels_object.get(attribute)
            if raw_gold is None:
                continue
            eligible += 1
            gold = canonical(str(raw_gold), spec)
            predicted = predicted_values(prediction_index[record_id], spec)
            gold_set = {gold}
            tp += len(gold_set & predicted)
            fp += len(predicted - gold_set)
            fn += len(gold_set - predicted)
            exact += int(predicted == gold_set)
            for value in classes:
                class_counts[value]["support"] += int(gold == value)
                class_counts[value]["tp"] += int(gold == value and value in predicted)
                class_counts[value]["fp"] += int(gold != value and value in predicted)
                class_counts[value]["fn"] += int(gold == value and value not in predicted)
        supported = [counts for counts in class_counts.values() if counts["support"]]
        macro_class_f1 = sum(f1(item["tp"], item["fp"], item["fn"]) for item in supported)
        macro_class_f1 /= max(1, len(supported))
        per_attribute[attribute] = {
            "eligible_rows": eligible,
            "coverage": round(eligible / max(1, len(label_index)), 6),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": round(tp / max(1, tp + fp), 6),
            "recall": round(tp / max(1, tp + fn), 6),
            "set_f1": round(f1(tp, fp, fn), 6),
            "accuracy": round(exact / max(1, eligible), 6),
            "macro_class_f1": round(macro_class_f1, 6),
            "supported_classes": len(supported),
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_exact += exact
        total_eligible += eligible
    field_f1 = [metrics["set_f1"] for metrics in per_attribute.values()]
    return {
        "rows": len(label_index),
        "attributes": list(config["attributes"]),
        "field_macro_set_f1": round(sum(field_f1) / max(1, len(field_f1)), 6),
        "micro_set_f1": round(f1(total_tp, total_fp, total_fn), 6),
        "micro_accuracy": round(total_exact / max(1, total_eligible), 6),
        "eligible_cell_coverage": round(
            total_eligible / max(1, len(label_index) * len(per_attribute)), 6
        ),
        "per_attribute": per_attribute,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--system-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("suite/catalog/evaluation_config.json"),
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    result = {
        "protocol_id": config["protocol_id"],
        "system_id": args.system_id,
        "input_sha256": {
            "labels": sha256_file(args.labels),
            "predictions": sha256_file(args.predictions),
            "config": sha256_file(args.config),
        },
        "score": score(read_jsonl(args.labels), read_jsonl(args.predictions), config),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
