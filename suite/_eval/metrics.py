"""Evaluation metrics for fashion attribute extraction."""

from __future__ import annotations

import json
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any


def fuzzy_match(pred: str, gold: str, threshold: float = 0.8) -> bool:
    """Check if two strings are a fuzzy match above the threshold."""
    pred_norm = pred.strip().lower()
    gold_norm = gold.strip().lower()
    if pred_norm == gold_norm:
        return True
    return SequenceMatcher(None, pred_norm, gold_norm).ratio() >= threshold


def per_attribute_f1(
    predictions: list[dict[str, Any]],
    golds: list[dict[str, Any]],
    fuzzy_threshold: float = 0.8,
) -> dict[str, dict[str, float]]:
    """Compute per-attribute precision, recall, F1 with fuzzy matching.

    Each entry is a dict with attribute names as keys and values that are either
    dicts with a 'value' key (single-label) or lists of dicts with 'value' keys (multi-label).
    """
    attr_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for pred, gold in zip(predictions, golds):
        all_keys = set(pred.keys()) | set(gold.keys())
        for key in all_keys:
            pred_val = pred.get(key)
            gold_val = gold.get(key)

            pred_values = _extract_values(pred_val)
            gold_values = _extract_values(gold_val)

            matched_golds: set[int] = set()
            for pv in pred_values:
                found = False
                for gi, gv in enumerate(gold_values):
                    if gi not in matched_golds and fuzzy_match(pv, gv, fuzzy_threshold):
                        attr_stats[key]["tp"] += 1
                        matched_golds.add(gi)
                        found = True
                        break
                if not found:
                    attr_stats[key]["fp"] += 1

            attr_stats[key]["fn"] += len(gold_values) - len(matched_golds)

    results: dict[str, dict[str, float]] = {}
    for attr, stats in attr_stats.items():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results[attr] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    return results


def macro_f1(per_attr_results: dict[str, dict[str, float]]) -> float:
    """Compute macro-averaged F1 across all attributes."""
    f1_scores = [v["f1"] for v in per_attr_results.values()]
    if not f1_scores:
        return 0.0
    return round(sum(f1_scores) / len(f1_scores), 4)


def expected_calibration_error(
    confidences: list[float],
    accuracies: list[bool],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE)."""
    if not confidences or len(confidences) != len(accuracies):
        return 0.0

    bin_counts: list[int] = [0] * n_bins
    bin_conf_sums: list[float] = [0.0] * n_bins
    bin_acc_sums: list[float] = [0.0] * n_bins

    for conf, acc in zip(confidences, accuracies):
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bin_counts[bin_idx] += 1
        bin_conf_sums[bin_idx] += conf
        bin_acc_sums[bin_idx] += float(acc)

    total = len(confidences)
    ece = 0.0
    for i in range(n_bins):
        if bin_counts[i] > 0:
            avg_conf = bin_conf_sums[i] / bin_counts[i]
            avg_acc = bin_acc_sums[i] / bin_counts[i]
            ece += (bin_counts[i] / total) * abs(avg_acc - avg_conf)

    return round(ece, 4)


def json_schema_validity_rate(
    outputs: list[str], required_fields: list[str] | None = None
) -> float:
    """Compute the fraction of outputs that parse as valid JSON with required fields."""
    if not outputs:
        return 0.0

    if required_fields is None:
        required_fields = ["master_category", "category"]

    valid = 0
    for output in outputs:
        try:
            parsed = json.loads(output)
            if isinstance(parsed, dict) and all(f in parsed for f in required_fields):
                valid += 1
        except (json.JSONDecodeError, TypeError):
            continue

    return round(valid / len(outputs), 4)


def _extract_values(val: Any) -> list[str]:
    """Extract string values from an attribute entry."""
    if val is None:
        return []
    if isinstance(val, dict):
        v = val.get("value")
        return [str(v)] if v is not None else []
    if isinstance(val, list):
        results = []
        for item in val:
            if isinstance(item, dict):
                v = item.get("value") or item.get("label")
                if v is not None:
                    results.append(str(v))
            elif isinstance(item, str):
                results.append(item)
        return results
    if isinstance(val, str):
        return [val]
    return []
