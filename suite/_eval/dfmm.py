"""Paper-compatible DeepFashion-MultiModal three-tier evaluation.

The protocol is defined by arXiv:2601.15711, accepted to WACVW 2026.
The paper does not publish its exact 14,000 image split IDs, so manifests built
from the public annotations are deliberately marked ``paper_compatible`` rather
than ``paper_exact`` unless an authoritative split file is supplied.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

NA = "NA"
MISSING = "__MISSING__"
HALLUCINATION = "__HALLUCINATION__"
PROTOCOL_ID = "dfmm-wacvw-2026-arxiv-2601.15711"


@dataclass(frozen=True)
class AttributeSpec:
    name: str
    group: str
    values: tuple[str, ...]
    source: str
    source_index: int
    raw_mapping: Mapping[int, str | None]

    @property
    def has_na(self) -> bool:
        return NA in self.values


FABRIC_VALUES = ("denim", "cotton", "leather", "furry", "knitted", "chiffon", "other", NA)
PATTERN_VALUES = (
    "floral",
    "graphic",
    "striped",
    "pure color",
    "lattice",
    "other",
    "color block",
    NA,
)


ATTRIBUTE_SPECS: tuple[AttributeSpec, ...] = (
    AttributeSpec(
        "sleeve_length",
        "shape",
        ("sleeveless", "short-sleeve", "medium-sleeve", "long-sleeve"),
        "shape",
        0,
        {
            0: "sleeveless",
            1: "short-sleeve",
            2: "medium-sleeve",
            3: "long-sleeve",
            4: None,
            5: None,
        },
    ),
    AttributeSpec(
        "lower_clothing_length",
        "shape",
        ("three-point", "medium short", "three-quarter", "long", NA),
        "shape",
        1,
        {0: "three-point", 1: "medium short", 2: "three-quarter", 3: "long", 4: NA},
    ),
    AttributeSpec(
        "socks",
        "shape",
        ("no", "socks", "leggings", NA),
        "shape",
        2,
        {0: "no", 1: "socks", 2: "leggings", 3: NA},
    ),
    AttributeSpec("hat", "shape", ("no", "yes", NA), "shape", 3, {0: "no", 1: "yes", 2: NA}),
    AttributeSpec(
        "glasses",
        "shape",
        ("no", "sunglasses", "have a glasses in hand or clothes", NA),
        "shape",
        4,
        {0: "no", 1: None, 2: "sunglasses", 3: "have a glasses in hand or clothes", 4: NA},
    ),
    AttributeSpec("neckwear", "shape", ("no", "yes", NA), "shape", 5, {0: "no", 1: "yes", 2: NA}),
    AttributeSpec(
        "wrist_wearing", "shape", ("no", "yes", NA), "shape", 6, {0: "no", 1: "yes", 2: NA}
    ),
    AttributeSpec("ring", "shape", ("no", "yes", NA), "shape", 7, {0: "no", 1: "yes", 2: NA}),
    AttributeSpec(
        "waist_accessories",
        "shape",
        ("no", "belt", "have a clothing", NA),
        "shape",
        8,
        {0: "no", 1: "belt", 2: "have a clothing", 3: NA, 4: NA},
    ),
    AttributeSpec(
        "neckline",
        "shape",
        ("V-shape", "square", "round", "standing", "lapel", "suspenders", NA),
        "shape",
        9,
        {0: "V-shape", 1: "square", 2: "round", 3: "standing", 4: "lapel", 5: "suspenders", 6: NA},
    ),
    AttributeSpec(
        "outer_clothing_cardigan",
        "shape",
        ("yes", "no"),
        "shape",
        10,
        {0: "yes", 1: "no", 2: "no"},
    ),
    AttributeSpec(
        "upper_clothing_covering_navel",
        "shape",
        ("no", "yes", NA),
        "shape",
        11,
        {0: "no", 1: "yes", 2: NA},
    ),
    AttributeSpec(
        "upper_fabric", "fabric", FABRIC_VALUES, "fabric", 0, dict(enumerate(FABRIC_VALUES))
    ),
    AttributeSpec(
        "lower_fabric", "fabric", FABRIC_VALUES, "fabric", 1, dict(enumerate(FABRIC_VALUES))
    ),
    AttributeSpec(
        "outer_fabric", "fabric", FABRIC_VALUES, "fabric", 2, dict(enumerate(FABRIC_VALUES))
    ),
    AttributeSpec(
        "upper_pattern", "pattern", PATTERN_VALUES, "pattern", 0, dict(enumerate(PATTERN_VALUES))
    ),
    AttributeSpec(
        "lower_pattern", "pattern", PATTERN_VALUES, "pattern", 1, dict(enumerate(PATTERN_VALUES))
    ),
    AttributeSpec(
        "outer_pattern", "pattern", PATTERN_VALUES, "pattern", 2, dict(enumerate(PATTERN_VALUES))
    ),
)

ATTRIBUTE_BY_NAME = {spec.name: spec for spec in ATTRIBUTE_SPECS}
ATTRIBUTE_NAMES = tuple(spec.name for spec in ATTRIBUTE_SPECS)
DFMM_AUXILIARY_FIELDS = ("master_category", "category")

_FILENAME_RE = re.compile(
    r"^(?P<gender>MEN|WOMEN)-(?P<category>.+?)-id_\d+-\d+_\d+_(?P<view>[^.]+)\.jpg$"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_filename(filename: str) -> dict[str, str]:
    match = _FILENAME_RE.fullmatch(filename)
    if not match:
        raise ValueError(f"Unsupported DeepFashion-MultiModal filename: {filename!r}")
    metadata = match.groupdict()
    metadata["category"] = metadata["category"].replace("_", " ").lower()
    metadata["gender"] = metadata["gender"].lower()
    metadata["description"] = (
        f"A {metadata['gender']}'s {metadata['category']} photographed from {metadata['view']} view"
    )
    return metadata


def read_annotation_table(path: str | Path, width: int) -> dict[str, tuple[int, ...]]:
    rows: dict[str, tuple[int, ...]] = {}
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != width + 1:
                raise ValueError(
                    f"{path}:{line_number}: expected {width} labels, got {len(parts) - 1}"
                )
            filename = parts[0]
            if filename in rows:
                raise ValueError(f"{path}:{line_number}: duplicate image {filename}")
            rows[filename] = tuple(int(value) for value in parts[1:])
    return rows


def load_public_annotations(
    shape_path: str | Path,
    fabric_path: str | Path,
    pattern_path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    shape = read_annotation_table(shape_path, 12)
    fabric = read_annotation_table(fabric_path, 3)
    pattern = read_annotation_table(pattern_path, 3)
    joined_ids = sorted(set(shape) & set(fabric) & set(pattern))
    excluded: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for filename in joined_ids:
        source_values = {
            "shape": shape[filename],
            "fabric": fabric[filename],
            "pattern": pattern[filename],
        }
        labels: dict[str, str] = {}
        valid = True
        for spec in ATTRIBUTE_SPECS:
            raw = source_values[spec.source][spec.source_index]
            if raw not in spec.raw_mapping or spec.raw_mapping[raw] is None:
                excluded[f"{spec.name}:{raw}"] += 1
                valid = False
                break
            labels[spec.name] = str(spec.raw_mapping[raw])
        if not valid:
            continue
        metadata = parse_filename(filename)
        rows.append(
            {
                "record_id": filename,
                "image_id": filename,
                "image_path": filename,
                "image": {"filename": filename, "image_group_id": filename},
                "metadata": {
                    "gender": metadata["gender"],
                    "category": metadata["category"],
                    "view": metadata["view"],
                },
                "text_description": metadata["description"],
                "dfmm_labels": labels,
                "attributes": {field: value for field, value in labels.items() if value != NA},
                "supervised_attributes": list(ATTRIBUTE_NAMES),
            }
        )
    audit = {
        "protocol": PROTOCOL_ID,
        "source_counts": {
            "shape": len(shape),
            "fabric": len(fabric),
            "pattern": len(pattern),
            "joined": len(joined_ids),
            "paper_taxonomy_eligible": len(rows),
        },
        "excluded_by_unsupported_source_label": dict(sorted(excluded.items())),
        "source_sha256": {
            "shape": sha256_file(shape_path),
            "fabric": sha256_file(fabric_path),
            "pattern": sha256_file(pattern_path),
        },
    }
    return rows, audit


def _stable_key(seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{value}".encode()).digest()


def _proportional_allocation(populations: Mapping[str, int], total: int) -> dict[str, int]:
    available = sum(populations.values())
    if total < 0 or total > available:
        raise ValueError(f"Cannot allocate {total} rows from population {available}")
    if not total:
        return dict.fromkeys(populations, 0)
    exact = {key: total * count / available for key, count in populations.items()}
    allocated = {key: min(math.floor(value), populations[key]) for key, value in exact.items()}
    remaining = total - sum(allocated.values())
    order = sorted(
        populations,
        key=lambda key: (exact[key] - math.floor(exact[key]), populations[key], key),
        reverse=True,
    )
    while remaining:
        progressed = False
        for key in order:
            if allocated[key] < populations[key]:
                allocated[key] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise RuntimeError("Proportional allocation could not satisfy requested total")
    return allocated


def build_paper_compatible_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    train_size: int = 7000,
    dev_size: int = 2000,
    test_size: int = 5000,
    seed: int = 20260122,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    total_size = train_size + dev_size + test_size
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        row = dict(source)
        metadata = row["metadata"]
        stratum = f"{metadata['gender']}|{metadata['category']}"
        strata[stratum].append(row)
    populations = {key: len(values) for key, values in strata.items()}
    selected_counts = _proportional_allocation(populations, total_size)
    selected: dict[str, list[dict[str, Any]]] = {}
    for key, values in strata.items():
        ordered = sorted(values, key=lambda row: _stable_key(seed, str(row["record_id"])))
        selected[key] = ordered[: selected_counts[key]]

    train_counts = _proportional_allocation(
        {key: len(values) for key, values in selected.items()}, train_size
    )
    remaining: dict[str, list[dict[str, Any]]] = {}
    result: list[dict[str, Any]] = []
    for key, values in selected.items():
        for row in values[: train_counts[key]]:
            result.append({**row, "split": "train"})
        remaining[key] = values[train_counts[key] :]
    dev_counts = _proportional_allocation(
        {key: len(values) for key, values in remaining.items()}, dev_size
    )
    for key, values in remaining.items():
        for row in values[: dev_counts[key]]:
            result.append({**row, "split": "dev"})
        for row in values[dev_counts[key] :]:
            result.append({**row, "split": "test"})

    result.sort(key=lambda row: (str(row["split"]), str(row["record_id"])))
    actual = Counter(str(row["split"]) for row in result)
    expected = {"train": train_size, "dev": dev_size, "test": test_size}
    if dict(actual) != expected:
        raise RuntimeError(f"Split construction failed: expected {expected}, got {dict(actual)}")
    audit = {
        "protocol": PROTOCOL_ID,
        "protocol_status": "paper_compatible_not_exact",
        "reason_not_exact": (
            "The paper specifies stratification and 7K/2K/5K sizes but does not publish "
            "the authoritative image IDs or random seed."
        ),
        "seed": seed,
        "split_counts": expected,
        "stratification_key": "gender|product_category_from_filename",
        "strata": len(strata),
        "selected_record_ids_sha256": hashlib.sha256(
            "\n".join(str(row["record_id"]) for row in result).encode()
        ).hexdigest(),
    }
    return result, audit


def classifier_vocabulary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    positive_rows = {name: 0 for name in ATTRIBUTE_NAMES}
    value_support = {name: Counter() for name in ATTRIBUTE_NAMES}
    for row in materialized:
        labels = row["dfmm_labels"]
        for name in ATTRIBUTE_NAMES:
            value = str(labels[name])
            if value != NA:
                positive_rows[name] += 1
                value_support[name][value] += 1
    return {
        "fields": list(ATTRIBUTE_NAMES),
        "values": {
            spec.name: [value for value in spec.values if value != NA] for spec in ATTRIBUTE_SPECS
        },
        "positive_rows": positive_rows,
        "value_support": {
            name: dict(sorted(counts.items())) for name, counts in value_support.items()
        },
        "multi_label_fields": [],
    }


def classifier_training_vocabulary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the classifier vocabulary used by the image-only DFMM adapter.

    The existing MODA classifier treats ``master_category`` and ``category`` as
    categorical heads.  DFMM does not score those fields, but gender and product
    category are available from every filename and make safe auxiliary targets.
    The benchmark output remains exactly the 18 fields in ``ATTRIBUTE_NAMES``.
    """
    materialized = list(rows)
    dfmm = classifier_vocabulary(materialized)
    from .conditional import normalize_value

    normalized_values = {
        spec.name: [
            normalize_value(spec.name, value)
            for value in spec.values
            if value != NA
        ]
        for spec in ATTRIBUTE_SPECS
    }
    normalized_support = {
        field: {
            normalize_value(field, value): count
            for value, count in counts.items()
        }
        for field, counts in dfmm["value_support"].items()
    }
    gender_support = Counter(str(row["metadata"]["gender"]) for row in materialized)
    category_support = Counter(str(row["metadata"]["category"]) for row in materialized)
    if not gender_support or not category_support:
        raise ValueError("DFMM classifier training rows require gender and category metadata")
    fields = [*DFMM_AUXILIARY_FIELDS, *ATTRIBUTE_NAMES]
    return {
        "fields": fields,
        "values": {
            "master_category": sorted(gender_support),
            "category": sorted(category_support),
            **normalized_values,
        },
        "positive_rows": {
            "master_category": len(materialized),
            "category": len(materialized),
            **dfmm["positive_rows"],
        },
        "value_support": {
            "master_category": dict(sorted(gender_support.items())),
            "category": dict(sorted(category_support.items())),
            **normalized_support,
        },
        "multi_label_fields": [],
    }


def build_classifier_release(
    split_rows: Sequence[Mapping[str, Any]],
    *,
    image_s3_prefix: str,
    calibration_size: int = 1000,
    seed: int = 20260808,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    """Convert paper splits into a record-ID-disjoint MODA classifier release.

    The paper's 2K development rows are deterministically divided into 1K
    threshold-calibration rows and 1K development rows.  The 5K test set is
    retained verbatim and is never consumed by the training entrypoint.  This
    function checks exact image IDs only; multiple views of the same product can
    still cross partitions and require a separate product-identity audit.
    """
    prefix = image_s3_prefix.rstrip("/")
    by_split: dict[str, list[Mapping[str, Any]]] = {
        split: [row for row in split_rows if row.get("split") == split]
        for split in ("train", "dev", "test")
    }
    if not all(by_split.values()):
        counts = {split: len(rows) for split, rows in by_split.items()}
        raise ValueError(f"DFMM paper release is incomplete: {counts}")
    if not 0 < calibration_size < len(by_split["dev"]):
        raise ValueError("calibration_size must leave at least one development row")

    def adapt(source: Mapping[str, Any], destination_split: str) -> dict[str, Any]:
        row = dict(source)
        metadata = dict(row["metadata"])
        filename = str(row["record_id"])
        attributes = dict(row.get("attributes") or {})
        attributes.update(
            {
                "master_category": str(metadata["gender"]),
                "category": str(metadata["category"]),
            }
        )
        image = dict(row.get("image") or {})
        image["s3_uri"] = f"{prefix}/{filename}"
        row.update(
            {
                "attributes": attributes,
                "image": image,
                "split": destination_split,
                "source_split": str(source["split"]),
                "supervised_attributes": [*DFMM_AUXILIARY_FIELDS, *ATTRIBUTE_NAMES],
            }
        )
        return row

    train = [adapt(row, "train") for row in by_split["train"]]
    ordered_dev = sorted(
        by_split["dev"],
        key=lambda row: _stable_key(seed, str(row["record_id"])),
    )
    calibration = [
        adapt(row, "calibration") for row in ordered_dev[:calibration_size]
    ]
    development = [
        adapt(row, "development") for row in ordered_dev[calibration_size:]
    ]
    test = [adapt(row, "test") for row in by_split["test"]]
    release = {
        "train": train,
        "calibration": calibration,
        "development": development,
        "test": test,
    }
    id_sets = {
        split: {str(row["record_id"]) for row in rows}
        for split, rows in release.items()
    }
    overlap = {
        f"{left}_{right}": len(id_sets[left] & id_sets[right])
        for index, left in enumerate(release)
        for right in tuple(release)[index + 1 :]
    }
    if any(overlap.values()):
        raise RuntimeError(f"DFMM classifier release contains split overlap: {overlap}")
    vocabulary = classifier_training_vocabulary(train)
    manifest = {
        "protocol": PROTOCOL_ID,
        "protocol_status": "paper_compatible_not_exact",
        "adapter": "moda_distilled_image_only_auxiliary_gender_category_v1",
        "benchmark_output_fields": list(ATTRIBUTE_NAMES),
        "auxiliary_training_fields": list(DFMM_AUXILIARY_FIELDS),
        "image_s3_prefix": prefix,
        "seed": seed,
        "split_rows": {split: len(rows) for split, rows in release.items()},
        "source_split_rows": {
            split: len(rows) for split, rows in by_split.items()
        },
        "split_overlap": overlap,
        "split_overlap_scope": "exact_record_id_only",
        "product_identity_overlap_audited": False,
        "test_used_for_training_or_calibration": False,
        "selected_record_ids_sorted_sha256": hashlib.sha256(
            "\n".join(sorted(set().union(*id_sets.values()))).encode()
        ).hexdigest(),
    }
    return release, vocabulary, manifest


def complete_dfmm_attributes(attributes: Mapping[str, Any]) -> dict[str, str]:
    """Drop auxiliary fields and convert sparse abstentions into explicit NA."""
    from .conditional import normalize_value

    completed: dict[str, str] = {}
    for spec in ATTRIBUTE_SPECS:
        value = attributes.get(spec.name)
        if isinstance(value, list):
            value = value[0] if value else None
        normalized_to_canonical = {
            normalize_value(spec.name, candidate): candidate for candidate in spec.values
        }
        if isinstance(value, str) and value in spec.values:
            completed[spec.name] = value
        elif isinstance(value, str) and value in normalized_to_canonical:
            completed[spec.name] = normalized_to_canonical[value]
        elif spec.has_na:
            completed[spec.name] = NA
        else:
            raise ValueError(f"Prediction omitted required non-NA field {spec.name}")
    return completed


def flatten_prediction(row: Mapping[str, Any]) -> tuple[dict[str, str], int]:
    source = row.get("predictions") or row.get("attributes") or row
    if not isinstance(source, Mapping):
        return {}, 0
    flattened: dict[str, str] = {}
    unknown_fields = 0
    for section in ("shape_attributes", "fabric_attributes", "pattern_attributes"):
        nested = source.get(section)
        if isinstance(nested, Mapping):
            for field, payload in nested.items():
                if field not in ATTRIBUTE_BY_NAME:
                    unknown_fields += 1
                    continue
                value = payload.get("value") if isinstance(payload, Mapping) else payload
                if isinstance(value, str):
                    flattened[field] = value.strip()
    for field, payload in source.items():
        if field in {"shape_attributes", "fabric_attributes", "pattern_attributes"}:
            continue
        if field not in ATTRIBUTE_BY_NAME:
            continue
        value = payload.get("value") if isinstance(payload, Mapping) else payload
        if isinstance(value, str):
            flattened[field] = value.strip()
    return flattened, unknown_fields


def _class_metrics(
    gold: Sequence[str], predicted: Sequence[str], labels: Sequence[str]
) -> dict[str, Any]:
    per_class: dict[str, dict[str, float | int]] = {}
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted, strict=True))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted, strict=True))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted, strict=True))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(value == label for value in gold),
        }
    return {
        "macro_precision": sum(float(item["precision"]) for item in per_class.values())
        / len(labels),
        "macro_recall": sum(float(item["recall"]) for item in per_class.values()) / len(labels),
        "macro_f1": sum(float(item["f1"]) for item in per_class.values()) / len(labels),
        "per_class": per_class,
    }


def _binary_na_metrics(gold: Sequence[str], predicted: Sequence[str]) -> dict[str, float | int]:
    tp = sum(g == NA and p == NA for g, p in zip(gold, predicted, strict=True))
    fp = sum(g != NA and p == NA for g, p in zip(gold, predicted, strict=True))
    fn = sum(g == NA and p != NA for g, p in zip(gold, predicted, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}


def score_three_tier(
    gold_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    gold_by_id = {str(row.get("record_id") or row.get("image_id")): row for row in gold_rows}
    pred_by_id = {
        str(row.get("record_id") or row.get("image_id") or row.get("id")): row
        for row in prediction_rows
    }
    missing_ids = sorted(set(gold_by_id) - set(pred_by_id))
    extra_ids = sorted(set(pred_by_id) - set(gold_by_id))
    per_attribute: dict[str, dict[str, Any]] = {}
    hallucinations = 0
    missing_fields = 0
    unknown_fields = 0
    group_tiers: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"tier1": [], "tier2": [], "tier3": []}
    )
    for spec in ATTRIBUTE_SPECS:
        gold: list[str] = []
        predicted: list[str] = []
        attr_hallucinations = 0
        for record_id, gold_row in gold_by_id.items():
            labels = gold_row.get("dfmm_labels") or gold_row.get("labels")
            if not isinstance(labels, Mapping) or spec.name not in labels:
                raise ValueError(f"Gold row {record_id} lacks {spec.name}")
            gold_value = str(labels[spec.name])
            prediction, row_unknown = flatten_prediction(pred_by_id.get(record_id, {}))
            unknown_fields += row_unknown
            predicted_value = prediction.get(spec.name, MISSING)
            if predicted_value == MISSING:
                missing_fields += 1
            elif predicted_value not in spec.values:
                attr_hallucinations += 1
                hallucinations += 1
                predicted_value = HALLUCINATION
            gold.append(gold_value)
            predicted.append(predicted_value)
        tier1 = _class_metrics(gold, predicted, spec.values)
        if spec.has_na:
            tier2 = _binary_na_metrics(gold, predicted)
            visible_indices = [index for index, value in enumerate(gold) if value != NA]
            tier3 = _class_metrics(
                [gold[index] for index in visible_indices],
                [predicted[index] for index in visible_indices],
                spec.values,
            )
            group_tiers[spec.group]["tier2"].append(float(tier2["f1"]))
        else:
            tier2 = None
            tier3 = tier1
        group_tiers[spec.group]["tier1"].append(float(tier1["macro_f1"]))
        group_tiers[spec.group]["tier3"].append(float(tier3["macro_f1"]))
        per_attribute[spec.name] = {
            "group": spec.group,
            "tier1": tier1,
            "tier2": tier2,
            "tier3": tier3,
            "hallucinations": attr_hallucinations,
        }

    tier1_values = [float(row["tier1"]["macro_f1"]) for row in per_attribute.values()]
    tier2_values = [
        float(row["tier2"]["f1"]) for row in per_attribute.values() if row["tier2"] is not None
    ]
    tier3_values = [float(row["tier3"]["macro_f1"]) for row in per_attribute.values()]

    def rounded_mean(values: Sequence[float]) -> float:
        return round(sum(values) / len(values), 6) if values else 0.0

    category_summary = {
        group: {tier: rounded_mean(values) for tier, values in tiers.items()}
        for group, tiers in group_tiers.items()
    }
    return {
        "protocol": PROTOCOL_ID,
        "rows": len(gold_rows),
        "model_level": {
            "tier1_macro_f1": rounded_mean(tier1_values),
            "tier2_na_f1": rounded_mean(tier2_values),
            "tier3_visible_macro_f1": rounded_mean(tier3_values),
        },
        "category_level": category_summary,
        "per_attribute": per_attribute,
        "schema": {
            "expected_predictions": len(gold_rows) * len(ATTRIBUTE_SPECS),
            "missing_record_ids": missing_ids,
            "extra_record_ids": extra_ids,
            "missing_fields": missing_fields,
            "hallucinations": hallucinations,
            "unknown_fields_in_nested_sections": unknown_fields,
        },
    }


def dump_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, sort_keys=True) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def manifest_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_split_id_integrity(
    split_record_ids: Mapping[str, Sequence[str]],
    *,
    expected_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Fail closed on duplicate IDs, unexpected counts, or split leakage."""
    normalized = {
        str(split): [str(record_id) for record_id in record_ids]
        for split, record_ids in split_record_ids.items()
    }
    counts: dict[str, int] = {}
    digests: dict[str, str] = {}
    id_sets: dict[str, set[str]] = {}
    for split, record_ids in normalized.items():
        unique_ids = set(record_ids)
        if len(unique_ids) != len(record_ids):
            duplicates = sorted(
                record_id
                for record_id, count in Counter(record_ids).items()
                if count > 1
            )
            raise ValueError(f"Duplicate record IDs in {split}: {duplicates[:10]}")
        if expected_counts is not None:
            if split not in expected_counts:
                raise ValueError(f"Missing expected count for split {split}")
            if len(record_ids) != int(expected_counts[split]):
                raise ValueError(
                    f"Unexpected {split} rows: {len(record_ids)} != "
                    f"{int(expected_counts[split])}"
                )
        counts[split] = len(record_ids)
        digests[split] = hashlib.sha256(
            "\n".join(sorted(unique_ids)).encode()
        ).hexdigest()
        id_sets[split] = unique_ids
    if expected_counts is not None and set(normalized) != set(expected_counts):
        raise ValueError(
            "Split names do not match expected counts: "
            f"{sorted(normalized)} != {sorted(expected_counts)}"
        )
    overlap: dict[str, int] = {}
    split_names = sorted(id_sets)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            shared = id_sets[left] & id_sets[right]
            overlap[f"{left}__{right}"] = len(shared)
            if shared:
                raise ValueError(
                    f"Split leakage between {left} and {right}: {sorted(shared)[:10]}"
                )
    return {
        "counts": counts,
        "sorted_record_ids_sha256": digests,
        "overlap": overlap,
    }


def enforce_cost_gate(
    config: Mapping[str, Any],
    *,
    stage: str,
    api_cost_usd: float,
    gpu_cost_usd: float,
    optimizer_steps: int | None = None,
    epochs: int | None = None,
) -> dict[str, Any]:
    """Fail closed when an estimated paid stage exceeds either budget cap."""
    stages = config.get("stages")
    if not isinstance(stages, Mapping) or stage not in stages:
        raise ValueError(f"Unknown cost-gate stage: {stage}")
    limits = stages[stage]
    if not isinstance(limits, Mapping):
        raise ValueError(f"Invalid cost-gate configuration for {stage}")
    max_api = float(limits["max_api_cost"])
    max_gpu = float(limits["max_gpu_cost"])
    if api_cost_usd < 0 or gpu_cost_usd < 0:
        raise ValueError("Cost estimates cannot be negative")
    if optimizer_steps is not None and optimizer_steps < 0:
        raise ValueError("optimizer_steps cannot be negative")
    if epochs is not None and epochs < 0:
        raise ValueError("epochs cannot be negative")
    max_steps = (
        int(limits["max_optimizer_steps"])
        if limits.get("max_optimizer_steps") is not None
        else None
    )
    max_epochs = int(limits["max_epochs"]) if limits.get("max_epochs") is not None else None
    steps_approved = max_steps is None or (
        optimizer_steps is not None and optimizer_steps <= max_steps
    )
    epochs_approved = max_epochs is None or (epochs is not None and epochs <= max_epochs)
    approved = (
        api_cost_usd <= max_api
        and gpu_cost_usd <= max_gpu
        and steps_approved
        and epochs_approved
    )
    decision = {
        "stage": stage,
        "approved": approved,
        "estimated_api_cost_usd": round(api_cost_usd, 6),
        "estimated_gpu_cost_usd": round(gpu_cost_usd, 6),
        "max_api_cost_usd": max_api,
        "max_gpu_cost_usd": max_gpu,
        "optimizer_steps": optimizer_steps,
        "max_optimizer_steps": max_steps,
        "epochs": epochs,
        "max_epochs": max_epochs,
    }
    if not approved:
        raise RuntimeError(json.dumps(decision, sort_keys=True))
    return decision
