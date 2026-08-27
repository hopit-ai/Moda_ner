"""Vocabulary and split contract for discriminative attribute training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

CATEGORY_FIELDS = ("master_category", "category")
PUBLIC_FIELDS = (
    "master_category",
    "category",
    "sub_category",
    "silhouette",
    "hemline",
    "sleeve_length",
    "sleeve_shape",
    "neckline",
    "collar_presence",
    "collar_style",
    "waist_type",
    "material",
    "surface_treatment",
    "pattern",
    "closure_type",
)
COLOR_FIELDS = ("color_palette_primary",)
COLOR_AWARE_FIELDS = PUBLIC_FIELDS + COLOR_FIELDS
MULTI_LABEL_FIELDS = frozenset({"material", "surface_treatment", "color_palette_primary"})


def _values(field: str, value: Any) -> tuple[str, ...]:
    from .._eval.conditional import normalize_value

    if isinstance(value, Mapping):
        value = value.get("value")
    if isinstance(value, str):
        raw = [value]
    elif isinstance(value, list):
        raw = [item.get("value") if isinstance(item, Mapping) else item for item in value]
        raw = [item for item in raw if isinstance(item, str)]
    else:
        raw = []
    normalized = {
        normalize_value(field, item)
        for item in raw
        if item.strip() and normalize_value(field, item)
    }
    return tuple(sorted(normalized))


def group_id(row: Mapping[str, Any], index: int = 0) -> str:
    """Return the source-image identity used for leakage-safe splits."""
    image = row.get("image") if isinstance(row.get("image"), Mapping) else {}
    return str(
        row.get("image_id")
        or image.get("image_group_id")
        or image.get("image_id")
        or row.get("record_id")
        or row.get("id")
        or f"row:{index}"
    )


def split_for_group(
    value: str,
    *,
    calibration_fraction: float = 0.1,
    development_fraction: float = 0.0,
    seed: int = 20260806,
) -> str:
    """Assign an entire image group to train, calibration, or development."""
    if calibration_fraction < 0.0 or development_fraction < 0.0:
        raise ValueError("split fractions cannot be negative")
    if not 0.0 < calibration_fraction + development_fraction < 1.0:
        raise ValueError("held-out split fractions must sum to between zero and one")
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    if bucket < calibration_fraction:
        return "calibration"
    if bucket < calibration_fraction + development_fraction:
        return "development"
    return "train"


@dataclass(frozen=True)
class AttributeVocabulary:
    """Frozen field-value indices and observed support."""

    fields: tuple[str, ...]
    values: dict[str, tuple[str, ...]]
    positive_rows: dict[str, int]
    value_support: dict[str, dict[str, int]]
    multi_label_fields: tuple[str, ...]

    def value_to_index(self, field: str) -> dict[str, int]:
        return {value: index for index, value in enumerate(self.values[field])}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AttributeVocabulary:
        return cls(
            fields=tuple(payload["fields"]),
            values={key: tuple(values) for key, values in payload["values"].items()},
            positive_rows={key: int(value) for key, value in payload["positive_rows"].items()},
            value_support={
                field: {value: int(count) for value, count in counts.items()}
                for field, counts in payload["value_support"].items()
            },
            multi_label_fields=tuple(payload["multi_label_fields"]),
        )

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def build_vocabulary(
    rows: Iterable[Mapping[str, Any]],
    *,
    fields: tuple[str, ...] = PUBLIC_FIELDS,
    multi_label_fields: frozenset[str] = MULTI_LABEL_FIELDS,
) -> AttributeVocabulary:
    """Build a deterministic vocabulary from training annotations only."""
    from .._eval.entries import attributes_from_entry

    support = {field: {} for field in fields}
    positive_rows = {field: 0 for field in fields}
    for row in rows:
        attributes = attributes_from_entry(dict(row))
        for field in fields:
            values = _values(field, attributes.get(field))
            if values:
                positive_rows[field] += 1
            counts = support[field]
            for value in values:
                counts[value] = counts.get(value, 0) + 1
    missing = [field for field in CATEGORY_FIELDS if not support[field]]
    if missing:
        raise ValueError(f"No training values for required fields: {missing}")
    return AttributeVocabulary(
        fields=fields,
        values={field: tuple(sorted(support[field])) for field in fields},
        positive_rows=positive_rows,
        value_support={field: dict(sorted(counts.items())) for field, counts in support.items()},
        multi_label_fields=tuple(sorted(multi_label_fields & set(fields))),
    )


def encode_attributes(
    attributes: Mapping[str, Any],
    vocabulary: AttributeVocabulary,
) -> dict[str, Any]:
    """Encode one sparse attribute mapping into category/applicability/value targets."""
    encoded: dict[str, Any] = {"categories": {}, "applicability": {}, "values": {}}
    for field in vocabulary.fields:
        values = _values(field, attributes.get(field))
        indices = vocabulary.value_to_index(field)
        known = tuple(indices[value] for value in values if value in indices)
        if field in CATEGORY_FIELDS:
            encoded["categories"][field] = known[0] if known else -100
            continue
        encoded["applicability"][field] = float(bool(known))
        encoded["values"][field] = known
    return encoded
