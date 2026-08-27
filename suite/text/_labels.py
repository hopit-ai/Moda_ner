"""Fashion NER label schema (13 entity types, 27 BIO labels)."""

from __future__ import annotations

ENTITY_TYPES: tuple[str, ...] = (
    "GARMENT_TYPE",
    "MATERIAL",
    "COLOR",
    "PATTERN",
    "SILHOUETTE",
    "FIT",
    "NECKLINE",
    "SLEEVE",
    "HEMLINE",
    "BRAND",
    "OCCASION",
    "AESTHETIC",
    "DETAIL",
)

LABEL_LIST: list[str] = ["O"]
for ent in ENTITY_TYPES:
    LABEL_LIST.extend([f"B-{ent}", f"I-{ent}"])

LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL: dict[int, str] = {i: label for label, i in LABEL2ID.items()}

NUM_LABELS = len(LABEL_LIST)

CONCRETE_TYPES = frozenset({"GARMENT_TYPE", "MATERIAL", "COLOR", "BRAND", "PATTERN"})
ABSTRACT_TYPES = frozenset({"OCCASION", "AESTHETIC", "FIT", "SILHOUETTE"})
