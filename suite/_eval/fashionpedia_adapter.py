"""Adapter to map Fashionpedia annotations to the Hopit attribute schema.

Fashionpedia annotates each apparel instance with a category plus a set of
fine-grained ``attribute_ids``. The attributes are organised into
*supercategories* (silhouette, neckline type, length, waistline, textile
pattern, opening type, ...) that map cleanly onto the Hopit Tier 1/2 schema.

Two correctness fixes over the original adapter:

1. **Full attribute coverage.** Previously only material/pattern/surface were
   mapped; silhouette, neckline, hemline, sleeve_length, waist_type and
   sub_category (the bulk of Tier 1) were dropped on the floor.
2. **Garment-part filtering.** Fashionpedia annotates ``sleeve``, ``neckline``,
   ``collar``, ``pocket``, closures and decorations as *separate* instances
   (47% of rows). Treating them as standalone garments poisons training and
   eval, so garment-level conversion skips them.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Category name -> Hopit master_category.
FASHIONPEDIA_TO_HOPIT_CATEGORY: dict[str, str] = {
    "shirt, blouse": "topwear",
    "top, t-shirt, sweatshirt": "topwear",
    "sweater": "topwear",
    "cardigan": "outerwear",
    "jacket": "outerwear",
    "vest": "topwear",
    "pants": "bottomwear",
    "shorts": "bottomwear",
    "skirt": "bottomwear",
    "coat": "outerwear",
    "dress": "dress",
    "jumpsuit": "dress",
    "cape": "outerwear",
    "glasses": "accessories",
    "hat": "headwear",
    "headband, head covering, hair accessory": "headwear",
    "tie": "accessories",
    "glove": "accessories",
    "watch": "accessories",
    "belt": "accessories",
    "leg warmer": "accessories",
    "tights, stockings": "bottomwear",
    "sock": "accessories",
    "shoe": "footwear",
    "bag, wallet": "accessories",
    "scarf": "accessories",
    "umbrella": "accessories",
}

# Fashionpedia category supercategories that are garment *parts*, not garments.
# These are skipped during garment-level conversion.
GARMENT_PART_SUPERCATEGORIES = {"garment parts", "closures", "decorations"}

# Attribute supercategory -> (Hopit field, multi_label).
ATTR_SUPERCATEGORY_TO_FIELD: dict[str, tuple[str, bool]] = {
    "silhouette": ("silhouette", False),
    "neckline type": ("neckline", False),
    "waistline": ("waist_type", False),
    "textile pattern": ("pattern", False),
    "animal": ("pattern", False),
    "opening type": ("closure_type", False),
    "textile finishing, manufacturing techniques": ("surface_treatment", True),
    "non-textile material type": ("material", True),
    "leather": ("material", True),
    "nickname": ("sub_category", False),
    # "length" handled specially (split into sleeve_length vs hemline).
}

# Length-supercategory values that denote sleeve length (everything else in the
# "length" supercategory is treated as garment hemline).
SLEEVE_LENGTH_VALUES = {
    "sleeveless",
    "elbow-length",
    "three quarter (length)",
    "wrist-length",
}

_QUALIFIER_RE = re.compile(
    r"\s*\((?:neck|neckline|collar|length|pattern|wasitline|waistline|a|opening)\)$"
)


def _clean_value(name: str) -> str:
    """Strip Fashionpedia parenthetical qualifiers, e.g. 'round (neck)' -> 'round'."""
    cleaned = _QUALIFIER_RE.sub("", name).strip()
    return cleaned or name


def _is_absence(name: str) -> bool:
    """True for sentinel values meaning the attribute is absent."""
    low = name.lower()
    return low.startswith("no ") or low in {"collarless", "no opening", "no waistline"}


def _resolve_image_path(images_dir: Path, file_name: str) -> str:
    """Resolve image path across train/test/val subfolders."""
    for sub in ("train", "test", "val", ""):
        candidate = images_dir / sub / file_name if sub else images_dir / file_name
        if candidate.exists():
            return str(candidate)
    return str(images_dir / file_name)


def is_garment_part(
    annotation: dict[str, Any],
    category_supercategory: dict[int, str] | None,
) -> bool:
    """True if the annotation is a garment part/closure/decoration (not a garment)."""
    if not category_supercategory:
        return False
    sup = category_supercategory.get(annotation.get("category_id", -1), "")
    return sup in GARMENT_PART_SUPERCATEGORIES


def convert_fashionpedia_annotation(
    annotation: dict[str, Any],
    category_mapping: dict[int, str] | None = None,
    attribute_mapping: dict[int, str] | None = None,
    attribute_supercategory: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Convert a single Fashionpedia annotation to nested Hopit-schema JSON."""
    cat_name = ""
    if category_mapping and "category_id" in annotation:
        cat_name = category_mapping.get(annotation["category_id"], "")

    master_cat = FASHIONPEDIA_TO_HOPIT_CATEGORY.get(cat_name, "accessories")

    result: dict[str, Any] = {
        "master_category": {"value": master_cat, "confidence": 0.9},
        "category": {"value": cat_name, "confidence": 0.85},
    }

    single_fields_seen: set[str] = set()

    if "attribute_ids" in annotation and attribute_mapping:
        for attr_id in annotation["attribute_ids"]:
            attr_name = attribute_mapping.get(attr_id, "")
            if not attr_name:
                continue
            if _is_absence(attr_name):
                # Absence is searchable evidence, not a missing label.
                if attr_name.lower() == "collarless":
                    result["collar_presence"] = {
                        "value": "absent",
                        "confidence": 0.9,
                    }
                    single_fields_seen.add("collar_presence")
                continue
            sup = (attribute_supercategory or {}).get(attr_id, "")

            if sup == "length":
                field = "sleeve_length" if attr_name in SLEEVE_LENGTH_VALUES else "hemline"
                multi = False
            else:
                mapping = ATTR_SUPERCATEGORY_TO_FIELD.get(sup)
                if mapping is None:
                    continue
                field, multi = mapping

            value = _clean_value(attr_name)
            if multi:
                result.setdefault(field, [])
                if not any(v.get("value") == value for v in result[field]):
                    result[field].append({"value": value, "confidence": 0.8})
            else:
                # Keep the first value for single-label fields (deterministic).
                if field not in single_fields_seen:
                    result[field] = {"value": value, "confidence": 0.8}
                    single_fields_seen.add(field)

    return result


# Part category name -> (Hopit field, attribute supercategories to pull, multi).
# Lets us recover neckline / sleeve attributes that Fashionpedia annotates on
# separate *part* segments and attach them to the containing garment.
PART_FIELD_RULES: dict[str, list[tuple[str, set[str], bool]]] = {
    "neckline": [("neckline", {"neckline type"}, False)],
    "sleeve": [
        ("sleeve_length", {"length"}, False),
        ("sleeve_shape", {"nickname"}, False),
    ],
    "collar": [("collar_style", {"nickname"}, False)],
}


def _bbox_xyxy(bbox: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    return x, y, x + w, y + h


def _contains(garment: list[float], part: list[float]) -> bool:
    gx1, gy1, gx2, gy2 = _bbox_xyxy(garment)
    px, py, pw, ph = part
    cx, cy = px + pw / 2, py + ph / 2
    return gx1 <= cx <= gx2 and gy1 <= cy <= gy2


def _area(bbox: list[float]) -> float:
    return float(bbox[2] * bbox[3])


def _attach_part(
    result: dict[str, Any],
    part: dict[str, Any],
    part_name: str,
    attribute_mapping: dict[int, str],
    attribute_supercategory: dict[int, str],
) -> None:
    """Map a part instance's attributes into the parent garment result dict."""
    rules = PART_FIELD_RULES.get(part_name, [])
    if part_name == "collar":
        result["collar_presence"] = {"value": "present", "confidence": 0.85}
    for field, supercats, _multi in rules:
        if field in result:
            continue
        for attr_id in part.get("attribute_ids", []):
            sup = attribute_supercategory.get(attr_id, "")
            if sup in supercats:
                name = attribute_mapping.get(attr_id, "")
                if not name or _is_absence(name):
                    continue
                result[field] = {"value": _clean_value(name), "confidence": 0.75}
                break


def convert_fashionpedia_dataset_linked(
    annotations_json: str,
    output_jsonl: str,
    images_dir: str | None = None,
    *,
    max_samples: int | None = None,
) -> int:
    """Image-centric conversion that links garment parts to parent garments.

    Garment-level attributes (silhouette, pattern, waistline, ...) come from the
    garment instance; neckline and sleeve attributes are pulled from the
    containing ``neckline`` / ``sleeve`` part segments (Fashionpedia annotates
    those separately). One output row per garment instance.
    """
    with open(annotations_json) as f:
        data = json.load(f)

    categories = {c["id"]: c["name"] for c in data.get("categories", [])}
    cat_supercat = {c["id"]: c.get("supercategory", "") for c in data.get("categories", [])}
    attributes = {a["id"]: a["name"] for a in data.get("attributes", [])}
    attr_supercat = {a["id"]: a.get("supercategory", "") for a in data.get("attributes", [])}
    img_lookup = {img["id"]: img["file_name"] for img in data.get("images", [])}

    by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in data.get("annotations", []):
        by_image.setdefault(ann.get("image_id", -1), []).append(ann)

    output = Path(output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    img_root = Path(images_dir) if images_dir else Path()

    count = 0
    with open(output, "w") as fout:
        for image_id, anns in by_image.items():
            garments = [a for a in anns if not is_garment_part(a, cat_supercat)]
            linkable_parts = [
                a for a in anns if categories.get(a["category_id"], "") in PART_FIELD_RULES
            ]
            for g in garments:
                result = convert_fashionpedia_annotation(g, categories, attributes, attr_supercat)
                gbox = g.get("bbox")
                if gbox:
                    # Attach the smallest containing part of each relevant type.
                    parts_in = [
                        p for p in linkable_parts if p.get("bbox") and _contains(gbox, p["bbox"])
                    ]
                    parts_in.sort(key=lambda p: _area(p["bbox"]))
                    for p in parts_in:
                        _attach_part(
                            result,
                            p,
                            categories.get(p["category_id"], ""),
                            attributes,
                            attr_supercat,
                        )

                image_file = img_lookup.get(image_id, "")
                image_path = _resolve_image_path(img_root, image_file) if image_file else ""
                entry = {
                    "image_path": image_path,
                    "attributes_json": json.dumps(result),
                    "source": "fashionpedia",
                    "bbox": gbox,
                    "image_id": image_id,
                    "annotation_id": g.get("id"),
                }
                fout.write(json.dumps(entry) + "\n")
                count += 1
                if max_samples and count >= max_samples:
                    logger.info("Converted %d (linked) → %s", count, output)
                    return count

    logger.info("Converted %d (linked) Fashionpedia garments → %s", count, output)
    return count


def convert_fashionpedia_dataset(
    annotations_json: str,
    output_jsonl: str,
    images_dir: str | None = None,
    *,
    garments_only: bool = True,
    max_samples: int | None = None,
) -> int:
    """Convert a Fashionpedia annotation file to Hopit training JSONL.

    Args:
        annotations_json: Path to ``instances_attributes_*2020.json``.
        output_jsonl: Output JSONL path.
        images_dir: Root images dir (train/test/val subfolders auto-resolved).
        garments_only: Skip garment-part/closure/decoration instances.
        max_samples: Cap the number of written rows (deterministic order).

    Returns:
        Number of rows written.
    """
    with open(annotations_json) as f:
        data = json.load(f)

    categories = {c["id"]: c["name"] for c in data.get("categories", [])}
    cat_supercat = {c["id"]: c.get("supercategory", "") for c in data.get("categories", [])}
    attributes = {a["id"]: a["name"] for a in data.get("attributes", [])}
    attr_supercat = {a["id"]: a.get("supercategory", "") for a in data.get("attributes", [])}
    img_lookup = {img["id"]: img["file_name"] for img in data.get("images", [])}

    output = Path(output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)

    img_root = Path(images_dir) if images_dir else Path()
    count = 0
    with open(output, "w") as fout:
        for ann in data.get("annotations", []):
            if garments_only and is_garment_part(ann, cat_supercat):
                continue
            converted = convert_fashionpedia_annotation(ann, categories, attributes, attr_supercat)
            image_file = img_lookup.get(ann.get("image_id", -1), "")
            image_path = _resolve_image_path(img_root, image_file) if image_file else ""

            entry = {
                "image_path": image_path,
                "attributes_json": json.dumps(converted),
                "source": "fashionpedia",
                "bbox": ann.get("bbox"),
                "image_id": ann.get("image_id"),
                "annotation_id": ann.get("id"),
            }
            fout.write(json.dumps(entry) + "\n")
            count += 1
            if max_samples and count >= max_samples:
                break

    logger.info("Converted %d Fashionpedia annotations → %s", count, output)
    return count
