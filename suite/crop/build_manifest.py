#!/usr/bin/env python3
"""Build a leakage-free, oracle-crop Fashionpedia -> MODA public benchmark.

The official Fashionpedia task combines localization and 294 native attributes.
MODA extracts a smaller product-search schema from an already isolated garment,
so this release deliberately reports a separate *mapped oracle-crop* protocol.
Every mapped field is judged on every garment row: an unsupported prediction is
therefore a false positive instead of being hidden by a positive-only mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]

from suite._eval.fashionpedia_adapter import (  # noqa: E402
    convert_fashionpedia_dataset_linked,
)

BENCHMARK_ID = "fashionpedia-2020-val-moda-oracle-crop-v1"
S3_IMAGE_PREFIX = ""  # not used publicly; images are never redistributed
EVALUATED_FIELDS = (
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flatten_attributes(raw: str | dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(raw) if isinstance(raw, str) else raw
    result: dict[str, Any] = {}
    for field, wrapped in payload.items():
        if field not in EVALUATED_FIELDS:
            continue
        if isinstance(wrapped, dict):
            value = wrapped.get("value")
        elif isinstance(wrapped, list):
            value = [item.get("value") if isinstance(item, dict) else item for item in wrapped]
            value = list(dict.fromkeys(str(item) for item in value if item is not None))
        else:
            value = wrapped
        if value in (None, "", []):
            continue
        result[field] = value
    return result


def _load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def _training_fashionpedia_identity(train_release: Path | None) -> tuple[set[str], set[str]]:
    image_ids: set[str] = set()
    file_names: set[str] = set()
    if train_release is None:
        return image_ids, file_names
    for row in _load_jsonl(train_release):
        provenance = row.get("provenance") or {}
        if provenance.get("source_slice") != "fashionpedia_train":
            continue
        product = provenance.get("source_product") or {}
        image_id = product.get("product_family_id")
        if image_id not in (None, ""):
            image_ids.add(str(image_id))
        source_path = (row.get("image") or {}).get("source_path")
        if source_path:
            file_names.add(Path(str(source_path)).name)
    return image_ids, file_names


def build_benchmark(
    *,
    annotations: Path,
    images_dir: Path,
    output_dir: Path,
    train_release: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="fashionpedia-benchmark-") as temporary:
        converted = Path(temporary) / "linked.jsonl"
        convert_fashionpedia_dataset_linked(
            str(annotations),
            str(converted),
            str(images_dir),
        )
        linked_rows = list(_load_jsonl(converted))

    official = json.loads(annotations.read_text())
    image_by_id = {str(item["id"]): item for item in official.get("images", [])}
    train_image_ids, train_file_names = _training_fashionpedia_identity(train_release)
    benchmark_image_ids = {str(row["image_id"]) for row in linked_rows}
    benchmark_file_names = {
        str(image_by_id[str(row["image_id"])]["file_name"]) for row in linked_rows
    }
    overlap_ids = sorted(benchmark_image_ids & train_image_ids)
    overlap_files = sorted(benchmark_file_names & train_file_names)
    if overlap_ids or overlap_files:
        raise RuntimeError(
            "Fashionpedia benchmark leaks into training: "
            f"image_ids={overlap_ids[:10]}, files={overlap_files[:10]}"
        )

    output_rows: list[dict[str, Any]] = []
    field_support: Counter[str] = Counter()
    value_classes: dict[str, set[str]] = {field: set() for field in EVALUATED_FIELDS}
    for raw in linked_rows:
        image_id = str(raw["image_id"])
        image_info = image_by_id[image_id]
        file_name = str(image_info["file_name"])
        attributes = _flatten_attributes(raw["attributes_json"])
        field_support.update(attributes.keys())
        for field, value in attributes.items():
            values = value if isinstance(value, list) else [value]
            value_classes[field].update(str(item) for item in values)
        output_rows.append(
            {
                "record_id": f"fashionpedia-val:{image_id}:{raw['annotation_id']}",
                "split": "fashionpedia_2020_val",
                "label_tier": "public_annotation_exhaustive_mapped_fields",
                "sample_weight": 1.0,
                "image": {
                    "local_path": str(raw["image_path"]),
                    "s3_uri": f"{S3_IMAGE_PREFIX}/{file_name}",
                    "bbox_xywh": raw.get("bbox"),
                    "image_group_id": f"fashionpedia-val:{image_id}",
                },
                "attributes": attributes,
                "evaluated_attributes": list(EVALUATED_FIELDS),
                "supervised_attributes": sorted(attributes),
                "provenance": {
                    "source": "fashionpedia",
                    "source_split": "official_val2020",
                    "protocol": "oracle_crop_moda_mapping_v1",
                    "image_id": int(image_id),
                    "annotation_id": raw["annotation_id"],
                },
            }
        )

    benchmark_path = output_dir / "benchmark.jsonl"
    with benchmark_path.open("w") as destination:
        for row in output_rows:
            destination.write(json.dumps(row, separators=(",", ":")) + "\n")
    manifest = {
        "benchmark_id": BENCHMARK_ID,
        "protocol": "Official Fashionpedia val2020 garment boxes; MODA mapping; oracle crops",
        "not_comparable_to": "Fashionpedia detection + localized-attribute AP_IoU+F1",
        "annotation_source": str(annotations),
        "annotation_sha256": _sha256(annotations),
        "official_native_attributes": len(official.get("attributes", [])),
        "official_native_categories": len(official.get("categories", [])),
        "official_images": len(official.get("images", [])),
        "garment_instances": len(output_rows),
        "unique_image_groups": len(benchmark_image_ids),
        "evaluated_fields": list(EVALUATED_FIELDS),
        "field_support": dict(sorted(field_support.items())),
        "observed_value_classes": {
            field: len(values) for field, values in sorted(value_classes.items())
        },
        "training_overlap": {
            "checked": train_release is not None,
            "image_id_count": len(overlap_ids),
            "file_name_count": len(overlap_files),
        },
        "artifacts": {
            "benchmark": benchmark_path.name,
            "images_s3_prefix": S3_IMAGE_PREFIX + "/",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=REPO_ROOT / "data/fashionpedia/annotations/instances_attributes_val2020.json",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=REPO_ROOT / "data/fashionpedia/images",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-release", type=Path, default=None)
    args = parser.parse_args()
    manifest = build_benchmark(
        annotations=args.annotations,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        train_release=args.train_release,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
