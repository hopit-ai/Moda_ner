"""Reproducible FashionCLIP linear baseline utilities for DFMM.

The WACVW 2026 paper uses frozen FashionCLIP embeddings and one class-balanced
logistic-regression classifier per attribute.  This module contains the pieces
that must be shared by the cost-gated Modal runner and its local tests.  It does
not download a model or launch compute.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .dfmm import ATTRIBUTE_NAMES, ATTRIBUTE_SPECS, score_three_tier

THREE_TIER_METRICS = (
    "tier1_macro_f1",
    "tier2_na_f1",
    "tier3_visible_macro_f1",
)

_PRODUCT_GROUP_RE = re.compile(
    r"^(?P<gender>MEN|WOMEN)-(?P<category>.+?)-(?P<product>id_\d+)-"
    r"(?P<shot>\d+)_(?P<sequence>\d+)_(?P<view>[^./]+)\.jpg$"
)


def product_group_id(record_id: str) -> str:
    """Return the DFMM product identity shared by all views of one product.

    DFMM record IDs are source filenames such as
    ``WOMEN-Dresses-id_00007176-01_4_full.jpg``.  Everything through the
    ``gender-category-id_N`` prefix is the product identity; the remaining
    suffix identifies a particular image/view.
    """
    match = _PRODUCT_GROUP_RE.fullmatch(str(record_id))
    if match is None:
        raise ValueError(f"Unsupported DFMM product record_id: {record_id!r}")
    return "-".join(
        (match.group("gender"), match.group("category"), match.group("product"))
    )


def partition_product_identity(
    training_record_ids: Sequence[str],
    test_record_ids: Sequence[str],
) -> dict[str, Any]:
    """Partition frozen test IDs into seen- and novel-product cohorts."""
    training_ids = [str(value) for value in training_record_ids]
    frozen_test_ids = [str(value) for value in test_record_ids]
    if not training_ids or not frozen_test_ids:
        raise ValueError("Training and test record IDs must both be non-empty")
    if len(set(training_ids)) != len(training_ids):
        raise ValueError("Duplicate training record IDs in product audit")
    if len(set(frozen_test_ids)) != len(frozen_test_ids):
        raise ValueError("Duplicate test record IDs in product audit")
    record_overlap = set(training_ids) & set(frozen_test_ids)
    if record_overlap:
        raise ValueError("Training and test record IDs overlap in product audit")

    training_groups = {product_group_id(record_id) for record_id in training_ids}
    assignments = []
    seen_record_ids = []
    novel_record_ids = []
    for record_id in frozen_test_ids:
        group_id = product_group_id(record_id)
        cohort = "seen_product" if group_id in training_groups else "novel_product"
        assignments.append(
            {
                "record_id": record_id,
                "product_group_id": group_id,
                "cohort": cohort,
            }
        )
        if cohort == "seen_product":
            seen_record_ids.append(record_id)
        else:
            novel_record_ids.append(record_id)

    seen_groups = {
        row["product_group_id"] for row in assignments if row["cohort"] == "seen_product"
    }
    novel_groups = {
        row["product_group_id"] for row in assignments if row["cohort"] == "novel_product"
    }
    return {
        "training_product_groups": sorted(training_groups),
        "seen_product_groups": sorted(seen_groups),
        "novel_product_groups": sorted(novel_groups),
        "seen_record_ids": seen_record_ids,
        "novel_record_ids": novel_record_ids,
        "assignments": assignments,
        "novel_training_group_overlap": sorted(novel_groups & training_groups),
    }


def fixed_vocabulary_macro_f1(
    estimator: Any,
    features: Any,
    gold: Sequence[str],
    *,
    vocabulary: Sequence[str],
    zero_division: int = 0,
) -> float:
    """Score every frozen class, including classes absent from a CV fold.

    sklearn 1.5.2's ``make_scorer(f1_score, labels=...)`` incorrectly enters
    its binary ``pos_label=1`` validation path for these string-valued
    multiclass targets. GridSearchCV also accepts this direct estimator scorer,
    which preserves the intended fixed-vocabulary macro-F1 definition.
    """
    from sklearn.metrics import f1_score

    predicted = estimator.predict(features)
    return float(
        f1_score(
            gold,
            predicted,
            average="macro",
            labels=list(vocabulary),
            zero_division=zero_division,
        )
    )


def requested_resource_cost_usd(
    *,
    timeout_seconds: int,
    gpu_rate_per_second: float,
    cpu_cores: float,
    cpu_rate_per_core_second: float,
    memory_gib: float,
    memory_rate_per_gib_second: float,
) -> float:
    """Return the fail-closed cost ceiling for a single Modal container."""
    if timeout_seconds <= 0 or cpu_cores <= 0 or memory_gib <= 0:
        raise ValueError("Timeout, CPU, and memory must be positive")
    rates = (
        gpu_rate_per_second,
        cpu_rate_per_core_second,
        memory_rate_per_gib_second,
    )
    if any(rate < 0 for rate in rates):
        raise ValueError("Resource rates cannot be negative")
    per_second = (
        gpu_rate_per_second
        + cpu_cores * cpu_rate_per_core_second
        + memory_gib * memory_rate_per_gib_second
    )
    return timeout_seconds * per_second


def validate_baseline_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_rows: int,
) -> dict[str, Any]:
    """Validate the exact inputs required by the frozen FashionCLIP baseline."""
    if len(rows) != expected_rows:
        raise ValueError(f"Unexpected rows: {len(rows)} != {expected_rows}")
    record_ids: list[str] = []
    label_support = {spec.name: {value: 0 for value in spec.values} for spec in ATTRIBUTE_SPECS}
    for row in rows:
        record_id = str(row.get("record_id") or "")
        if not record_id:
            raise ValueError("Every baseline row requires record_id")
        record_ids.append(record_id)
        description = row.get("text_description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Row {record_id} lacks text_description")
        image = row.get("image")
        if not isinstance(image, Mapping) or not str(image.get("s3_uri", "")).startswith("s3://"):
            raise ValueError(f"Row {record_id} lacks an S3 image URI")
        labels = row.get("dfmm_labels")
        if not isinstance(labels, Mapping) or set(labels) != set(ATTRIBUTE_NAMES):
            raise ValueError(f"Row {record_id} does not have exactly 18 DFMM labels")
        for spec in ATTRIBUTE_SPECS:
            value = str(labels[spec.name])
            if value not in spec.values:
                raise ValueError(f"Row {record_id} has invalid {spec.name}={value!r}")
            label_support[spec.name][value] += 1
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("Duplicate record IDs in FashionCLIP baseline rows")
    return {
        "rows": len(rows),
        "record_ids_sha256": hashlib.sha256("\n".join(record_ids).encode()).hexdigest(),
        "label_support": label_support,
    }


def build_prediction_rows(
    record_ids: Sequence[str],
    predictions_by_attribute: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    """Build scorer-ready dense prediction rows and reject schema drift."""
    if set(predictions_by_attribute) != set(ATTRIBUTE_NAMES):
        raise ValueError("Predictions must contain exactly the 18 DFMM attributes")
    rows: list[dict[str, Any]] = []
    for index, record_id in enumerate(record_ids):
        attributes: dict[str, str] = {}
        for spec in ATTRIBUTE_SPECS:
            values = predictions_by_attribute[spec.name]
            if len(values) != len(record_ids):
                raise ValueError(f"Prediction length mismatch for {spec.name}")
            value = str(values[index])
            if value not in spec.values:
                raise ValueError(f"Out-of-vocabulary prediction {spec.name}={value!r}")
            attributes[spec.name] = value
        rows.append({"record_id": str(record_id), "attributes": attributes})
    return rows


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_manifest(root: str | Path, *, exclude: Sequence[str] = ()) -> dict[str, Any]:
    """Hash every file below ``root`` for an auditable S3 release."""
    base = Path(root)
    excluded = set(exclude)
    files = []
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        relative = path.relative_to(base).as_posix()
        if relative in excluded:
            continue
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(row["bytes"]) for row in files),
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _prediction_source(row: Mapping[str, Any]) -> Mapping[str, Any]:
    source = row.get("predictions") or row.get("attributes")
    if not isinstance(source, Mapping):
        raise ValueError("Prediction row lacks a predictions/attributes mapping")
    return source


def _confusion_contributions(
    gold_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> tuple[Any, list[tuple[int, int, int | None]]]:
    """Encode per-row confusion contributions for fast exact bootstrapping."""
    import torch

    predictions = {
        str(row.get("record_id") or row.get("image_id") or ""): _prediction_source(row)
        for row in prediction_rows
    }
    gold_ids = [str(row.get("record_id") or row.get("image_id") or "") for row in gold_rows]
    if not all(gold_ids) or set(predictions) != set(gold_ids) or len(predictions) != len(gold_ids):
        raise ValueError("Prediction IDs must exactly match unique gold IDs")
    layout: list[tuple[int, int, int | None]] = []
    offset = 0
    for spec in ATTRIBUTE_SPECS:
        na_index = spec.values.index("NA") if spec.has_na else None
        layout.append((offset, len(spec.values), na_index))
        offset += len(spec.values) ** 2
    contributions = torch.zeros((len(gold_rows), offset), dtype=torch.float32)
    for row_index, (record_id, row) in enumerate(zip(gold_ids, gold_rows, strict=True)):
        labels = row.get("dfmm_labels") or row.get("labels")
        if not isinstance(labels, Mapping):
            raise ValueError(f"Gold row {record_id} lacks dfmm_labels")
        predicted = predictions[record_id]
        for spec, (start, width, _) in zip(ATTRIBUTE_SPECS, layout, strict=True):
            gold_value = str(labels.get(spec.name))
            predicted_value = str(predicted.get(spec.name))
            if gold_value not in spec.values or predicted_value not in spec.values:
                raise ValueError(f"Invalid gold/predicted value for {record_id}:{spec.name}")
            gold_index = spec.values.index(gold_value)
            predicted_index = spec.values.index(predicted_value)
            contributions[row_index, start + gold_index * width + predicted_index] = 1.0
    return contributions, layout


def _safe_f1(tp: Any, fp: Any, fn: Any) -> Any:
    import torch

    denominator = 2 * tp + fp + fn
    return torch.where(denominator > 0, 2 * tp / denominator, torch.zeros_like(tp))


def _metrics_from_confusions(confusions: Any, layout: Sequence[tuple[int, int, int | None]]) -> Any:
    """Vectorize the repository's three-tier metric over bootstrap samples."""
    import torch

    tier1_attributes = []
    tier2_attributes = []
    tier3_attributes = []
    for start, width, na_index in layout:
        confusion = confusions[:, start : start + width * width].reshape(-1, width, width)
        tp = torch.diagonal(confusion, dim1=1, dim2=2)
        fp = confusion.sum(dim=1) - tp
        fn = confusion.sum(dim=2) - tp
        tier1 = _safe_f1(tp, fp, fn).mean(dim=1)
        tier1_attributes.append(tier1)
        if na_index is None:
            tier3_attributes.append(tier1)
            continue
        na_tp = confusion[:, na_index, na_index]
        na_fp = confusion[:, :, na_index].sum(dim=1) - na_tp
        na_fn = confusion[:, na_index, :].sum(dim=1) - na_tp
        tier2_attributes.append(_safe_f1(na_tp, na_fp, na_fn))
        visible = confusion.clone()
        visible[:, na_index, :] = 0
        visible_tp = torch.diagonal(visible, dim1=1, dim2=2)
        visible_fp = visible.sum(dim=1) - visible_tp
        visible_fn = visible.sum(dim=2) - visible_tp
        tier3_attributes.append(_safe_f1(visible_tp, visible_fp, visible_fn).mean(dim=1))
    return torch.stack(
        (
            torch.stack(tier1_attributes, dim=1).mean(dim=1),
            torch.stack(tier2_attributes, dim=1).mean(dim=1),
            torch.stack(tier3_attributes, dim=1).mean(dim=1),
        ),
        dim=1,
    )


def paired_bootstrap_three_tier(
    gold_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    comparator_rows: Sequence[Mapping[str, Any]],
    *,
    reference_name: str,
    comparator_name: str,
    iterations: int = 10_000,
    seed: int = 20260808,
    confidence: float = 0.95,
    chunk_size: int = 256,
    device: str | None = None,
) -> dict[str, Any]:
    """Run a deterministic paired row bootstrap with exact three-tier metrics.

    NumPy generates the resampled row indices on CPU from a pinned seed.  Torch
    only aggregates those fixed samples, allowing the same function to use the
    Modal GPU for 10,000 iterations and CPU for focused unit tests.
    """
    import numpy as np
    import torch

    if not gold_rows or iterations <= 0 or chunk_size <= 0:
        raise ValueError("Gold rows, iterations, and chunk size must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    reference_report = score_three_tier(gold_rows, reference_rows)
    comparator_report = score_three_tier(gold_rows, comparator_rows)
    for name, report in ((reference_name, reference_report), (comparator_name, comparator_report)):
        schema = report["schema"]
        if any(
            (
                schema["missing_record_ids"],
                schema["extra_record_ids"],
                schema["missing_fields"],
                schema["hallucinations"],
                schema["unknown_fields_in_nested_sections"],
            )
        ):
            raise ValueError(f"Cannot bootstrap invalid {name} predictions: {schema}")
    reference_contributions, layout = _confusion_contributions(gold_rows, reference_rows)
    comparator_contributions, comparator_layout = _confusion_contributions(
        gold_rows, comparator_rows
    )
    if comparator_layout != layout:
        raise RuntimeError("Reference and comparator confusion layouts differ")
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if selected_device.startswith("cuda"):
        # Bootstrap aggregation is integer counting expressed as float32 GEMM.
        # Disabling TF32 keeps every count exact for this 5K-row benchmark.
        torch.backends.cuda.matmul.allow_tf32 = False
    combined = torch.cat((reference_contributions, comparator_contributions), dim=1).to(
        selected_device
    )
    width = reference_contributions.shape[1]
    row_count = len(gold_rows)
    reference_samples = np.empty((iterations, len(THREE_TIER_METRICS)), dtype=np.float32)
    comparator_samples = np.empty_like(reference_samples)
    rng = np.random.default_rng(seed)
    cursor = 0
    while cursor < iterations:
        size = min(chunk_size, iterations - cursor)
        sampled = rng.integers(0, row_count, size=(size, row_count), dtype=np.int32)
        sampled_tensor = torch.from_numpy(sampled).to(selected_device, dtype=torch.long)
        weights = torch.zeros((size, row_count), dtype=torch.float32, device=selected_device)
        weights.scatter_add_(
            1,
            sampled_tensor,
            torch.ones_like(sampled_tensor, dtype=torch.float32),
        )
        aggregated = weights @ combined
        reference_metrics = _metrics_from_confusions(aggregated[:, :width], layout)
        comparator_metrics = _metrics_from_confusions(aggregated[:, width:], layout)
        reference_samples[cursor : cursor + size] = reference_metrics.cpu().numpy()
        comparator_samples[cursor : cursor + size] = comparator_metrics.cpu().numpy()
        cursor += size
    delta_samples = reference_samples - comparator_samples
    alpha = (1.0 - confidence) / 2.0

    def point(report: Mapping[str, Any]) -> dict[str, float]:
        values = report["model_level"]
        return {metric: float(values[metric]) for metric in THREE_TIER_METRICS}

    reference_point = point(reference_report)
    comparator_point = point(comparator_report)
    delta_point = {
        metric: round(reference_point[metric] - comparator_point[metric], 6)
        for metric in THREE_TIER_METRICS
    }

    def intervals(samples: Any) -> dict[str, dict[str, float]]:
        return {
            metric: {
                "lower": round(float(np.quantile(samples[:, index], alpha)), 6),
                "upper": round(float(np.quantile(samples[:, index], 1.0 - alpha)), 6),
            }
            for index, metric in enumerate(THREE_TIER_METRICS)
        }

    return {
        "method": "paired_nonparametric_row_bootstrap_percentile",
        "rows": row_count,
        "iterations": iterations,
        "seed": seed,
        "confidence": confidence,
        "sampling_backend": "numpy.PCG64",
        "aggregation_device": selected_device,
        "tf32_disabled_for_exact_counts": selected_device.startswith("cuda"),
        "reference": reference_name,
        "comparator": comparator_name,
        "point_estimates": {
            reference_name: reference_point,
            comparator_name: comparator_point,
            "delta_reference_minus_comparator": delta_point,
        },
        "confidence_intervals": {
            reference_name: intervals(reference_samples),
            comparator_name: intervals(comparator_samples),
            "delta_reference_minus_comparator": intervals(delta_samples),
        },
        "reference_win_probability": {
            metric: round(float((delta_samples[:, index] > 0).mean()), 6)
            for index, metric in enumerate(THREE_TIER_METRICS)
        },
    }


def paired_cluster_bootstrap_three_tier(
    gold_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    comparator_rows: Sequence[Mapping[str, Any]],
    *,
    cluster_ids: Sequence[str],
    reference_name: str,
    comparator_name: str,
    iterations: int = 10_000,
    seed: int = 20260808,
    confidence: float = 0.95,
    chunk_size: int = 256,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run a paired product-cluster bootstrap over exact three-tier metrics.

    All views belonging to one product are summed before products are sampled
    with replacement. This preserves within-product dependence and avoids the
    falsely narrow intervals produced by treating garment views as independent
    rows.
    """
    import numpy as np
    import torch

    if not gold_rows or iterations <= 0 or chunk_size <= 0:
        raise ValueError("Gold rows, iterations, and chunk size must be positive")
    if len(cluster_ids) != len(gold_rows):
        raise ValueError("cluster_ids must align one-to-one with gold rows")
    if any(not str(cluster_id) for cluster_id in cluster_ids):
        raise ValueError("Every bootstrap row requires a non-empty cluster ID")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False

    reference_report = score_three_tier(gold_rows, reference_rows)
    comparator_report = score_three_tier(gold_rows, comparator_rows)
    for name, report in ((reference_name, reference_report), (comparator_name, comparator_report)):
        schema = report["schema"]
        if any(
            (
                schema["missing_record_ids"],
                schema["extra_record_ids"],
                schema["missing_fields"],
                schema["hallucinations"],
                schema["unknown_fields_in_nested_sections"],
            )
        ):
            raise ValueError(f"Cannot bootstrap invalid {name} predictions: {schema}")

    reference_contributions, layout = _confusion_contributions(gold_rows, reference_rows)
    comparator_contributions, comparator_layout = _confusion_contributions(
        gold_rows, comparator_rows
    )
    if comparator_layout != layout:
        raise RuntimeError("Reference and comparator confusion layouts differ")

    clusters = sorted({str(cluster_id) for cluster_id in cluster_ids})
    cluster_lookup = {cluster_id: index for index, cluster_id in enumerate(clusters)}
    row_cluster_indices = torch.tensor(
        [cluster_lookup[str(cluster_id)] for cluster_id in cluster_ids], dtype=torch.long
    )
    combined_rows = torch.cat((reference_contributions, comparator_contributions), dim=1)
    combined_clusters = torch.zeros(
        (len(clusters), combined_rows.shape[1]), dtype=torch.float32
    )
    combined_clusters.index_add_(0, row_cluster_indices, combined_rows)
    combined_clusters = combined_clusters.to(device)
    width = reference_contributions.shape[1]

    reference_samples = np.empty((iterations, len(THREE_TIER_METRICS)), dtype=np.float32)
    comparator_samples = np.empty_like(reference_samples)
    rng = np.random.default_rng(seed)
    cursor = 0
    while cursor < iterations:
        size = min(chunk_size, iterations - cursor)
        sampled = rng.integers(
            0,
            len(clusters),
            size=(size, len(clusters)),
            dtype=np.int32,
        )
        sampled_tensor = torch.from_numpy(sampled).to(device, dtype=torch.long)
        weights = torch.zeros(
            (size, len(clusters)), dtype=torch.float32, device=device
        )
        weights.scatter_add_(
            1,
            sampled_tensor,
            torch.ones_like(sampled_tensor, dtype=torch.float32),
        )
        aggregated = weights @ combined_clusters
        reference_metrics = _metrics_from_confusions(aggregated[:, :width], layout)
        comparator_metrics = _metrics_from_confusions(aggregated[:, width:], layout)
        reference_samples[cursor : cursor + size] = reference_metrics.cpu().numpy()
        comparator_samples[cursor : cursor + size] = comparator_metrics.cpu().numpy()
        cursor += size

    delta_samples = reference_samples - comparator_samples
    alpha = (1.0 - confidence) / 2.0

    def point(report: Mapping[str, Any]) -> dict[str, float]:
        values = report["model_level"]
        return {metric: float(values[metric]) for metric in THREE_TIER_METRICS}

    def intervals(samples: Any) -> dict[str, dict[str, float]]:
        return {
            metric: {
                "lower": round(float(np.quantile(samples[:, index], alpha)), 6),
                "upper": round(float(np.quantile(samples[:, index], 1.0 - alpha)), 6),
            }
            for index, metric in enumerate(THREE_TIER_METRICS)
        }

    reference_point = point(reference_report)
    comparator_point = point(comparator_report)
    delta_point = {
        metric: round(reference_point[metric] - comparator_point[metric], 6)
        for metric in THREE_TIER_METRICS
    }
    return {
        "method": "paired_nonparametric_product_cluster_bootstrap_percentile",
        "rows": len(gold_rows),
        "clusters": len(clusters),
        "sampling_unit": "product_group_id",
        "cluster_order": "lexicographic",
        "iterations": iterations,
        "seed": seed,
        "confidence": confidence,
        "sampling_backend": "numpy.PCG64",
        "aggregation_device": device,
        "tf32_disabled_for_exact_counts": device.startswith("cuda"),
        "reference": reference_name,
        "comparator": comparator_name,
        "point_estimates": {
            reference_name: reference_point,
            comparator_name: comparator_point,
            "delta_reference_minus_comparator": delta_point,
        },
        "confidence_intervals": {
            reference_name: intervals(reference_samples),
            comparator_name: intervals(comparator_samples),
            "delta_reference_minus_comparator": intervals(delta_samples),
        },
        "reference_win_probability": {
            metric: round(float((delta_samples[:, index] > 0).mean()), 6)
            for index, metric in enumerate(THREE_TIER_METRICS)
        },
    }
