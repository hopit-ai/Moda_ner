"""Benchmark integrity checks: leaks, coverage, eval/sample alignment."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from ._labels import ABSTRACT_TYPES, ENTITY_TYPES
from ._spans import normalize_text


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_no_cross_split_leaks(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    seen: dict[str, str] = {}
    leaks: list[dict[str, str]] = []
    for split_name, part in splits.items():
        for row in part:
            key = normalize_text(row["text"]).lower()
            if len(key) < 5:
                continue
            if key in seen and seen[key] != split_name:
                leaks.append({"text": key[:80], "split_a": seen[key], "split_b": split_name})
            else:
                seen[key] = split_name
    return {
        "unique_texts": len(seen),
        "cross_split_leaks": len(leaks),
        "leak_examples": leaks[:10],
        "ok": len(leaks) == 0,
    }


def entity_coverage(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Per-split entity span counts and doc counts per type."""
    report: dict[str, Any] = {}
    for split_name, part in splits.items():
        span_counts: Counter[str] = Counter()
        doc_counts: Counter[str] = Counter()
        for row in part:
            labels_in_doc: set[str] = set()
            for e in row.get("entities", []):
                lab = e.get("label") or e.get("type")
                if lab:
                    span_counts[lab] += 1
                    labels_in_doc.add(lab)
            for lab in labels_in_doc:
                doc_counts[lab] += 1
        report[split_name] = {
            "n_docs": len(part),
            "span_counts": dict(span_counts),
            "doc_counts": dict(doc_counts),
        }
    return report


def check_test_entity_coverage(
    coverage: dict[str, Any],
    *,
    min_docs: int = 3,
    min_corpus_docs: int = 15,
) -> dict[str, Any]:
    """Flag entity types absent from test but present enough in full corpus."""
    issues: list[str] = []
    all_docs: Counter[str] = Counter()
    test_docs: Counter[str] = Counter()
    for split_name, data in coverage.items():
        for ent, n in data.get("doc_counts", {}).items():
            if split_name == "test":
                test_docs[ent] = n
            if split_name != "test_hard":
                all_docs[ent] += n
    for ent in ENTITY_TYPES:
        corpus_n = all_docs.get(ent, 0)
        test_n = test_docs.get(ent, 0)
        required_docs = min_docs if corpus_n >= 50 else (1 if corpus_n >= 10 else 0)
        if required_docs and corpus_n >= min_corpus_docs and test_n < required_docs:
            issues.append(
                f"{ent}: {corpus_n} docs in corpus but only {test_n} in test (need>={required_docs})"
            )
    abstract_in_test = sum(test_docs.get(t, 0) for t in ABSTRACT_TYPES)
    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "abstract_types_in_test_docs": abstract_in_test,
        "test_doc_counts": dict(test_docs),
    }


def verify_eval_sample_counts(
    bench_dir: Path,
    results_dir: Path,
) -> dict[str, Any]:
    """Ensure eval JSON n_samples matches jsonl line counts."""
    issues: list[str] = []
    checks: dict[str, Any] = {}
    mapping = {
        "test": "results_modernbert_test.json",
        "dev": "results_modernbert_dev.json",
        "test_hard": "results_modernbert_test_hard.json",
    }
    for split, result_name in mapping.items():
        jsonl = bench_dir / f"{split}.jsonl"
        result_path = results_dir / result_name
        expected = len(load_jsonl(jsonl))
        if not result_path.exists():
            issues.append(f"missing {result_name}")
            continue
        data = json.loads(result_path.read_text())
        got = data.get("n_samples")
        checks[split] = {"expected": expected, "reported": got, "ok": got == expected}
        if got != expected:
            issues.append(f"{split}: n_samples={got} != jsonl lines={expected}")
    return {"ok": not issues, "issues": issues, "splits": checks}
