#!/usr/bin/env python3
"""Build FashionNER benchmark from open-source data (no LLM APIs).

Sources:
  - Livostyle catalog (HF arturayupov/womens-fashion-catalog)
  - Fashionpedia annotations → synthetic attribute-grounded text

Outputs JSONL splits under data/fashion_ner_benchmark/.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ._spans import (
    align_phrases_to_text,
    find_span,
    merge_non_overlapping,
    normalize_text,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "fashion_ner_benchmark"
FASHIONPEDIA_ANN = ROOT / "data" / "fashionpedia" / "annotations" / "instances_attributes_train2020.json"

# Keyword → entity type for Livostyle tags and Fashionpedia attribute names
TAG_KEYWORDS: list[tuple[str, str]] = [
    (r"\bv-neck\b|\bv neck\b", "NECKLINE"),
    (r"\bcrew\b|\bturtleneck\b|\bscoop\b|\bhalter\b|\boff-shoulder\b", "NECKLINE"),
    (r"\blong sleeve\b|\bshort sleeve\b|\bsleeveless\b|\bcap sleeve\b|\bpuff\b", "SLEEVE"),
    (r"\bmidi\b|\bmaxi\b|\bmini\b|\bcropped\b|\bfloor-length\b|\bknee-length\b", "HEMLINE"),
    (r"\boversized\b|\ba-line\b|\bbodycon\b|\brelaxed\b|\bslim\b|\bwide-leg\b", "SILHOUETTE"),
    (r"\bfloral\b|\bstripes?\b|\bplaid\b|\bgingham\b|\bhoundstooth\b|\bprint\b", "PATTERN"),
    (r"\blinen\b|\bsilk\b|\bcotton\b|\bdenim\b|\bviscose\b|\bwool\b|\bknit\b|\bmesh\b|\bleather\b", "MATERIAL"),
    (r"\bbeach\b|\bvacation\b|\bworkwear\b|\bevening\b|\bcasual\b|\bformal\b", "OCCASION"),
    (r"\bboho\b|\bminimal\b|\by2k\b|\bgrunge\b|\bstreetwear\b", "AESTHETIC"),
    (r"\bblack\b|\bwhite\b|\bnavy\b|\bbeige\b|\burgundy\b|\bivory\b|\bgreen\b|\bblue\b|\bpink\b|\bred\b", "COLOR"),
]

FASHIONPEDIA_ATTR_PATTERNS: list[tuple[str, str]] = [
    (r"pattern|plaid|stripe|floral|print|dot", "PATTERN"),
    (r"neck|collar|lapel", "NECKLINE"),
    (r"sleeve", "SLEEVE"),
    (r"length|midi|mini|maxi|hem", "HEMLINE"),
    (r"linen|silk|cotton|wool|denim|leather|chiffon|jersey|knit|woven|textile", "MATERIAL"),
    (r"fit|loose|tight|oversized", "FIT"),
    (r"color|dye", "COLOR"),
    (r"button|zip|pocket|pleat|embroid|ruffle|lace|sequin|fringe", "DETAIL"),
]


def _match_label(phrase: str, patterns: list[tuple[str, str]]) -> str | None:
    pl = phrase.lower()
    for pat, label in patterns:
        if re.search(pat, pl):
            return label
    return None


def _livostyle_phrases(row: dict[str, Any], text: str) -> list[tuple[str, str]]:
    phrases: list[tuple[str, str]] = []
    product_type = str(row.get("product_type") or "").strip()
    if product_type and product_type.lower() not in ("", "uncategorized"):
        if find_span(text, product_type):
            phrases.append((product_type, "GARMENT_TYPE"))
    tags_raw = str(row.get("tags") or "")
    for tag in re.split(r"\s*\|\s*", tags_raw):
        tag = tag.strip()
        if len(tag) < 2 or tag.lower() in ("issues", "ship from overseas"):
            continue
        label = _match_label(tag, TAG_KEYWORDS)
        if label:
            phrases.append((tag, label))
        elif len(tag.split()) <= 3 and tag[0].islower():
            phrases.append((tag, "DETAIL"))
    return phrases


def sample_from_livostyle(limit: int | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("arturayupov/womens-fashion-catalog", "products", split="train")
    rows: list[dict[str, Any]] = []
    n = len(ds) if limit is None else min(limit, len(ds))

    for i in range(n):
        row = ds[i]
        product_id = row.get("id", i)
        tags_raw = str(row.get("tags") or "")
        for field, raw in (
            ("title", str(row.get("title") or "")),
            ("description", str(row.get("description") or "")),
        ):
            text = normalize_text(raw)
            if field == "title" and len(text) < 5:
                continue
            if field == "description" and len(text) < 20:
                continue
            if field == "description":
                text = text[:512]
            phrases = _livostyle_phrases(row, text)
            entities = align_phrases_to_text(text, phrases)
            if not entities:
                continue
            rows.append(
                {
                    "id": f"livostyle_{product_id}_{field}",
                    "text": text,
                    "source": "livostyle",
                    "entities": entities,
                }
            )
    logger.info("Livostyle: %d samples with entities", len(rows))
    return rows


def _garment_phrase(category_name: str) -> str:
    """Use last segment of Fashionpedia category as garment phrase."""
    parts = [p.strip() for p in category_name.split(",")]
    return parts[-1] if parts else category_name


def sample_from_fashionpedia(ann_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not ann_path.exists():
        logger.warning("Fashionpedia annotations missing: %s (run make data-annotations)", ann_path)
        return []

    with open(ann_path) as f:
        data = json.load(f)

    categories = {c["id"]: c["name"] for c in data.get("categories", [])}
    attributes = {a["id"]: a["name"] for a in data.get("attributes", [])}

    rows: list[dict[str, Any]] = []
    count = 0
    for ann in data.get("annotations", []):
        if limit is not None and count >= limit:
            break
        cat_name = categories.get(ann.get("category_id", -1), "")
        if not cat_name:
            continue
        garment = _garment_phrase(cat_name)
        attr_names: list[str] = []
        for aid in ann.get("attribute_ids", []):
            name = attributes.get(aid, "")
            if name:
                attr_names.append(name)

        phrases: list[tuple[str, str]] = [(garment, "GARMENT_TYPE")]
        for attr in attr_names[:6]:
            label = _match_label(attr, FASHIONPEDIA_ATTR_PATTERNS)
            if label:
                short = attr.split(",")[0].strip()[:48]
                phrases.append((short, label))

        # Synthetic caption: attributes + garment (order shuffled for variety)
        parts = [p for p, _ in phrases if p.lower() != garment.lower()]
        random.shuffle(parts)
        text = normalize_text(" ".join(parts + [garment]))
        entities = align_phrases_to_text(text, phrases)
        if len(entities) < 2:
            continue
        rows.append(
            {
                "id": f"fashionpedia_{ann.get('id', count)}",
                "text": text,
                "source": "fashionpedia_synthetic",
                "entities": entities,
            }
        )
        count += 1

    logger.info("Fashionpedia synthetic: %d samples", len(rows))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dedupe_by_text(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One sample per normalized text; prefer livostyle > fashionpedia."""
    priority = {"livostyle": 0, "fashionpedia_synthetic": 1}
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = normalize_text(row["text"]).lower()
        if len(key) < 5:
            continue
        src = row.get("source", "")
        if key not in best or priority.get(src, 9) < priority.get(best[key].get("source", ""), 9):
            best[key] = row
    out = list(best.values())
    logger.info("Deduped %d -> %d unique texts", len(rows), len(out))
    return out


ABSTRACT_TYPES_SPLIT = frozenset({"OCCASION", "AESTHETIC", "FIT", "SILHOUETTE"})
RARE_OVERSAMPLE_TYPES = frozenset({"COLOR", "OCCASION", "AESTHETIC", "SILHOUETTE", "NECKLINE"})


def _row_labels(row: dict[str, Any]) -> set[str]:
    return {e["label"] for e in row.get("entities", []) if e.get("label")}


def _has_abstract(row: dict[str, Any]) -> bool:
    return bool(ABSTRACT_TYPES_SPLIT & _row_labels(row))


def _split_pool(
    pool: list[dict[str, Any]],
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Per-source stratified split for a pool of rows."""
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in pool:
        by_source.setdefault(row.get("source", "unknown"), []).append(row)

    train, dev, test = [], [], []
    for source_rows in by_source.values():
        random.shuffle(source_rows)
        n = len(source_rows)
        if n == 1:
            train.append(source_rows[0])
            continue
        n_test = max(1, int(n * test_ratio))
        n_dev = max(1, int(n * dev_ratio)) if n > 2 else 0
        if n_test + n_dev >= n:
            n_test = 1
            n_dev = 1 if n > 2 else 0
        test.extend(source_rows[:n_test])
        dev.extend(source_rows[n_test : n_test + n_dev])
        train.extend(source_rows[n_test + n_dev :])
    return train, dev, test


def ensure_entity_in_test(
    splits: dict[str, list[dict[str, Any]]],
    *,
    min_test_docs: int = 3,
    min_corpus_docs: int = 12,
) -> dict[str, list[dict[str, Any]]]:
    """Move docs from train→test so rare types appear in test (no text duplicates)."""
    from ._labels import ENTITY_TYPES

    train = list(splits["train"])
    dev = list(splits["dev"])
    test = list(splits["test"])
    test_texts = {normalize_text(r["text"]).lower() for r in test}

    for ent in ENTITY_TYPES:
        test_count = sum(1 for r in test if ent in _row_labels(r))
        need = min_test_docs if sum(
            1 for r in train + test + dev + splits.get("test_hard", []) if ent in _row_labels(r)
        ) >= 50 else 1
        if test_count >= need:
            continue
        corpus_count = sum(
            1 for r in train + test + dev + splits.get("test_hard", [])
            if ent in _row_labels(r)
        )
        if corpus_count < min_corpus_docs:
            continue
        for pool_name, pool in (("train", train), ("dev", dev)):
            candidates = [r for r in pool if ent in _row_labels(r)]
            random.shuffle(candidates)
            for row in candidates:
                if test_count >= need:
                    break
                key = normalize_text(row["text"]).lower()
                if key in test_texts:
                    continue
                pool.remove(row)
                test.append(row)
                test_texts.add(key)
                test_count += 1

    splits["train"] = train
    splits["dev"] = dev
    splits["test"] = test
    return splits


def oversample_rare_train(
    train: list[dict[str, Any]],
    *,
    target_min_docs: int = 40,
) -> list[dict[str, Any]]:
    """Duplicate train docs containing rare types (new ids) to balance head classes."""
    out = list(train)
    for ent in RARE_OVERSAMPLE_TYPES:
        hits = [r for r in train if ent in _row_labels(r)]
        if len(hits) < target_min_docs and hits:
            need = target_min_docs - len(hits)
            for i in range(need):
                src = hits[i % len(hits)]
                dup = {**src, "id": f"{src['id']}_dup_{ent}_{i}"}
                out.append(dup)
    random.shuffle(out)
    logger.info("Train oversample: %d -> %d rows", len(train), len(out))
    return out


def split_dataset(
    rows: list[dict[str, Any]],
    train_ratio: float = 0.75,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.15,
    seed: int = 42,
    hard_max: int = 400,
    min_abstract_in_test: int = 8,
) -> dict[str, list[dict[str, Any]]]:
    """Split with separate abstract pool so test retains abstract gold labels."""
    random.seed(seed)
    abstract_rows = [r for r in rows if _has_abstract(r)]
    other_rows = [r for r in rows if not _has_abstract(r)]

    train_a, dev_a, test_a = _split_pool(abstract_rows, train_ratio, dev_ratio, test_ratio)
    train_o, dev_o, test_o = _split_pool(other_rows, train_ratio, dev_ratio, test_ratio)

    train = train_a + train_o
    dev = dev_a + dev_o
    test = test_a + test_o

    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)

    # test_hard: at most half of abstract-in-test, keep rest in test for headline metrics
    abs_in_test = [r for r in test if _has_abstract(r)]
    random.shuffle(abs_in_test)
    n_move = min(hard_max, max(0, len(abs_in_test) - min_abstract_in_test))
    hard = abs_in_test[:n_move]
    hard_ids = {r["id"] for r in hard}
    test = [r for r in test if r["id"] not in hard_ids]

    splits = {"train": train, "dev": dev, "test": test, "test_hard": hard}
    splits = ensure_entity_in_test(splits)
    splits = _promote_rare_from_hard(splits)
    splits = _force_types_to_test(splits, ("AESTHETIC", "BRAND"), min_test=1)
    splits["train"] = oversample_rare_train(splits["train"])
    return splits


def _force_types_to_test(
    splits: dict[str, list[dict[str, Any]]],
    types: tuple[str, ...],
    min_test: int = 1,
) -> dict[str, list[dict[str, Any]]]:
    test = list(splits["test"])
    test_texts = {normalize_text(r["text"]).lower() for r in test}
    for ent in types:
        have = sum(1 for r in test if ent in _row_labels(r))
        if have >= min_test:
            continue
        for src in ("train", "dev", "test_hard"):
            pool = list(splits[src])
            moved = False
            for row in pool:
                if ent not in _row_labels(row):
                    continue
                key = normalize_text(row["text"]).lower()
                if key in test_texts:
                    continue
                pool.remove(row)
                test.append(row)
                test_texts.add(key)
                moved = True
                break
            splits[src] = pool
            if moved:
                break
    splits["test"] = test
    return splits


def _promote_rare_from_hard(
    splits: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Move docs from test_hard to test when they are the only test exposure for a rare type."""
    from ._labels import ENTITY_TYPES

    hard = list(splits.get("test_hard", []))
    test = list(splits["test"])
    test_texts = {normalize_text(r["text"]).lower() for r in test}

    for ent in ENTITY_TYPES:
        if sum(1 for r in test if ent in _row_labels(r)) >= 1:
            continue
        for row in hard:
            if ent not in _row_labels(row):
                continue
            key = normalize_text(row["text"]).lower()
            if key in test_texts:
                continue
            hard.remove(row)
            test.append(row)
            test_texts.add(key)
            break

    splits["test_hard"] = hard
    splits["test"] = test
    return splits


def verify_splits(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    from ._benchmark_verify import (
        check_test_entity_coverage,
        entity_coverage,
        verify_no_cross_split_leaks,
    )

    leak_report = verify_no_cross_split_leaks(splits)
    if not leak_report["ok"]:
        raise ValueError(
            f"Benchmark split leakage: {leak_report['cross_split_leaks']} texts in multiple splits."
        )
    coverage = entity_coverage(splits)
    coverage_check = check_test_entity_coverage(coverage)
    report = {
        **leak_report,
        "entity_coverage": coverage,
        "test_coverage_ok": coverage_check["ok"],
        "test_coverage_issues": coverage_check.get("issues", []),
        "abstract_types_in_test_docs": coverage_check.get("abstract_types_in_test_docs", 0),
    }
    if coverage_check["issues"]:
        logger.warning("Test coverage gaps: %s", coverage_check["issues"][:5])
    return report


def write_label_schema(out_dir: Path) -> None:
    from ._labels import ENTITY_TYPES, LABEL_LIST

    schema = {
        "entity_types": list(ENTITY_TYPES),
        "bio_labels": LABEL_LIST,
        "annotation": "rule-based silver labels from open sources (no LLM)",
        "sources": ["livostyle", "fashionpedia_synthetic"],
    }
    (out_dir / "label_schema.json").write_text(json.dumps(schema, indent=2))


def write_readme(out_dir: Path, counts: dict[str, int]) -> None:
    readme = f"""# FashionNER Benchmark (open-source silver)

Rule-aligned span labels from public fashion catalogs — **no LLM API** used.

## Splits

| Split | Samples |
|-------|---------|
| train | {counts.get('train', 0)} |
| dev | {counts.get('dev', 0)} |
| test | {counts.get('test', 0)} |
| test_hard | {counts.get('test_hard', 0)} |

## Format (JSONL)

Each line: `id`, `text`, `source`, `entities` (list of `start`, `end`, `text`, `label`).

## Rebuild

```bash
uv run python scripts/build_fashion_ner_benchmark.py
python -m suite.text.build_splits --fetch-annotations  # Fashionpedia JSON only
```
"""
    (out_dir / "README.md").write_text(readme)


def download_fashionpedia_annotations() -> None:
    ann_dir = FASHIONPEDIA_ANN.parent
    ann_dir.mkdir(parents=True, exist_ok=True)
    url = (
        "https://s3.amazonaws.com/ifashionist-dataset/annotations/"
        "instances_attributes_train2020.json"
    )
    dest = ann_dir / "instances_attributes_train2020.json"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return
    logger.info("Downloading Fashionpedia annotations...")
    subprocess.run(
        ["curl", "-L", "--fail", "-o", str(dest), url],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FashionNER open benchmark")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--livostyle-limit", type=int, default=None)
    parser.add_argument("--fashionpedia-limit", type=int, default=12000)
    parser.add_argument("--download-fp", action="store_true")
    parser.add_argument("--skip-livostyle", action="store_true")
    parser.add_argument("--skip-fashionpedia", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.download_fp:
        download_fashionpedia_annotations()

    all_rows: list[dict[str, Any]] = []
    if not args.skip_livostyle:
        all_rows.extend(sample_from_livostyle(args.livostyle_limit))
    if not args.skip_fashionpedia:
        all_rows.extend(sample_from_fashionpedia(FASHIONPEDIA_ANN, args.fashionpedia_limit))

    if not all_rows:
        raise SystemExit(
            "No benchmark samples built. Run with --download-fp or python -m suite.text.build_splits --fetch-annotations"
        )

    all_rows = dedupe_by_text(all_rows)
    splits = split_dataset(all_rows, seed=args.seed)
    split_report = verify_splits(splits)
    counts: dict[str, int] = {}
    for name, part in splits.items():
        path = args.out_dir / f"{name}.jsonl"
        write_jsonl(path, part)
        counts[name] = len(part)
        logger.info("Wrote %s (%d)", path, len(part))

    write_label_schema(args.out_dir)
    write_readme(args.out_dir, counts)
    manifest = {
        "counts": counts,
        "total": sum(counts.values()),
        "seed": args.seed,
        "deduped": True,
        "split_verification": split_report,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Done. Total samples: %d", manifest["total"])


if __name__ == "__main__":
    main()
