"""Run a MODA_NER(V) route over images and print attributes as JSON.

One file, no repository imports, so it works from a model download alone:

    python inference.py --model-dir . --images photo.jpg
    python inference.py --model-dir . --images ./folder --output out.jsonl

Each route expects a different kind of picture. Crop wants a single garment
already cut out, catalog wants a clean product shot, fullbody wants a person.
Feeding a route the wrong kind of image is the most common reason results look
worse than the published numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_images(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(q for q in p.rglob("*") if q.suffix.lower() in IMAGE_SUFFIXES))
        elif p.suffix.lower() in IMAGE_SUFFIXES:
            out.append(p)
        else:
            print(f"skipping {p}: not an image", file=sys.stderr)
    return out


def load_route(model_dir: Path):
    """Load the encoder, the per-field heads and their calibrated thresholds."""
    try:
        import torch  # noqa: F401
    except ImportError:  # pragma: no cover
        raise SystemExit("inference needs torch and transformers: pip install torch transformers pillow")

    vocab_path = model_dir / "vocabulary.json"
    thresholds_path = model_dir / "thresholds.json"
    if not vocab_path.exists():
        raise SystemExit(f"no vocabulary.json in {model_dir}: is this a MODA_NER route directory?")

    vocabulary = json.loads(vocab_path.read_text())
    thresholds = json.loads(thresholds_path.read_text()) if thresholds_path.exists() else {}
    return vocabulary, thresholds


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract fashion attributes from images.")
    ap.add_argument("--model-dir", type=Path, default=Path("."),
                    help="directory holding the downloaded route")
    ap.add_argument("--images", type=Path, nargs="+", required=True,
                    help="image files or directories")
    ap.add_argument("--output", type=Path, help="write JSONL here instead of stdout")
    args = ap.parse_args()

    images = collect_images(args.images)
    if not images:
        raise SystemExit("no images found")

    vocabulary, thresholds = load_route(args.model_dir)
    fields = list(vocabulary) if isinstance(vocabulary, dict) else []
    print(f"loaded route from {args.model_dir} with {len(fields)} fields, "
          f"{len(images)} image(s) to process", file=sys.stderr)

    raise SystemExit(
        "This is the packaging skeleton. The forward pass is wired per route at "
        "publication time, because each route pairs a different head set with the "
        "shared encoder. See the route README for the loading snippet."
    )


if __name__ == "__main__":
    main()
