"""Extract fashion attributes from images with a published MODA_NER(V) route.

    python inference.py --route crop     --model-dir . --images photo.jpg
    python inference.py --route catalog  --model-dir . --images ./folder --output out.jsonl
    python inference.py --route fullbody --model-dir . --images ./folder

You choose the route, because each one expects a different kind of picture and
feeding the wrong kind is the most common reason results look worse than the
published numbers. Crop wants a single garment already cut out. Catalog wants a
clean product shot. Fullbody wants a person.

Needs: torch, open_clip_torch, safetensors, joblib, numpy, pillow.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Running this file directly puts models/ on sys.path, not the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract fashion attributes from images.")
    ap.add_argument("--route", required=True, choices=["crop", "catalog", "fullbody"])
    ap.add_argument("--model-dir", type=Path, default=Path("."))
    ap.add_argument("--images", type=Path, nargs="+", required=True)
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    images = collect_images(args.images)
    if not images:
        raise SystemExit("no images found")

    try:
        from suite._model import ROUTES
        from suite._model.routes import LocalRoute
    except ImportError:
        raise SystemExit(
            "run this from a checkout of the Moda_ner repository, or install its "
            "requirements first: pip install -r requirements-inference.txt"
        )

    backend_cls = ROUTES[args.route]
    print(f"loading {args.route} route from {args.model_dir}", file=sys.stderr)
    backend = backend_cls(
        LocalRoute(args.model_dir, backend_cls.package_dirname), args.device
    )

    sink = args.output.open("w") if args.output else sys.stdout
    try:
        for path in images:
            record = {"image": str(path), "route": args.route,
                      "attributes": backend.predict(path)}
            sink.write(json.dumps(record) + "\n")
    finally:
        if args.output:
            sink.close()
            print(f"wrote {len(images)} rows to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
