"""Score exact-span predictions against the text track gold.

Strict span match: a prediction counts only when its character offsets and its
entity type both match a gold span exactly. Partial overlap scores nothing,
which is the point — a span that is nearly right is not usable downstream.

Dependency-free by design, so anyone can run it.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .. import banner, markdown_footer, stamp


def _spans(row: dict) -> set[tuple]:
    out = set()
    for ent in row.get("entities", []) or []:
        out.add((int(ent["start"]), int(ent["end"]), str(ent["label"])))
    return out


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score(gold_path: Path, pred_path: Path) -> dict:
    gold = {}
    for line in gold_path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            gold[row["id"]] = row

    preds = {}
    for line in pred_path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            preds[row["id"]] = row

    missing = sorted(set(gold) - set(preds))
    extra = sorted(set(preds) - set(gold))
    if missing:
        raise SystemExit(f"scorer failed closed: {len(missing)} gold rows have no prediction "
                         f"(first: {missing[:3]})")
    if extra:
        raise SystemExit(f"scorer failed closed: {len(extra)} predicted ids are not in the gold "
                         f"(first: {extra[:3]})")

    tp = fp = fn = 0
    per_entity = defaultdict(lambda: [0, 0, 0])
    for rid, grow in gold.items():
        g, p = _spans(grow), _spans(preds[rid])
        for span in g & p:
            tp += 1; per_entity[span[2]][0] += 1
        for span in p - g:
            fp += 1; per_entity[span[2]][1] += 1
        for span in g - p:
            fn += 1; per_entity[span[2]][2] += 1

    precision, recall, f1 = _prf(tp, fp, fn)
    entities = {}
    for label, (a, b, c) in sorted(per_entity.items()):
        ep, er, ef = _prf(a, b, c)
        entities[label] = {"precision": round(ep, 6), "recall": round(er, 6),
                           "f1": round(ef, 6), "gold_spans": a + c}

    return {
        "rows": len(gold),
        "strict_span": {"precision": round(precision, 6), "recall": round(recall, 6),
                        "f1": round(f1, 6), "tp": tp, "fp": fp, "fn": fn},
        "per_entity": entities,
        "label_note": "Labels are cleaned silver, rule-derived and already opened during "
                      "development. Not independent human gold.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Score span predictions on the text track.")
    ap.add_argument("--gold", required=True, type=Path)
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    print(banner("text"), file=__import__("sys").stderr)
    result = stamp(score(args.gold, args.predictions), "text")
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    print(markdown_footer("text"), file=__import__("sys").stderr)


if __name__ == "__main__":
    main()
