"""How much of the crop-track score is Fashionpedia's annotation sparsity?

The crop protocol is exhaustive: every one of the fifteen mapped fields is judged on every
garment, and a field with no gold value counts as absent, so predicting it is a false positive.
Fashionpedia annotates only about a third of those (garment, field) cells. Where a garment
really has the attribute but nobody labelled it, a correct prediction is scored as an error.

This rescores each system two ways with the suite's own point-metric code:

  exhaustive       the published protocol (suite.crop.score, unmodified)
  annotated-only   each garment judged only on the fields its source annotation covers

Annotated-only also forgives genuine hallucinations on fields that truly do not apply, so it is
an UPPER bound on what sparsity costs, not an estimate of the true score. The truth sits between
the two numbers.

Guard: the masked function with the mask switched off must reproduce the published protocol
exactly for every system, which proves the mask is the only difference.

Section 6 of the resource report, arXiv:2609.13279 v2.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite-root", type=Path, required=True)
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--system", action="append", required=True, help="name=path/to/predictions.jsonl")
    ap.add_argument("--bootstrap-samples", type=int, default=10000)  # suite protocol
    ap.add_argument("--seed", type=int, default=42)                  # scorer default
    args = ap.parse_args()

    sys.path.insert(0, str(args.suite_root))
    from suite.crop import score as S  # the unmodified suite scorer  # noqa: E402

    def masked_group_stats(gold_rows, predictions, *, mask: bool):
        """Copy of suite.crop.score._build_group_stats. The ONLY change is the `if mask` skip."""
        pred_by_id = {S._row_id(r, i): r for i, r in enumerate(predictions)}
        groups: dict[str, dict[str, Any]] = {}
        fields = sorted({str(f) for r in gold_rows for f in r.get("evaluated_attributes", [])
                         if f not in S.CATEGORY_FIELDS})
        for i, gold in enumerate(gold_rows):
            pred = pred_by_id.get(S._row_id(gold, i), {})
            g_attrs = S.attributes_from_entry(gold)
            p_attrs = pred.get("attributes") or {}
            gid = str((gold.get("image") or {}).get("image_group_id") or S._row_id(gold, i))
            st = groups.setdefault(gid, {"rows": 0, "category_correct": 0, "master_correct": 0,
                                         "tp": 0, "fp": 0, "fn": 0,
                                         "fields": defaultdict(lambda: [0, 0, 0, 0])})
            st["rows"] += 1
            for f, key in (("category", "category_correct"), ("master_category", "master_correct")):
                st[key] += int(S._values(f, g_attrs.get(f)) == S._values(f, p_attrs.get(f)))
            annotated = set(gold.get("supervised_attributes", []))
            for f in fields:
                if mask and f not in annotated:
                    continue  # the one change: do not judge a field the source never annotated
                gv, pv = S._values(f, g_attrs.get(f)), S._values(f, p_attrs.get(f))
                tp, fp, fn = len(gv & pv), len(pv - gv), len(gv - pv)
                st["tp"] += tp; st["fp"] += fp; st["fn"] += fn
                c = st["fields"][f]; c[0] += tp; c[1] += fp; c[2] += fn; c[3] += len(gv)
        return groups

    def per_field(groups):
        tot = defaultdict(lambda: [0, 0, 0, 0])
        for g in groups.values():
            for f, c in g["fields"].items():
                for k in range(4): tot[f][k] += c[k]
        return {f: round(S._prf(c[0], c[1], c[2])["f1"], 4) for f, c in sorted(tot.items()) if c[3] > 0}

    gold = S._load_jsonl(args.gold)
    systems = dict(s.split("=", 1) for s in args.system)
    preds = {n: S._load_jsonl(Path(p)) for n, p in systems.items()}
    keys = ("attribute_micro_f1", "attribute_field_macro_f1",
            "attribute_micro_precision", "attribute_micro_recall")

    per_system, groups_masked = {}, {}
    for n in sorted(systems):
        published, _ = S._build_group_stats(gold, preds[n])
        replica = masked_group_stats(gold, preds[n], mask=False)
        pub_pt, rep_pt = S._point_metrics(list(published.values())), S._point_metrics(list(replica.values()))
        if any(abs(pub_pt[k] - rep_pt[k]) > 1e-12 for k in pub_pt):
            raise SystemExit(f"GUARD FAILED for {n}: unmasked replica differs from the published scorer")
        masked = masked_group_stats(gold, preds[n], mask=True)
        groups_masked[n] = masked
        m_pt = S._point_metrics(list(masked.values()))
        per_system[n] = {
            "exhaustive": {k: round(pub_pt[k], 4) for k in keys},
            "annotated_only": {k: round(m_pt[k], 4) for k in keys},
            "per_field_f1": {"exhaustive": per_field(published), "annotated_only": per_field(masked)},
        }
        print(f"{n:34} micro {pub_pt['attribute_micro_f1']:.4f} -> {m_pt['attribute_micro_f1']:.4f}   "
              f"macro {pub_pt['attribute_field_macro_f1']:.4f} -> {m_pt['attribute_field_macro_f1']:.4f}", flush=True)

    # Does the ranking hold once sparsity is forgiven? Same paired image-clustered bootstrap as compare().
    pairs = []
    for base, cand in itertools.combinations(sorted(systems), 2):
        ids = sorted(groups_masked[base])
        if ids != sorted(groups_masked[cand]):
            raise SystemExit(f"image groups differ between {base} and {cand}")
        bg, cg = groups_masked[base], groups_masked[cand]
        point = {k: S._point_metrics([cg[i] for i in ids])[k] - S._point_metrics([bg[i] for i in ids])[k]
                 for k in keys[:2]}
        rng, dist = random.Random(args.seed), {k: [] for k in keys[:2]}
        for _ in range(args.bootstrap_samples):
            smp = [ids[rng.randrange(len(ids))] for _ in ids]
            cp, bp = S._point_metrics([cg[i] for i in smp]), S._point_metrics([bg[i] for i in smp])
            for k in keys[:2]: dist[k].append(cp[k] - bp[k])
        ci = {k: {"low": round(S._percentile(v, 0.025), 4), "high": round(S._percentile(v, 0.975), 4)}
              for k, v in dist.items()}
        pairs.append({"baseline": base, "candidate": cand, "protocol": "annotated_only",
                      "delta": {k: round(v, 4) for k, v in point.items()}, "ci95": ci})
        print(f"  annotated-only {base} vs {cand}: micro {point['attribute_micro_f1']:+.4f} {ci['attribute_micro_f1']}"
              f"  macro {point['attribute_field_macro_f1']:+.4f} {ci['attribute_field_macro_f1']}", flush=True)

    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    out = {
        "question": "How much of the crop-track score is annotation sparsity?",
        "bound": "annotated-only is an upper bound: it also forgives hallucinations on fields that do not apply",
        "gold": {"sha256": sha(args.gold), "rows": len(gold),
                 "annotated_cells": sum(len(r["supervised_attributes"]) for r in gold),
                 "judged_cells": sum(len(r["evaluated_attributes"]) for r in gold)},
        "guard": "unmasked replica reproduced the published scorer exactly for every system",
        "systems": {n: {"sha256": sha(p), **per_system[n]} for n, p in systems.items()},
        "paired_annotated_only": pairs,
        "bootstrap": {"samples": args.bootstrap_samples, "seed": args.seed, "unit": "source image"},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sparse_annotation_rescore.json").write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", args.out_dir / "sparse_annotation_rescore.json")


if __name__ == "__main__":
    main()
