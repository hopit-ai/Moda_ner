"""Does the crop-track ranking depend on how the fifteen fields are aggregated?

The crop track's pre-registered headline metric is attribute micro-F1, which pools every
(garment, field) decision, so frequent fields dominate. Field-macro-F1 averages per-field F1,
so each of the fifteen fields counts equally. Both are computed by the unmodified suite scorer.

For every pair of systems with shipped predictions this runs the suite's own paired,
image-clustered bootstrap (suite.crop.score.compare) and records whether the two aggregations
agree on which system is ahead, and whether each delta's 95% interval excludes zero.

Section 6 of the resource report, arXiv:2609.13279 v2.

    python3 -m suite.crop.analysis.aggregation_sensitivity \\
        --suite-root <Moda_ner checkout> --gold <benchmark.jsonl> --out-dir <dir> \\
        --system moda-ner-v-crop=<preds.jsonl> --system other=<preds.jsonl> ...
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

# The two aggregations under test, plus the category metrics the scorer returns alongside.
METRICS = ("attribute_micro_f1", "attribute_field_macro_f1",
           "category_accuracy", "master_category_accuracy")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _excludes_zero(ci: dict) -> bool:
    return ci["low"] > 0 or ci["high"] < 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite-root", type=Path, required=True)
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--system", action="append", required=True, help="name=path/to/predictions.jsonl")
    # The suite protocol quotes 10,000 paired resamples; the scorer's CLI default is 1,000.
    ap.add_argument("--bootstrap-samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)  # the scorer's own default seed
    args = ap.parse_args()

    sys.path.insert(0, str(args.suite_root))
    from suite.crop.score import _load_jsonl, compare  # noqa: E402  the unmodified suite scorer

    systems = dict(s.split("=", 1) for s in args.system)
    gold = _load_jsonl(args.gold)
    preds = {name: _load_jsonl(Path(path)) for name, path in systems.items()}

    pairs = []
    for base, cand in itertools.combinations(sorted(systems), 2):
        t0 = time.time()
        c = compare(gold, preds[base], preds[cand], baseline_name=base, candidate_name=cand,
                    bootstrap_samples=args.bootstrap_samples, seed=args.seed)
        d, ci = c["candidate_minus_baseline"], c["paired_cluster_bootstrap_delta_95ci"]
        micro, macro = d["attribute_micro_f1"], d["attribute_field_macro_f1"]
        pairs.append({
            "baseline": base, "candidate": cand,
            "delta": {m: d[m] for m in METRICS},
            "ci95": {m: ci[m] for m in METRICS},
            "p_candidate_better": {m: c["bootstrap_probability_candidate_better"][m] for m in METRICS},
            # The question this script exists to answer.
            "aggregations_disagree_on_leader": (micro > 0) != (macro > 0) and micro != 0 and macro != 0,
            "micro_ci_excludes_zero": _excludes_zero(ci["attribute_micro_f1"]),
            "macro_ci_excludes_zero": _excludes_zero(ci["attribute_field_macro_f1"]),
            "per_field_value_f1_delta": c["per_field_value_f1_delta"],
            "seconds": round(time.time() - t0, 1),
        })
        print(f"{base} vs {cand}: micro {micro:+.4f} {ci['attribute_micro_f1']}  "
              f"macro {macro:+.4f} {ci['attribute_field_macro_f1']}  "
              f"disagree={pairs[-1]['aggregations_disagree_on_leader']}", flush=True)

    out = {
        "question": "Does the crop-track ranking depend on micro vs field-macro aggregation?",
        "gold": {"path": str(args.gold), "sha256": _sha256(args.gold), "rows": len(gold)},
        "systems": {n: {"path": p, "sha256": _sha256(Path(p))} for n, p in systems.items()},
        "bootstrap": {"samples": args.bootstrap_samples, "seed": args.seed,
                      "unit": "source image (the scorer's clustering unit)"},
        "pairs": pairs,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "aggregation_sensitivity.json").write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", args.out_dir / "aggregation_sensitivity.json")


if __name__ == "__main__":
    main()
