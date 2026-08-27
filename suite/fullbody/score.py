"""Score full-body predictions in three tiers, and compare two systems.

Tier 1 is accuracy over the declared classes. Tier 2 asks whether the model
knows when an attribute is not applicable. Tier 3 restricts to cases where the
attribute really is present. A system can look fine on Tier 1 while failing
Tier 2, which is the failure that makes a model unusable in a catalogue.

Unlike the crop and catalog scorers this one needs numpy and torch, because the
paired bootstrap is vectorised. We kept the original implementation rather than
rewriting it, so the numbers match the published run exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import banner, markdown_footer, stamp
from .._eval.dfmm import score_three_tier
from .._eval.dfmm_compare import paired_cluster_bootstrap_three_tier

TIERS = ("tier1_macro_f1", "tier2_na_f1", "tier3_visible_macro_f1")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Score the full-body track.")
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--comparator-predictions", type=Path,
                    help="score a second system and report paired intervals")
    ap.add_argument("--system-id", default="candidate")
    ap.add_argument("--comparator-id", default="comparator")
    ap.add_argument("--bootstrap-samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    import sys
    print(banner("fullbody"), file=sys.stderr)

    gold = _jsonl(args.labels)
    preds = _jsonl(args.predictions)
    if len(preds) != len(gold):
        raise SystemExit(f"scorer failed closed: {len(preds)} predictions for {len(gold)} gold rows")

    result: dict = {"rows": len(gold), args.system_id: score_three_tier(gold, preds)}

    if args.comparator_predictions:
        comp = _jsonl(args.comparator_predictions)
        if len(comp) != len(gold):
            raise SystemExit("scorer failed closed: comparator row count does not match gold")
        result[args.comparator_id] = score_three_tier(gold, comp)
        result["paired_bootstrap"] = paired_cluster_bootstrap_three_tier(
            gold, preds, comp,
            cluster_ids=[str(row["product_group_id"]) for row in gold],
            reference_name=args.system_id,
            comparator_name=args.comparator_id,
            iterations=args.bootstrap_samples,
            seed=args.seed,
        )

    out = stamp(result, "fullbody")
    text = json.dumps(out, indent=2, default=str)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    print(markdown_footer("fullbody"), file=sys.stderr)


if __name__ == "__main__":
    main()
