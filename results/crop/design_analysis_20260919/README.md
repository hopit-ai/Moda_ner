# Crop-track design analyses, 2026-09-19

The receipts behind Section 6 of the resource report (arXiv:2609.13279, version 2). None of them
changes a published `crop` number; each measures how that number should be read. Scripts are in
`suite/crop/analysis/`; all scoring goes through the unmodified `suite.crop.score`.

| File | Paper | What it is |
|---|---|---|
| `aggregation_sensitivity.json` | §6.1 | All 15 pairs of the six systems with committed `crop` predictions, micro-F1 and field-macro F1, paired image-clustered bootstrap (10,000 samples, seed 42) |
| `sparse_annotation_rescore.json` | §6.2 | Each system scored as published and on annotated fields only |
| `applicability_ablation.json` | §6.3 | The released `moda-ner-v-crop` decoded three ways, changing only the per-field applicability threshold |
| `predictions_shipped.jsonl` | §6.3 | Shipped thresholds; reproduces the published file on 4,598 of 4,688 rows |
| `predictions_uncalibrated.jsonl` | §6.3 | Applicability threshold 0.5 on every field |
| `predictions_applicability_off.jsonl` | §6.3 | Applicability threshold 0.0: every field emitted |

SHA-256 of the three prediction files is recorded in `applicability_ablation.json`
(`prediction_sha256`). The zero-shot baselines' predictions used in §6.1 and §6.2 are in
`../fashionsiglip-zero-shot/` and `../qwen3vl-8b-zero-shot/`; their hashes were committed in
`../RESULTS_MANIFEST.json` before scoring.

## Reproduce

Build the gold first (`suite/crop/build_manifest.py`, from the official Fashionpedia 2020
validation annotations); its SHA-256 must be
`00a60391c5fdc39d4e900c957c5b758123c419f58cbad8c7f955f46fe8e04002`.

```bash
R=results/crop
SYSTEMS="--system moda-ner-v-crop=$R/moda-ner-v-crop/evaluation_predictions.jsonl \
  --system fashionsiglip-spatial-residual=$R/fashionsiglip-spatial-residual-full130k-e1/evaluation_predictions.jsonl \
  --system parent=$R/parent/evaluation_predictions.jsonl \
  --system corrective=$R/corrective/evaluation_predictions.jsonl \
  --system fashionsiglip-zero-shot=$R/fashionsiglip-zero-shot/evaluation_predictions.jsonl \
  --system qwen3vl-8b-zero-shot=$R/qwen3vl-8b-zero-shot/evaluation_predictions.jsonl"

python3 -m suite.crop.analysis.aggregation_sensitivity --suite-root . --gold benchmark.jsonl --out-dir /tmp/agg $SYSTEMS
python3 -m suite.crop.analysis.sparse_annotation_rescore --suite-root . --gold benchmark.jsonl --out-dir /tmp/sparse $SYSTEMS

# the three re-decodings score directly
python3 -m suite.crop.score --gold benchmark.jsonl \
  --predictions results/crop/design_analysis_20260919/predictions_applicability_off.jsonl --output /tmp/off.json
```

`suite/crop/analysis/applicability_ablation.py` regenerates the three prediction files from the
released weights and the Fashionpedia validation images. It needs a GPU or Apple silicon for
reasonable speed; rows at a threshold edge can differ across hardware.

## What §6.3 can and cannot show

Switching the applicability decision off drops micro-F1 from 0.6295 to 0.2696, so most of the
headline rests on deciding which fields to leave out. The annotated-only column cannot say
whether those decisions are right: every cell in it has a gold value, so there is no absent case
to test. Fashionpedia's labels record no explicit absence, so this track cannot distinguish a
correct absence decision from a learned habit of the source annotators.
