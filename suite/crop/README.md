# crop track

Attribute extraction from a localized garment crop. Fifteen fields, oracle boxes supplied by
the evaluator, 4,688 crops across 1,158 source images. Splits and resampling cluster at the
source image, because one photograph yields several crops and they succeed and fail together.

## Reproduce our headline number

Gold labels are not committed here, on this track or any other. Fashionpedia's annotations are
CC BY 4.0, but its images are not ours to redistribute, so the split is rebuilt locally from
`manifest.json`:

```bash
python -m suite.crop.build_manifest --output-dir suite/crop   # writes benchmark.jsonl
```

Then score against the predictions that ship with this repository:

```bash
python -m suite.crop.score \
  --gold suite/crop/benchmark.jsonl \
  --predictions results/crop/moda-ner-v-crop/evaluation_predictions.jsonl \
  --output /tmp/recomputed.json
```

That regenerates attribute micro-F1 `0.6300`, field-macro `0.6074`, category accuracy
`0.8825` and master-category accuracy `0.9215` — the figures published in
`results/crop/moda-ner-v-crop/community_metrics.json`. `tests/test_identity.py` runs this on
every commit, so if it ever stops matching, CI says so before we do.

## Score your own model

```bash
# 1. predict on the same record_ids, without reading labels
# 2. commit the hash before any label is opened
python -m suite.crop.commit --predictions my_predictions.jsonl

# 3. score, and compare against ours in the same run
python -m suite.crop.score \
  --gold suite/crop/benchmark.jsonl \
  --predictions my_predictions.jsonl \
  --baseline-predictions results/crop/moda-ner-v-crop/evaluation_predictions.jsonl \
  --candidate-name my-model --baseline-name moda-ner-v-crop \
  --output /tmp/mine.json --comparison-output /tmp/vs_moda.json
```

The comparison output carries paired image-clustered bootstrap intervals, which is the only
number worth quoting.

## Images

Not redistributed. `record_id` carries the Fashionpedia image id, so a local copy of the
official 2020 validation images resolves every row. `build_manifest.py` rebuilds this split
from the official annotations if you want to verify the split itself rather than trust it.

## Attribution

The annotations underlying this track are CC BY 4.0 and require credit: Jia et al.,
*Fashionpedia: Ontology, Segmentation, and an Attribute Localization Dataset*, ECCV 2020.
