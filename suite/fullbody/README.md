# fullbody track

Attribute extraction from a full-body fashion photograph. Eighteen fields with an explicit
not-applicable class, scored in three tiers: overall accuracy, whether the model knows when an
attribute is absent, and accuracy on the cases where it is present. 5,000 images across 1,751
product groups, with no overlap against any earlier experiment. Splits and resampling cluster
at the product group.

Gold labels are not committed here. Obtain the source dataset from its authors under their
terms.

## Published evidence

| File | Contents |
|---|---|
| `results/fullbody/moda-ner-v-fullbody/predictions.jsonl` | 5,000 predictions from MODA v7 |
| `results/fullbody/fashionclip-matched/predictions.jsonl` | 5,000 from the matched comparator |
| `results/fullbody/moda-ner-v-fullbody/benchmark_result.json` | Both systems' three-tier scores |
| `results/fullbody/scoring/paired_product_cluster_bootstrap.json` | 10,000-sample paired bootstrap, clustered by product group |
| `results/fullbody/scoring/claim_gate.json` | The promotion gate and whether it passed |

Scores: Tier-1 macro-F1 `0.691671`, Tier-2 N/A-F1 `0.663655`, Tier-3 visible macro-F1
`0.578509`, against `0.594298` / `0.608772` / `0.496860` for the comparator. Differences and
their intervals are in the bootstrap file.

## Reproduce our numbers

```bash
python -m suite.fullbody.score \
  --labels <your rebuilt shadow split> \
  --predictions results/fullbody/moda-ner-v-fullbody/predictions.jsonl \
  --comparator-predictions results/fullbody/fashionclip-matched/predictions.jsonl \
  --system-id moda_v7 --comparator-id fashionclip \
  --output /tmp/recomputed.json
```

That regenerates all six three-tier scores and all three paired confidence intervals exactly
as published, including the 10,000-sample bootstrap clustered by product group.

Unlike `crop` and `catalog`, this scorer needs `numpy` and `torch`: the paired bootstrap is
vectorised, and we kept the original implementation rather than rewriting it so that the
numbers match the published run rather than approximating it.

## Licence note

The source corpus is a non-commercial research dataset whose terms extend to derived data.
Weights trained on it are released CC BY-NC 4.0, and that restriction binds us too: those
weights are not part of Hopit's paid product.
