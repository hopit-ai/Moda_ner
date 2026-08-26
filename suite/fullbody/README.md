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

## Status of the scorer

The scoring code for this track currently lives inside a cloud execution wrapper and has not
yet been extracted into a standalone module. Until it is, the artifacts above let you inspect
every prediction and every published score, but not recompute them locally with one command
the way `crop` and `catalog` allow. That extraction is the next piece of work on this track,
and we would rather say so than imply parity.

## Licence note

The source corpus is a non-commercial research dataset whose terms extend to derived data.
Weights trained on it are released CC BY-NC 4.0, and that restriction binds us too: those
weights are not part of Hopit's paid product.
