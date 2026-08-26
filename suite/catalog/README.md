# catalog track

Attribute extraction from a clean catalogue product image. Ten fields including colour and
fit, 9,995 test images, 61,384 eligible attribute cells. Splits and resampling cluster at the
image.

Gold labels are not committed here. Obtain the source dataset from its authors under their
terms; the frozen split is reconstructed from IDs and checksums.

## Reproduce our number

```bash
python -m suite.catalog.score \
  --labels <your rebuilt test split> \
  --predictions results/catalog/moda-ner-v-catalog/predictions.jsonl \
  --system-id moda-ner-v-catalog \
  --output /tmp/recomputed.json
```

That regenerates field-macro set F1 `0.829235` and micro set F1 `0.814007`, matching
`results/catalog/moda-ner-v-catalog/result.json`, along with all ten per-attribute scores.

The comparator's predictions are in `results/catalog/fashionclip-probe/`, so the published
comparison can be recomputed too.

## Licence note

The source corpus is an academic, non-commercial research dataset. Weights trained on it are
released CC BY-NC 4.0, and that restriction binds us as well: those weights are not part of
Hopit's paid product.
