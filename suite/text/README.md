# text track

Exact-span extraction from fashion product text. Thirteen entity types, 1,071 evaluation rows
(765 standard plus 306 hard). Strict span match: character offsets and entity type must both
be exact, because a nearly-right span is not usable downstream.

This is the only **self-contained** track. The labels are ours, so the splits ship here and
nothing needs downloading. It is also the only track with **no accompanying model** — we
publish the benchmark and our score on it, and keep the checkpoint.

## Files

| File | Rows | Use |
|---|---:|---|
| `train.jsonl` | — | not shipped; the clean release is evaluation-only |
| `dev.jsonl` | 711 | development |
| `test.jsonl` | 765 | standard test |
| `test_hard.jsonl` | 306 | hard test |
| `test_full.jsonl` | 1,071 | `test` + `test_hard`; the split our published number uses |
| `label_schema.json` | — | 13 entity types and their BIO labels |

## Score your model

```bash
python -m suite.text.commit --predictions my_predictions.jsonl
python -m suite.text.score \
  --gold suite/text/test_full.jsonl \
  --predictions my_predictions.jsonl \
  --output /tmp/mine.json
```

Predictions are JSONL, one row per gold `id`, each with an `entities` list of
`{start, end, label}`. The scorer fails closed: a missing row or an unknown id stops it rather
than quietly scoring a subset.

## Our number, and an honest gap

MODA_NER(T) scores strict-span F1 **0.8723** (precision 0.8838, recall 0.8610) on
`test_full`. Two things you should know before using that figure.

**It is a micro average, and four of the thirteen types are weak or unmeasurable.** SILHOUETTE
is data-starved, AESTHETIC has a single gold span in the split, BRAND has no gold support and
is not scored, and COLOR and DETAIL sit near 0.63. The per-entity table in the model card is
the real description of what works.

**Our own number is not yet checkable here.** We ship the gold and the scorer, so you can
score any model you like. We have not yet shipped a prediction file for MODA_NER(T), so
0.8723 currently rests on our word rather than on something you can recompute. That is
weaker than the `crop` track, where the shipped predictions regenerate the published figure
exactly, and we would rather say so than let the difference pass unnoticed.

## Labels

Cleaned silver: rule-derived from public fashion catalogues, no LLM involved, and already
opened during development. Not independent human gold, and not fresh confirmation.

## Sources and attribution

Rows derive from `arturayupov/womens-fashion-catalog` (MIT) and Fashionpedia-derived
synthetic text. The latter requires credit under CC BY 4.0: Jia et al., *Fashionpedia*,
ECCV 2020.
