# MODA_NER(T)  `***`

**Tier `***`** — closed code, closed weights, benchmarks published. **Not distributed.**

We have one text checkpoint worth serving. The prior one scores 0.8601 against this one's
0.8723, a gap of 0.0122, so there is no weaker-but-useful version to publish while keeping a
stronger one back. Releasing anything here means releasing everything, and we are not ready
to do that on the strength of a silver-labelled evaluation.

**The benchmark, however, is public.** The `text` track ships in this repository with its
splits, label schema and scorer, so the number below is checkable and beatable even though
the model behind it is not distributed. That is the arrangement we would defend generally:
the ruler is open, the checkpoint is not.

A 149,625,627-parameter ModernBERT span extractor for fashion text.

**Input contract:** product titles, descriptions and captions.
**Output:** exact character spans, with per-entity confidence thresholds frozen on development.

## Results

| Split | Precision | Recall | Strict-span F1 | Rows |
|---|---:|---:|---:|---:|
| Development | 0.8867 | 0.8759 | 0.8813 | 711 |
| Test (full) | 0.8838 | 0.8610 | **0.8723** | 1,071 |

The previous served checkpoint scored 0.8601 strict F1 on the same split, so the calibrated
route adds 0.0122.

### Per entity, all of it

| Entity | F1 | Read this as |
|---|---:|---|
| FIT | 1.0000 | strong on silver labels |
| HEMLINE | 0.9892 | strong on silver labels |
| PATTERN | 0.9767 | strong on silver labels |
| MATERIAL | 0.9513 | strong on silver labels |
| SLEEVE | 0.9064 | strong on silver labels |
| OCCASION | 0.9063 | promising, limited support |
| GARMENT_TYPE | 0.8840 | strong on silver labels |
| NECKLINE | 0.8571 | useful on silver labels |
| DETAIL | 0.6396 | weak, ambiguous catch-all |
| COLOR | 0.6341 | weak, noisy labels |
| SILHOUETTE | 0.1538 | data-starved, do not rely on it |
| AESTHETIC | 0.0000 | unmeasurable — one gold span in the split |
| BRAND | not scored | no gold support in this benchmark |

The headline 0.8723 is a micro average. Four of the thirteen entity types are weak,
unmeasurable, or unscored, and the average does not show you that.

## What this model is not for

On a retailer diagnostic mapping title-derived garment-type spans onto a retailer's own
`product_type` metadata, this model loses to a trivial baseline:

| Route | Exact accuracy | Mean token F1 | Coverage |
|---|---:|---:|---:|
| Terminal title n-grams | **0.2667** | **0.3050** | **1.000** |
| This model, calibrated | 0.2200 | 0.2244 | 0.400 |

Those figures come from one retailer's catalogue, 300 rows. The diagnostic also covered 300
rows from a second retailer whose `product_type` values arrive from an API rather than being
derivable from the title, so neither route can recover them and both score zero. We exclude
those: pooling them would halve both aggregates and make a property of the source look like a
property of the models.

Two different tasks are in play. Extracting a span is not the same as producing a retailer's
own taxonomy value, and this model was built for the former. But if the second is what you
need, take the n-gram baseline and skip the model. Aligning an extractor to a particular
retailer's ontology is domain adaptation work, and no general checkpoint does it for you.

## Evidence class

The evaluation labels are cleaned silver: rule-derived, and already opened during
development. They are not independent human gold and not fresh confirmation. Treat 0.8723 as
a development figure rather than a validated one, and treat the entity table above as the
real description of what works.

**If you want this capability.** The honest answer is that a general checkpoint is not the
useful artefact here — ontology alignment to your catalogue is. That is what MODA_NER Pro
does. [Talk to us](https://hopit.ai).

**Credit.** Cite the MODA General Attribute Suite (`CITATION.cff`) when reporting numbers
from this track.
