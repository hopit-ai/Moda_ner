# MODA_NER(T)  `**`

**Hugging Face:** `HopitAI/moda-ner-t`
**Tier `**`** — open weights, code closed. **Weights: MIT**, pending verification of two
training-source licences.

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

On a 600-row retailer diagnostic mapping title-derived garment-type spans onto retailer
`product_type` metadata, this model loses to a trivial baseline:

| Route | Exact accuracy | Mean token F1 | Coverage |
|---|---:|---:|---:|
| Terminal title n-grams | **0.1333** | **0.4037** | **1.000** |
| This model, calibrated | 0.1100 | 0.1793 | 0.5217 |
| Previous checkpoint | 0.0817 | 0.1510 | 0.6867 |

Those are two different tasks. Extracting a span is not the same as producing a retailer's
own product taxonomy value, and this model was built for the former. But if the second task
is what you need, take the n-gram baseline and skip the model. Aligning an extractor to a
particular retailer's ontology is domain adaptation work, and no general checkpoint does it
for you.

## Evidence class

The evaluation labels are cleaned silver: rule-derived, and already opened during
development. They are not independent human gold and not fresh confirmation. Treat 0.8723 as
a development figure rather than a validated one, and treat the entity table above as the
real description of what works.

**Credit.** Cite the MODA General Attribute Suite (`CITATION.cff`) when reporting numbers
from this track.
