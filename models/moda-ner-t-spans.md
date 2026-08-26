# MODA_NER(T) — Spans  `**`

**Hugging Face:** `HopitAI/moda-ner-t-spans`
**Tier `**`** — open weights, closed code. **Weights: MIT** (pending source-licence
verification of two training corpora).

A 149,625,627-parameter ModernBERT span extractor for fashion text.

**Input contract:** product titles, descriptions and captions.
**Output:** exact character spans for 13 entity types — garment type, material, colour,
pattern, silhouette, fit, neckline, sleeve, hemline, brand, occasion, aesthetic, detail.

| Metric (textspan13 track) | Result |
|---|---:|
| Strict-span F1 | 0.8723 |
| Precision | 0.8838 |
| Recall | 0.8610 |
| Evaluation rows | 1,071 |

Selected entity F1: fit 1.0000, hemline 0.9892, pattern 0.9767, material 0.9513, sleeve
0.9064, occasion 0.9063, garment type 0.8840, neckline 0.8571.

**Evidence class, stated plainly.** These labels are cleaned silver — rule-derived, and
already opened during development. They are not independent human gold and not fresh
confirmation. That makes this number a weaker class of evidence than the image tracks, which
use corpus-native annotation. Treat 0.8723 as a development figure, not a validated one.

**Credit for this model.** Cite the MODA General Attribute Suite (`CITATION.cff`).
