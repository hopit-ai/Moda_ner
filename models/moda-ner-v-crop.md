# MODA_NER(V) — Crop  `*`

**Hugging Face:** `HopitAI/moda-ner-v-crop`
**Tier `*`** — open code, open weights. **Weights: MIT.**

Conditional attribute heads on the frozen MIT encoder
[`HopitAI/moda-fashion-distilled`](https://huggingface.co/HopitAI/moda-fashion-distilled).

**Input contract:** one localized garment crop.
**Output:** 15 sparse fields — master category, category, sub-category, silhouette, hemline,
sleeve length and shape, neckline, collar presence and style, waist, material, surface
treatment, pattern, closure. No colour, no fit; the evaluation corpus for this track carries
no equivalent labels.

| | Attribute micro-F1 (crop15 track) |
|---|---:|
| **This released checkpoint** | **TBD at upload** |
| Best internal checkpoint | 0.6300 |
| Strongest external open baseline | 0.6245 |
| Zero-shot open VLM reference | 0.1805 |

The released checkpoint is a strong open baseline and is deliberately not our strongest; both
figures are stated so nothing is implied. Weak fields: material, surface treatment, collar
style, neckline, waist type.

**Attribution.** The annotations behind this track are licensed CC BY 4.0 and require credit:
Jia et al., *Fashionpedia: Ontology, Segmentation, and an Attribute Localization Dataset*,
ECCV 2020. No images from that corpus are redistributed in the weights or the repository.

**Credit for this model.** If you use these weights or report numbers from this track, cite
the MODA General Attribute Suite (see `CITATION.cff`). Scorers stamp the suite name and
version into every result file for exactly this reason.
