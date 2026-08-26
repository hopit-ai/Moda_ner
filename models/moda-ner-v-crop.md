# MODA_NER(V) — Crop  `*`

**Hugging Face:** `HopitAI/moda-ner-v-crop`
**Tier `*`** — open code, open weights. **Weights: MIT.**

Conditional attribute heads on our own frozen encoder (see Provenance below).

**Input contract:** one localized garment crop.
**Output:** 15 sparse fields — master category, category, sub-category, silhouette, hemline,
sleeve length and shape, neckline, collar presence and style, waist, material, surface
treatment, pattern, closure. No colour, no fit; the evaluation corpus for this track carries
no equivalent labels.

| | Attribute micro-F1 (crop track) |
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

## Provenance

The encoder these heads run on is ours: [`HopitAI/moda-fashion-distilled`](https://huggingface.co/HopitAI/moda-fashion-distilled), MIT, already public. Nothing from another vendor is loaded at inference time.

That is worth stating plainly, because the comparator on this track is a FashionSigLIP-based system and it would be easy to assume this model is that system with heads attached. It is not. FashionSigLIP appears in two other roles:

- **As the distillation teacher.** An earlier ladder of checkpoints put conditional heads on frozen Marqo-FashionSigLIP. We distilled that system into our own encoder; the teacher is used during training and is not needed to serve.
- **As the baseline we measure against.** The comparator figure quoted above is that same FashionSigLIP-based system.

Lineage, stated once rather than implied: `moda-fashion-distilled` is itself a distilled student built on ViT-B/16-SigLIP, from a teacher ensemble that included our own DeepFashion2 fine-tune. Marqo-FashionSigLIP is Apache-2.0. The DeepFashion2 corpus is research-only, so we do not describe this pipeline as provenance-clean end to end.

**Credit for this model.** If you use these weights or report numbers from this track, cite
the MODA General Attribute Suite (see `CITATION.cff`). Scorers stamp the suite name and
version into every result file for exactly this reason.
