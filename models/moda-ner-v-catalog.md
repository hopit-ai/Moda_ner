# MODA_NER(V) — Catalog  `*`

**Hugging Face:** `HopitAI/moda-ner-v-catalog`
**Tier `*`** — open code, open weights. **Weights: CC BY-NC 4.0.**

Ten field-specific supervised heads on our own frozen encoder (see Provenance below).

**Input contract:** one clean catalogue product image.
**Output:** category, collar, **colour**, fabric, fastening, **fit**, neckline, pattern,
pocket, sleeve length.

| | Field-macro set F1 (catalog track) |
|---|---:|
| **This released checkpoint** | **TBD at upload** |
| Best internal checkpoint | 0.8292 |
| Strongest external open baseline | 0.6657 |

Selected best-internal fields: colour 0.7053, fit 0.6708, fabric 0.6693, pocket 0.9491.

**Why non-commercial.** This track is evaluated against a research-only corpus whose terms do
not permit commercial use of models trained on it. We honour those terms, and they bind us as
well: **these weights are not part of Hopit's hosted product.** For commercial deployment we
fine-tune on the customer's own catalogue, which raises accuracy on their taxonomy and
produces a model with no dependency on research-licensed data.

## Provenance

The encoder these heads run on is ours: [`HopitAI/moda-fashion-distilled`](https://huggingface.co/HopitAI/moda-fashion-distilled), MIT, already public. Nothing from another vendor is loaded at inference time. (Recorded in the programme documentation; we re-confirm it against this track's frozen artifacts before release.)

That is worth stating plainly, because the comparator on this track is a FashionSigLIP-based system and it would be easy to assume this model is that system with heads attached. It is not. FashionSigLIP appears in two other roles:

- **As the distillation teacher.** An earlier ladder of checkpoints put conditional heads on frozen Marqo-FashionSigLIP. We distilled that system into our own encoder; the teacher is used during training and is not needed to serve.
- **As the baseline we measure against.** The comparator figure quoted above is that same FashionSigLIP-based system.

Lineage, stated once rather than implied: `moda-fashion-distilled` is itself a distilled student built on ViT-B/16-SigLIP, from a teacher ensemble that included our own DeepFashion2 fine-tune. Marqo-FashionSigLIP is Apache-2.0. The DeepFashion2 corpus is research-only, so we do not describe this pipeline as provenance-clean end to end.

**Credit for this model.** CC BY-NC requires attribution. Cite the MODA General Attribute
Suite (`CITATION.cff`) when reporting numbers from this track.
