# MODA_NER(V) — Catalog  `*`

**Hugging Face:** `HopitAI/moda-ner-v-catalog`
**Tier `*`** — open code, open weights. **Weights: CC BY-NC 4.0.**

Ten field-specific supervised heads on the frozen encoder
[`HopitAI/moda-fashion-distilled`](https://huggingface.co/HopitAI/moda-fashion-distilled).

**Input contract:** one clean catalogue product image.
**Output:** category, collar, **colour**, fabric, fastening, **fit**, neckline, pattern,
pocket, sleeve length.

| | Field-macro set F1 (catalog10 track) |
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

**Credit for this model.** CC BY-NC requires attribution. Cite the MODA General Attribute
Suite (`CITATION.cff`) when reporting numbers from this track.
