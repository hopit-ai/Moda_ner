# MODA_NER(V) — Full-body  `*`

**Hugging Face:** `HopitAI/moda-ner-v-fullbody`
**Tier `*`** — open code, open weights. **Weights: CC BY-NC 4.0.**

Eighteen class-balanced heads with an explicit N/A class, on our own frozen encoder (see Provenance below).

**Input contract:** one full-body fashion photograph.
**Output:** upper, lower and outer fabric; upper, lower and outer pattern; neckline; sleeve
length; lower-garment length; cardigan; navel coverage; hat; glasses; neckwear; waist
accessories; wrist accessories; ring; socks or leggings. Every field can return N/A, and N/A
is scored as a real class rather than treated as a negative.

| Measure (fullbody track) | Best internal | Strongest external open baseline | Paired 95% CI |
|---|---:|---:|---:|
| Tier-1 macro-F1 | 0.6917 | 0.5943 | [+0.0891, +0.1053] |
| Tier-2 N/A-F1 | 0.6637 | 0.6088 | [+0.0433, +0.0657] |
| Tier-3 visible macro-F1 | 0.5785 | 0.4969 | [+0.0723, +0.0905] |

**This released checkpoint: TBD at upload.** Evaluated on a fresh product-group-disjoint
split: 5,000 images, 1,751 groups, zero overlap with any prior experiment. Per-field range is
wide (0.3287 to 0.9187) — read the per-field table before relying on a single field.

**Why non-commercial.** Same as the catalog route: the evaluation corpus is research-only and
its terms extend to derived data. These weights are not in Hopit's hosted product. Commercial
deployments are fine-tuned on customer data instead.

## Provenance

The encoder these heads run on is ours: [`HopitAI/moda-fashion-distilled`](https://huggingface.co/HopitAI/moda-fashion-distilled), MIT, already public. Nothing from another vendor is loaded at inference time. (Recorded in the programme documentation; we re-confirm it against this track's frozen artifacts before release.)

That is worth stating plainly, because the comparator on this track is a FashionSigLIP-based system and it would be easy to assume this model is that system with heads attached. It is not. FashionSigLIP appears in two other roles:

- **As the distillation teacher.** An earlier ladder of checkpoints put conditional heads on frozen Marqo-FashionSigLIP. We distilled that system into our own encoder; the teacher is used during training and is not needed to serve.
- **As the baseline we measure against.** The comparator figure quoted above is that same FashionSigLIP-based system.

Lineage, stated once rather than implied: `moda-fashion-distilled` is itself a distilled student built on ViT-B/16-SigLIP, from a teacher ensemble that included our own DeepFashion2 fine-tune. Marqo-FashionSigLIP is Apache-2.0. The DeepFashion2 corpus is research-only, so we do not describe this pipeline as provenance-clean end to end.

**Credit for this model.** CC BY-NC requires attribution. Cite the MODA General Attribute
Suite (`CITATION.cff`).
