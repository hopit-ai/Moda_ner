# MODA_NER(V) — Full-body  `*`

**Hugging Face:** `HopitAI/moda-ner-v-fullbody`
**Tier `*`** — open code, open weights. **Weights: CC BY-NC 4.0.**

Eighteen class-balanced heads with an explicit N/A class, on the frozen encoder
[`HopitAI/moda-fashion-distilled`](https://huggingface.co/HopitAI/moda-fashion-distilled).

**Input contract:** one full-body fashion photograph.
**Output:** upper, lower and outer fabric; upper, lower and outer pattern; neckline; sleeve
length; lower-garment length; cardigan; navel coverage; hat; glasses; neckwear; waist
accessories; wrist accessories; ring; socks or leggings. Every field can return N/A, and N/A
is scored as a real class rather than treated as a negative.

| Measure (fullbody18 track) | Best internal | Strongest external open baseline | Paired 95% CI |
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

**Credit for this model.** CC BY-NC requires attribution. Cite the MODA General Attribute
Suite (`CITATION.cff`).
