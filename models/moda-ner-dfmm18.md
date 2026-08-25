# moda-ner-dfmm18 — model card (HF mirror)

**License: CC BY-NC 4.0** (weights). DeepFashion-MultiModal is a non-commercial research
dataset whose agreement covers derived data; this weight license honors those terms. It
binds us too: **these weights are not part of Hopit's hosted product.** Commercial use of
these weights is not permitted.

Eighteen class-balanced heads with an explicit N/A class on the frozen encoder
[`HopitAI/moda-fashion-distilled`](https://huggingface.co/HopitAI/moda-fashion-distilled).
Input contract: a full-body fashion image. Fields: upper/lower/outer fabric,
upper/lower/outer pattern, neckline, sleeve length, lower-garment length, cardigan, navel
coverage, hat, glasses, neckwear, waist accessories, wrist accessories, ring,
socks/leggings.

| | MODA (best internal) | FashionCLIP baseline | Paired 95% CI |
|---|---:|---:|---:|
| Tier-1 macro-F1 | 0.6917 | 0.5943 | [+0.0891, +0.1053] |
| Tier-2 N/A-F1 | 0.6637 | 0.6088 | [+0.0433, +0.0657] |
| Tier-3 visible macro-F1 | 0.5785 | 0.4969 | [+0.0723, +0.0905] |

**This released checkpoint: TBD at upload** (per-tier numbers stated on upload). Evaluated
on the fresh product-group-disjoint shadow: 5,000 images, 1,751 groups, zero overlap with
any prior experiment. Range is wide across fields (lower fabric 0.3287 → cardigan 0.9187) —
read the per-field table in the suite before relying on any single field.
