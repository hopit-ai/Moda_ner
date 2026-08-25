# moda-ner-fashionpedia15 — model card (HF mirror)

**License: MIT** (weights). Fashionpedia annotations/ontology are CC BY 4.0 — this model
was trained using those annotations; attribution: Jia et al., *Fashionpedia* (ECCV 2020).
Images were never redistributed and are not contained in the weights release.

Conditional attribute heads on the frozen MIT encoder
[`HopitAI/moda-fashion-distilled`](https://huggingface.co/HopitAI/moda-fashion-distilled).
Input contract: a localized garment crop (oracle box). Output: sparse MODA-15 fields —
master category, category, sub-category, silhouette, hemline, sleeve length/shape,
neckline, collar presence/style, waist, material, surface treatment, pattern, closure.
No color, no fit (Fashionpedia publishes no equivalent labels).

| | Attribute micro-F1 (4,688 crops, 1,158 image clusters) |
|---|---:|
| **This released checkpoint** | **TBD — filled at upload** |
| Best internal checkpoint (hosted tier) | 0.6300 [0.6132-class CI reported in suite] |
| Strongest external open baseline | FashionSigLIP spatial residual 0.6245 |
| Zero-shot reference | Qwen3-VL-8B 0.1805 |

The released checkpoint is a strong open baseline, deliberately not our best; both numbers
are stated so nothing is implied. Known weak fields: material, surface treatment, collar
style, neckline, waist type. Evaluate with the suite in this repo before production use.
