# moda-ner-shopping100k10 — model card (HF mirror)

**License: CC BY-NC 4.0** (weights). Shopping100k is an academic, non-commercial research
dataset; this weight license honors those terms. It binds us too: **these weights are not
part of Hopit's hosted product.** Commercial use of these weights is not permitted.

Ten field-specific supervised heads on the frozen encoder
[`HopitAI/moda-fashion-distilled`](https://huggingface.co/HopitAI/moda-fashion-distilled).
Input contract: a clean catalog product image. Fields: category, collar, **color**, fabric,
fastening, **fit**, neckline, pattern, pocket, sleeve length.

| | Field-macro set F1 (9,995 images, 61,384 cells) |
|---|---:|
| **This released checkpoint** | **TBD — filled at upload** |
| Best internal checkpoint | 0.8292 |
| Strongest external open baseline | FashionCLIP 2.0 + matched heads 0.6657 |

Selected best-internal field results: color 0.7053, fit 0.6708, fabric 0.6693, pocket
0.9491. Dataset access: on request to the Shopping100k authors; gold labels are rebuilt
from the source — never redistributed here.
