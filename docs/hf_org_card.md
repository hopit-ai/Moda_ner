# Hopit AI

World models for fashion: search, retrieval, ranking, and multimodal reasoning.

We publish two tracks. Each one ships its weights, its evaluation protocol, and the
prediction files behind every number we quote, so our claims can be checked rather
than taken on trust.

## Moda: search and retrieval

Fashion image encoders for search, recommendation, and visual similarity. The
distilled encoder is the one we serve, and it is the backbone under the attribute
models below.

| Model | What it is |
|---|---|
| [moda-fashion-distilled](https://huggingface.co/HopitAI/moda-fashion-distilled) | The served encoder. Start here. |
| [moda-fashion-matryoshka](https://huggingface.co/HopitAI/moda-fashion-matryoshka) | Nested embeddings, so one model serves several dimensions |
| [moda-fashion-distilled-512d](https://huggingface.co/HopitAI/moda-fashion-distilled-512d) | Smaller embeddings for tighter index budgets |
| [moda-fashion-crossdomain](https://huggingface.co/HopitAI/moda-fashion-crossdomain) | Street photos matched to catalogue product shots |
| [moda-fashion-vision-fp16](https://huggingface.co/HopitAI/moda-fashion-vision-fp16) | Half-precision vision tower for cheaper inference |

Benchmarks: [hopit-ai.github.io/Moda](https://hopit-ai.github.io/Moda/) ·
Code: [github.com/hopit-ai/Moda](https://github.com/hopit-ai/Moda)

## MODA_NER: attribute extraction

Turning a fashion image into structured product data: category, colour, fit,
neckline, sleeve length, pattern, material. Each model is scored on one frozen
track of the MODA General Attribute Suite, and the tracks are never averaged
together, because a model can be strong on clean product shots and weak on
full-body photos.

| Model | Input | Licence |
|---|---|---|
| [moda-ner-v-crop](https://huggingface.co/HopitAI/moda-ner-v-crop) | A cropped garment | MIT |
| [moda-ner-v-catalog](https://huggingface.co/HopitAI/moda-ner-v-catalog) | A catalogue product image | CC BY-NC 4.0 |
| [moda-ner-v-fullbody](https://huggingface.co/HopitAI/moda-ner-v-fullbody) | A full-body photo | CC BY-NC 4.0 |

Two of these are evaluated against research-only corpora whose terms reach derived
data, so their weights are non-commercial. That binds us too: those weights are not
part of our paid product.

Benchmarks: [hopit-ai.github.io/Moda_ner](https://hopit-ai.github.io/Moda_ner/) ·
Code: [github.com/hopit-ai/Moda_ner](https://github.com/hopit-ai/Moda_ner)

## How we evaluate

We freeze the test set before running a model, keep related images out of both
training and test data, hash every prediction file before scoring it, score every
row instead of quietly dropping the awkward ones, and require a real gain on each
track rather than a good average. We publish the losing runs alongside the winning
ones.

Contact: [hopit.ai](https://hopit.ai)
