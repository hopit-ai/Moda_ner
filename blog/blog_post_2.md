# The numbers, the code, and what we are releasing

**MODA_NER, Part 2: results and release**

*Series: Fashion attribute extraction from images*
*Previous: [Why we built the test system before the model](blog_post_1.md)*

The last post argued that a test system should come before the models it judges. This one puts our own models through it and publishes everything needed to check the result: the scorers, the builders that reconstruct each frozen split, our prediction files with their hashes, and the model weights.

If you disagree with a number here, you can recompute it.

## What the model is asked to predict

The `crop` track uses fifteen fields: three category levels, shape (`silhouette`, `hemline`, `waist_type`), sleeves (`sleeve_length`, `sleeve_shape`), neck and collar (`neckline`, `collar_presence`, `collar_style`), surface (`material`, `surface_treatment`, `pattern`), and `closure_type`. Each has a fixed vocabulary published with the track: 21 silhouettes, 21 necklines, and so on.

Colour and fit are not in that list, because the annotations behind `crop` do not cover them. They are scored on `catalog` instead, which uses ten fields. `fullbody` uses eighteen with an explicit not-applicable class. `text` identifies thirteen entity types in product copy.

A benchmark cannot fairly judge an answer if it will not say which answers are allowed, so all of that ships.

## Results

| Track | What the score measures | MODA | Comparator | 95% interval |
|---|---|---:|---:|---:|
| `crop` | share of predictions matching exactly, 15 fields | **0.6300** | 0.6245 | [+0.0014, +0.0097] |
| `catalog` | score per field, averaged over 10 fields | **0.8292** | 0.6657 | [+0.1595, +0.1676] |
| `fullbody` overall | score per field, averaged over 18 fields | **0.6917** | 0.5943 | [+0.0891, +0.1053] |
| `fullbody` knows when not to answer | reliability of saying an attribute is absent | **0.6637** | 0.6088 | [+0.0433, +0.0657] |
| `fullbody` when visible | the 18-field average on attributes that are present | **0.5785** | 0.4969 | [+0.0723, +0.0905] |

Do not read across rows. Different images, different fields, different metrics. Ten catalogue fields on clean studio photos is an easier problem than fifteen on a cropped garment, so 0.8292 is not "better" than 0.6300. Each row compares one system against one comparator on one track.

On `catalog` and `fullbody` the comparator is FashionCLIP 2.0 with matched supervised heads, a genuine external system. On `crop` it is not: that 0.6245 is our own architecture and training release running on a frozen third-party encoder instead of ours. So that row is an encoder experiment, not a win over another company's product. The real third-party baselines on `crop` are zero-shot systems at 0.1805 and 0.1817, neither built for this task, which is why we do not make much of the gap.

The claim this supports covers the three image tracks only:

> MODA General is the best of the named open systems evaluated under the frozen public MODA General Attribute Suite, spanning localized garment crops, catalog product images, color and fit, and applicability-aware full-body attributes.

Not world-best, not human-level, not ready for every production case. The `text` track sits outside that sentence deliberately: it was added after the claim was frozen, and no external system has been evaluated on it under the same conditions.

## The number behind the number

On `crop`, 0.6300 means about 63% of individual attribute predictions match the label exactly. On 49.66% of garments, all fifteen fields are right.

The average hides a spread of more than fifty points.

| Field | F1 | Field | F1 |
|---|---:|---|---:|
| `master_category` | 0.9215 | `sub_category` | 0.5966 |
| `category` | 0.8825 | `silhouette` | 0.5535 |
| `pattern` | 0.8356 | `waist_type` | 0.5002 |
| `sleeve_length` | 0.8073 | `neckline` | 0.4650 |
| `closure_type` | 0.7545 | `collar_style` | 0.4566 |
| `collar_presence` | 0.7508 | `surface_treatment` | 0.4398 |
| `sleeve_shape` | 0.6787 | `material` | 0.4148 |
| `hemline` | 0.6428 | | |

Telling a coat from a dress is close to solved. Telling denim from twill is not: `material` sits at 0.4148 on 146 labelled cases, both the hardest field and the thinnest evidence. Read this table before trusting the headline.

## Architectural decisions

**One sequence became many heads.** The Florence-2 system from the last post generated every attribute in a single autoregressive response, which couples fields that have nothing to do with each other, lets rare labels lose out to common ones, and leaves no way to abstain. We replaced it with a frozen vision encoder and separate heads per field.

**Applicability is its own decision.** Each conditional field gets a binary head answering "does this apply here?" before any value head runs, and it is scored separately. That is the `fullbody` row above. A model that invents a neckline for a pair of trousers should not be rewarded for confidence.

**Thresholds are calibrated, not guessed**, on a split that never touches development or test. Multi-label fields such as material use an asymmetric focal loss, with balancing capped so a common positive cannot swamp a rare one.

**The backbone was tested rather than assumed.** We compared a general SigLIP-2 encoder against a fashion-pretrained one under an identical development protocol. SigLIP-2 reached 0.6018 against 0.6163. That settled it for about eleven cents of compute, before anyone had to argue.

**The encoder is ours, and this is worth being precise about.** An earlier ladder of checkpoints put our heads on a frozen third-party fashion encoder. We distilled that system into our own encoder, which is what serves today. The third-party encoder appears twice in this work  -  once as the distillation teacher during training, once as the comparator on the `crop` row  -  and never as the model being served.

## What we are releasing

| Model | Input | Weights | Commercial use |
|---|---|---|---|
| **MODA_NER(V) Crop** | cropped garment | MIT | yes |
| **MODA_NER(V) Catalog** | catalogue product image | CC BY-NC 4.0 | no |
| **MODA_NER(V) Full-body** | full-body photo | CC BY-NC 4.0 | no |

Two of the four tracks are evaluated against research-only corpora whose terms extend to derived data, so weights trained on them are non-commercial. That restriction binds us as well: those exact weights are not part of our paid product, and we verified that no serving path loads them before publishing this.

We are not holding back a better model either. The published `crop` checkpoint is the same one that produced the 0.6300 above. Download it, score it, and you should get that number.

**MODA_NER(T)**, the text model, is not distributed. Its benchmark is: the track ships with a builder that regenerates the splits from their two public sources, a scorer, and our score on it. If you want to beat 0.8723 at span extraction, the track is right there.

## Reproduce

We do not ship gold labels on any track. You obtain each corpus from its original source under that source's terms, and the builders reconstruct the exact frozen split from record IDs and checksums. What ships is the protocol, the scorers, the uncertainty code, and our own prediction files with their hashes, including the runs we lost.

```
python -m suite.crop.score \
  --gold <your rebuilt split> \
  --predictions results/crop/moda-ner-v-crop/evaluation_predictions.jsonl \
  --output /tmp/recomputed.json
```

Every figure in this post was regenerated this way before publishing, including all three `fullbody` confidence intervals from the 10,000-sample bootstrap clustered over 1,751 product groups. The `crop` and `catalog` scorers need nothing beyond the Python standard library.

## Run your own model

Freeze your checkpoint. Predict on the same record IDs without reading labels. Commit the prediction hash. Score each track and report the paired interval against our published predictions.

If another team beats us under this protocol, we want to know. It is better to find a weakness in a reproducible test than in a customer's production pipeline.

*Next: what none of this tells you about your own catalogue.*
