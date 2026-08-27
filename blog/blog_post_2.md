# The numbers, the code, and what we are releasing

**MODA_NER, Part 2: results and release**

*Series: Fashion attribute extraction from images*  
*Previous: [Why we built the test system before the model](blog_post_1.md)*

In the first post, we explained why a test system should come before the models it judges. This post puts our models through that system, and releases the evidence needed to check the result.

That includes the scorers, builders that recreate the frozen splits, prediction files with their hashes, and the model weights.

If you disagree with a number here, you can recompute it.

## The claim, and its boundaries

The claim supported by these results applies only to the three image tracks:

> MODA General is the best of the named open systems evaluated under the frozen public MODA General Attribute Suite, spanning localized garment crops, catalog product images, color and fit, and applicability-aware full-body attributes.

This is not a claim of world-best performance, human-level quality, or readiness for every production case.

The `text` track is deliberately outside the claim. It was added after the claim was frozen, and no external model has been evaluated on it under the same conditions.

## What we ask the models to predict

The `crop` track covers fifteen fields: three category levels (`master_category`, `category`, `sub_category`); shape (`silhouette`, `hemline`, `waist_type`); sleeves (`sleeve_length`, `sleeve_shape`); neck and collar (`neckline`, `collar_presence`, `collar_style`); surface (`material`, `surface_treatment`, `pattern`); and `closure_type`.

Every field has a fixed, published vocabulary: 21 silhouettes, 21 necklines, and so on. A benchmark cannot fairly judge an answer if it does not state which answers are valid.

Colour and fit are not included in `crop`, because its source annotations do not cover them. They are evaluated in `catalog`, which has ten fields. `fullbody` has eighteen fields and an explicit not-applicable class. `text` identifies thirteen entity types in product copy.

## Results

| Track | What the score measures | MODA | Comparator | 95% interval |
|---|---|---:|---:|---:|
| `crop` | Exact matches across 15 fields | **0.6300** | 0.6245 | [+0.0014, +0.0097] |
| `catalog` | Per-field score, averaged over 10 fields | **0.8292** | 0.6657 | [+0.1595, +0.1676] |
| `fullbody`: overall | Per-field score, averaged over 18 fields | **0.6917** | 0.5943 | [+0.0891, +0.1053] |
| `fullbody`: knows when not to answer | Correctly identifies an absent attribute | **0.6637** | 0.6088 | [+0.0433, +0.0657] |
| `fullbody`: when visible | The 18-field average where the attribute is present | **0.5785** | 0.4969 | [+0.0723, +0.0905] |

Do not compare scores across rows. The tracks use different images, fields, and metrics. Ten catalogue fields on clean studio photos are an easier problem than fifteen fields on a cropped garment, so 0.8292 is not "better" than 0.6300. Each row compares one system with one comparator on one task.

The `catalog` and `fullbody` comparator is FashionCLIP 2.0 with matched supervised heads: a genuine external system. The `crop` row is different. Its 0.6245 comparator uses our architecture and training release, but a frozen third-party encoder instead of ours. It is an encoder experiment, not a win over another company's product.

The real third-party `crop` baselines are zero-shot systems at 0.1805 and 0.1817. Neither was built for this task, which is why we do not make much of the gap.

## What about the frontier models?

It is the first question people ask, so here is what we have and why we do not treat it as a result.

We ran a cost-gated checkpoint against a commercial vision-language model: 100 rows shared across all three tracks, with a continuation rule fixed in advance. It lost on every track.

| Track | Commercial VLM | MODA |
|---|---:|---:|
| `crop` attribute micro-F1 | 0.3624 | **0.6667** |
| `catalog` field-macro set F1 | 0.5830 | **0.8057** |
| `fullbody` Tier-1 macro-F1 | 0.3522 | **0.4482** |

The gate failed, so we did not buy the remaining 900 calls.

We report that as a spending decision, not a benchmark. A hundred rows is not a finding, any more than the twenty-seven rows in the first post were. The same run showed why: on colour alone the model appeared to lead 0.7302 to 0.6296, from 27 eligible examples. Tested again on 400 balanced rows it scored 0.5537 against our 0.6245, a paired interval of [-0.1180, -0.0228]. The lead did not survive contact with a larger sample.

A full-scale frontier comparison, run on complete tracks rather than a checkpoint, is the first thing on our list. Until then we say what we measured and how little of it there was.

## The headline number is not the whole story

On `crop`, 0.6300 means about 63% of individual attribute predictions match the label exactly. On 49.66% of garments, every one of the fifteen fields is right.

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

Telling a coat from a dress is close to solved. Telling denim from twill is not: material scores 0.4148 on 146 labelled cases, making it both the hardest field and the thinnest evidence. Read the field-level table before relying on the headline.

## The design choices behind the model

**One sequence became many heads.** The Florence-2 system in the first post generated every attribute in one autoregressive response. That made unrelated fields interfere with one another, let common labels crowd out rare ones, and gave the model no way to abstain. We replaced it with a frozen vision encoder and separate heads for each field.

**Applicability is its own decision.** Every conditional field has a binary head that asks, "Does this apply here?" before a value head runs. We score this separately in the `fullbody` track. A model that invents a neckline for trousers should not get credit for confidence.

**Thresholds are calibrated, not guessed.** Calibration happens on a split that never touches development or test data. Multi-label fields such as material use an asymmetric focal loss, with balancing capped so common positives do not swamp rare ones.

**The backbone was tested rather than assumed.** With an identical development protocol, general SigLIP-2 scored 0.6018 and the fashion-pretrained option scored 0.6163. The experiment cost about eleven cents and settled the decision before it became an argument.

**The served encoder is ours.** Earlier checkpoints used our heads with a frozen third-party fashion encoder. We distilled that system into our own encoder, which serves today. The third-party encoder appears only as a distillation teacher during training and as the `crop` comparator, never as the served model.

## What we are releasing

| Model | Input | Weights | Commercial use |
|---|---|---|---|
| **MODA_NER(V) Crop** | Cropped garment | MIT | Yes |
| **MODA_NER(V) Catalog** | Catalogue product image | CC BY-NC 4.0 | No |
| **MODA_NER(V) Full-body** | Full-body photo | CC BY-NC 4.0 | No |

Two tracks are evaluated against research-only corpora whose terms extend to derived data, so the weights trained on them are non-commercial. That restriction applies to us too: those exact weights are not part of our paid product, and we verified before publishing that no serving path loads them.

We are not holding back a better model. The published `crop` checkpoint is the checkpoint that produced 0.6300. Download it, score it, and you should reproduce that number.

Running one takes a route name and a folder:

```bash
python models/inference.py --route crop --model-dir . --images photo.jpg
```

Each route expects a different kind of picture. `crop` wants a single garment already cut out, `catalog` a clean product shot, `fullbody` a person. Feeding a route the wrong kind of image is the most common reason results look worse than the numbers above.

**MODA_NER(T)**, the text model, is not distributed. Its benchmark is. The track includes a builder that recreates the splits from two public sources, a scorer, and our score. If you want to beat 0.8723 at span extraction, the track is available.

## Reproduce the results

We do not ship gold labels. You obtain every source corpus under its own terms, and the builders recreate the frozen split from record IDs and checksums. We publish the protocol, scorers, uncertainty code, and our prediction files with hashes, including the runs we lost.

```bash
python -m suite.crop.score \
  --gold <your rebuilt split> \
  --predictions results/crop/moda-ner-v-crop/evaluation_predictions.jsonl \
  --output /tmp/recomputed.json
```

We regenerated every figure in this post this way before publishing. This includes all three `fullbody` confidence intervals, calculated with a 10,000-sample bootstrap clustered over 1,751 product groups. The `crop` and `catalog` scorers require only the Python standard library. The `fullbody` scorer needs numpy and torch, because its paired bootstrap is vectorised and we kept the original implementation rather than rewriting it, so the numbers match the published run rather than approximating it. Rebuilding the `text` splits needs the `datasets` library.

## Run your own model

Freeze your checkpoint. Predict on the same record IDs without reading labels. Commit the prediction hash. Score each track and report the paired interval against our published predictions.

If another team beats us under this protocol, we want to know. It is better to find a weakness in a reproducible test than in a customer's production pipeline.

*Next: what none of this tells you about your own catalogue.*
