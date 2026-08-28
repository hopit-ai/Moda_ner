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

Colour and fit are not included in `crop`, because its source annotations do not cover them. They are evaluated in `catalog`, which has ten fields. `fullbody` has eighteen fields and an explicit not-applicable class. `text` defines thirteen entity types in product copy, twelve of which appear in its test split.

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

## What about the large vision-language models?

It is the first question people ask, so here is what we ran and what it does and does not show.

We tested **Gemini 3.5 Flash** in August 2026 on 100 rows shared across all three tracks, with a continuation rule fixed before the run: keep paying only if it stays competitive. It did not.

| Track (100-row checkpoint) | Gemini 3.5 Flash | MODA |
|---|---:|---:|
| `crop` attribute micro-F1 | 0.3624 | **0.6667** |
| `catalog` field-macro set F1 | 0.5830 | **0.8057** |
| `fullbody` Tier-1 macro-F1 | 0.3522 | **0.4482** |

The gate failed, so we stopped and did not buy the remaining 900 calls.

Three honest qualifications. A hundred rows is a checkpoint, not a benchmark - the same standard that made us discard a twenty-seven-row result in the first post applies here to our own favour. Flash is the cost-optimised tier rather than the top one, so this says nothing about what a frontier-tier model would do. And a prompt-only model competing against a schema-routed system trained for this exact task is not a fair fight in either direction; it is a question about whether you need a specialist at all.

That same run showed why the sample size caveat matters. On colour alone Gemini appeared to lead 0.7302 to 0.6296 - from 27 eligible examples. Tested again on 400 balanced rows it scored 0.5537 against our 0.6245, a paired interval of [-0.1180, -0.0228]. The lead did not survive a larger sample.

Running five or more modern models on complete tracks is the first thing on our list. Until that happens, this is what we measured and how little of it there was.

## What it costs to run

Accuracy is one axis. Cost and latency separate the two approaches more sharply, and in the
opposite direction to what the headline scores suggest.

The thousand-row VLM run was budgeted before it started at a projected $18.21 per thousand
images. That figure is an internal route estimate rather than a provider guarantee, which is
why the gate was written as a hard spending ceiling rather than a forecast. We stopped at a
hundred rows and spent about a tenth of it.

The published routes have no per-call cost at all. You download the weights and run them on
hardware you already have, so the marginal cost of the ten-thousandth image is electricity.
On our own serving path, inference plus post-processing takes 0.0228 s at p95 for the
full-body route at concurrency one, and the contract check that validates every response
costs about 0.1 ms.

That is the real shape of the decision. A prompt-only model is faster to start and cheaper at
ten images. A specialist is cheaper at ten thousand and answers in tens of milliseconds
rather than a network round trip.

One thing worth stating plainly, because it is the obvious next question. Our hosted tier
serves these same routes. For a given schema its accuracy is that route's published score,
not a better one, and there is no private checkpoint held back. What it adds is the loop
around the model: resolving a route from a caller-declared schema, per-field calibration, and
mapping to a retailer's own taxonomy. The benchmark above is the evidence that the loop
works, not a teaser for something stronger.

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

## The text track, and the sharpest example of the same problem

We have been quiet about `text` so far, which is unfair to it. It scores 0.8723 strict span F1
on 1,071 rows of product copy, at 0.8838 precision and 0.8610 recall, and it is the highest
headline number in the suite. It is also the clearest demonstration of why we do not trust
headline numbers.

Thirteen entity types are defined. Twelve have a test score; `BRAND` has no instances in the
test split at all, which is itself worth knowing before anyone relies on it.

| Entity | F1 | Entity | F1 |
|---|---:|---|---:|
| `FIT` | 1.0000 | `GARMENT_TYPE` | 0.8840 |
| `HEMLINE` | 0.9892 | `NECKLINE` | 0.8571 |
| `PATTERN` | 0.9767 | `DETAIL` | 0.6396 |
| `MATERIAL` | 0.9513 | `COLOR` | 0.6341 |
| `SLEEVE` | 0.9064 | `SILHOUETTE` | 0.1538 |
| `OCCASION` | 0.9062 | `AESTHETIC` | 0.0000 |

A headline of 0.8723 contains an entity the model gets right every time and an entity it never
gets right at all. Split the schema the way the model actually behaves and the picture is
plain: concrete entities score 0.8615, abstract ones 0.5150. Material and hemline are written
on the label. Aesthetic and silhouette are judgements, and the model has not learned them from
product copy.

This track sits outside the claim on purpose. No external system has been evaluated on it
under the same conditions, so there is nothing here to be best of.

## The design choices behind the model

The Florence-2 system in the first post generated every attribute in one autoregressive response. Unrelated fields interfered with one another, common labels crowded out rare ones, and the model had no way to abstain. We replaced it with a frozen vision encoder and separate heads for each field.

Applicability became its own decision. Every conditional field has a binary head that asks whether the attribute applies at all, and it runs before any value head. We score that separately in the `fullbody` track. A model that invents a neckline for trousers should not get credit for confidence.

Thresholds are calibrated rather than guessed, on a split that never touches development or test data. Multi-label fields such as material use an asymmetric focal loss, with balancing capped so common positives do not swamp rare ones.

We tested the backbone rather than assuming it. Under an identical development protocol, general SigLIP-2 scored 0.6018 and the fashion-pretrained option 0.6163. The experiment cost about eleven cents and settled the decision before it became an argument.

The served encoder is ours. Earlier checkpoints paired our heads with a frozen third-party fashion encoder, and we distilled that system into our own encoder, which serves today. The third-party encoder appears only twice: as a distillation teacher during training, and as the `crop` comparator. It is never the model being served.

## What we are releasing

| Model | Input | Weights | Commercial use |
|---|---|---|---|
| **MODA_NER(V) Crop** | Cropped garment | MIT | Yes |
| **MODA_NER(V) Catalog** | Catalogue product image | CC BY-NC 4.0 | No |
| **MODA_NER(V) Full-body** | Full-body photo | CC BY-NC 4.0 | No |

Two tracks are evaluated against research-only corpora whose terms extend to derived data, so the weights trained on them are non-commercial. That restriction applies to us too: those exact weights are not part of our paid product, and we verified before publishing that no serving path loads them.

We are not holding back a better model. The published `crop` checkpoint is the checkpoint that produced 0.6300. Download it, score it, and you should reproduce that number.

Running one takes three commands from a fresh clone:

```bash
pip install -r requirements-inference.txt
huggingface-cli download HopitAI/moda-ner-v-crop --local-dir ./moda-ner-v-crop
python models/inference.py --route crop --model-dir ./moda-ner-v-crop --images photo.jpg
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

## Where everything lives

- Code, scorers, prediction files and hashes: [github.com/hopit-ai/Moda_ner](https://github.com/hopit-ai/Moda_ner)
- Benchmark tables and protocol: [hopit-ai.github.io/Moda_ner](https://hopit-ai.github.io/Moda_ner/)
- Weights: [MODA_NER(V) Crop](https://huggingface.co/HopitAI/moda-ner-v-crop) (MIT), [Catalog](https://huggingface.co/HopitAI/moda-ner-v-catalog) and [Full-body](https://huggingface.co/HopitAI/moda-ner-v-fullbody) (CC BY-NC 4.0)
- Both our benchmark suites: [hopit-ai.github.io](https://hopit-ai.github.io/)

*Next: what none of this tells you about your own catalogue.*
