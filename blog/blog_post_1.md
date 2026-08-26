# Why we built the test system before the model

**MODA_NER, Part 1: the MODA General Attribute Suite**

*Series: Fashion attribute extraction from images*  
*Sibling series: [Building a fashion search engine from scratch](https://github.com/hopit-ai/Moda)*

Fashion images contain data that retailers need: category, colour, sleeve length, neckline, fit, pattern, and material. Extracting those attributes helps power catalogues, search, recommendations, and trend analysis.

The hard part is not making a model give a believable answer. The hard part is knowing whether that answer is reliable.

Does the model work on both catalogue and full-body photos? Does it know when an attribute is not visible? Did a training change actually improve the system, or just move a score around?

Those questions are why we built the test system before building more models.

## Why one score is not enough

We could not find a single existing benchmark for this job. Fashion datasets cover different images and different attributes, and they use different labels.

Fashionpedia has detailed expert labels but no colour or fit. Shopping100k includes colour and fit, but only on catalogue images. DeepFashion-MultiModal (DFMM) includes full-body images and supports "not applicable," but uses a different label system. General vision-language tests focus on captions, not structured product data.

Combining all of them into one score would hide too much. A model can be strong on clean product shots and weak on full-body images, yet still look good in an average.

So we built four separate tests:

| Track | Input | What it checks |
|---|---|---|
| `crop` | Cropped garment image | Detailed garment attributes |
| `catalog` | Catalogue product image | Colour, fit, and retail-style images |
| `fullbody` | Full-body photo | Attributes that may be absent or out of view |
| `text` | Product title or description | Attributes in product copy |

We report each result separately. A good score in one track does not cancel a bad result in another.

## What changed our approach

We did not begin with a perfect evaluation process. We began with a model.

Our first serious system used Florence-2 to generate all attributes as one JSON response. A second training run looked roughly unchanged if you only watched recall: 0.6013 became 0.5885.

But precision fell from 0.5661 to 0.3158.

The model had started guessing more often. It still performed well on easy fields, which made the overall score look less alarming than it was. Field-by-field evaluation exposed the regression.

We changed the model design: instead of generating every attribute in one long sequence, it now uses separate heads and explicitly decides whether each field applies. More importantly, we paused model work and built a rigorous way to test it.

## How we keep the test honest

The harness follows five simple rules.

1. **Freeze the test before running the model.** A manifest records the data version, exact examples, allowed answers, scoring rules, and random seed.
2. **Keep related images together.** A single source photo can produce several crops, and a product can have several photos. We prevent related examples appearing in both training and test data.
3. **Predict before seeing the answers.** We hash every prediction file before scoring it, so it cannot be replaced after labels are revealed.
4. **Score every row.** Missing or unsupported predictions are reported instead of silently dropped. The scorer also distinguishes "wrong" from "not visible."
5. **Do not hide weak results in an average.** A model must show a real improvement on every required track, not just win on average.

This is not exotic research infrastructure. It is a practical way to stop a benchmark from flattering its owner.

## What MODA can do today

On the `crop` track, MODA scores **0.6300**. Across all fifteen attributes, about 63% of individual predictions exactly match the dataset label. On **49.66%** of garments, it gets every field right.

The fifteen fields are three category levels (`master_category`, `category`, `sub_category`), shape (`silhouette`, `hemline`, `waist_type`), sleeves (`sleeve_length`, `sleeve_shape`), neck and collar (`neckline`, `collar_presence`, `collar_style`), surface (`material`, `surface_treatment`, `pattern`), and `closure_type`. Each has a fixed vocabulary published with the track - 21 possible silhouettes, 21 necklines, and so on - because a test cannot fairly judge an answer whose allowed values are secret.

That headline hides important differences between them:

| Stronger areas | Harder areas |
|---|---|
| Master category: 0.9215 | Material: 0.4148 |
| Category: 0.8825 | Surface treatment: 0.4398 |
| Pattern: 0.8356 | Collar style: 0.4566 |
| Sleeve length: 0.8073 | Neckline: 0.4650 |

Category recognition is relatively strong. Material, neckline, collar style, and surface treatment still need work. If your workflow depends on those fields, this model is not yet ready for that use case.

## Results across image tasks

| Track | What the score measures | MODA | Comparator | 95% range of MODA's advantage | Audit now? |
|---|---|---:|---:|---:|---|
| `crop` | Share of predictions matching exactly, across 15 fields | **0.6300** | 0.6245 | [+0.0014, +0.0097] | **Yes** |
| `catalog` | Score per field, averaged over 10 fields | **0.8292** | 0.6657 | [+0.1595, +0.1676] | Not yet |
| `fullbody`: overall | Score per field, averaged over 18 fields | **0.6917** | 0.5943 | [+0.0891, +0.1053] | Not yet |
| `fullbody`: knows when not to answer | How reliably it says an attribute is not there | **0.6637** | 0.6088 | [+0.0433, +0.0657] | Not yet |
| `fullbody`: when visible | The 18-field average, on cases where the attribute is present | **0.5785** | 0.4969 | [+0.0723, +0.0905] | Not yet |

Do not compare scores across rows. The tracks use different images, attributes, and scoring methods. Each row only says how MODA performed against its comparator on that specific task.

The `catalog` and `fullbody` comparator is FashionCLIP 2.0 with matched supervised heads. The `crop` comparison is different: it holds our architecture and training constant, changing only the encoder to FashionSigLIP. It is an encoder comparison, not a claim that we beat another company's product.

Only `crop` is fully auditable today. Its weights, scorer, and prediction files are published, so once you have obtained the source dataset you can recompute 0.6300 yourself and tell us if we are wrong. We are opening the remaining tracks as their builders and evidence files are published.

The `text` track is not part of the comparative claim because we have not yet tested another system on it under the same conditions.

The narrow claim our current evidence supports is:

> MODA General is the best of the named open systems evaluated under the frozen public MODA General Attribute Suite, spanning localized garment crops, catalog product images, color and fit, and applicability-aware full-body attributes.

That does not mean world-best, human-level, or ready for every production use case.

## What the harness has already caught

The harness made us publish a loss. In an early full-body evaluation, FashionCLIP with matched heads beat our standalone system, and a hybrid beat both. We kept that result, chose the next candidate using development data, built a fresh test split, and evaluated it once.

It also stopped us from overreacting to a small sample. A commercial model looked better on colour, 0.730 versus our 0.630, but this came from only 27 rows. It did not pass our broader 100-row checkpoint. When we later tested colour on 400 balanced rows, the apparent advantage disappeared.

Twenty-seven rows are a reason to investigate, not a conclusion.

## The problem an internal test cannot solve

Our largest recent issue came from outside the suite.

On an independent human-labelled set of 1,110 images, our model scored 0.7169 versus FashionCLIP's 0.6605. But our production composite route scored 0.5764 on those same images, about 14 points lower.

The model had not regressed. The datasets divided neckline types differently, and our mapping between those label systems lost a real distinction. Our internal test could not catch this because its labels were internally consistent.

That is the limit of an internal benchmark: it can measure a clearly defined task well, but it cannot prove that the task definition matches every retailer's taxonomy.

## Open the ruler

Our approach is simple: open the ruler; sell what the ruler measures.

We publish the protocol, splits, scorers, uncertainty code, prediction files, and hashes, including losing runs. We do not redistribute third-party datasets; users obtain them from their original sources and our builders recreate the frozen test splits.

What remains closed is customer-specific work: decision thresholds, model-routing logic, and mappings to a retailer's own taxonomy. Two tracks use research-only data, so weights trained on them are non-commercial. Those exact weights are not part of our paid product.

Next, we will add independent human labels on a public track, run more modern open models at full test size, and seek a holdout built and administered by someone else.

## Run your own model

Freeze a checkpoint. Predict without seeing labels. Commit the prediction hash. Score each available track and report the comparison range against our published predictions.

The suite opens one track at a time, starting with `crop`. The published `crop` checkpoint is MIT-licensed, commercially usable, and is the checkpoint that produced the score above. The `catalog` and `fullbody` weights are non-commercial.

If another team beats us under this protocol, we want to know. It is better to find a weakness in a reproducible test than through a customer's production pipeline.

*Next: what a benchmark win does not tell you: independent human gold, the neckline taxonomy mismatch, and why our production system is calibrated on none of the public data.*
