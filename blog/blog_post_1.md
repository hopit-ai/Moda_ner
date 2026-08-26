# Why we built the evaluation harness before the model

**MODA_NER, Part 1: the MODA General Attribute Suite**

*Series: Fashion attribute extraction from images*  
*Sibling series: [Building a fashion search engine from scratch](https://github.com/hopit-ai/Moda)*

If you need to turn a fashion image into structured data, you need more than a model that produces plausible answers. You need to know when it is right, where it fails, and whether a change actually improves it.

That is why we are starting this series with an evaluation harness rather than a model announcement.

The task sounds straightforward: given a photo of a dress, identify its category, sleeve length, neckline, colour, fit, pattern, and material. In practice, those fields power cataloguing, search, personalisation, trend analysis, and every other downstream system that needs to understand an item. A model that is merely impressive in a demo is not enough. We need to answer practical questions:

- Which vision model should we use?
- Does it work on product shots, street photos, and cropped garments?
- Is a higher score real, or an artefact of the test split?
- Does it know when an attribute is not visible or does not apply?
- Is the gain large enough to justify the cost at runtime?

We looked for an existing benchmark that answered those questions and did not find one. There are many fashion datasets, but they cover different attribute sets, image styles, and labeling conventions. Their taxonomies often disagree.

For example, Fashionpedia provides expert garment-level annotations, but not colour or fit—two fields merchandisers care about deeply. Shopping100k includes colour and fit, but only for catalogue images and without a way to say that a field is inapplicable. DeepFashion-MultiModal (DFMM) includes full-body images and an explicit N/A class, but uses a closed vocabulary. General vision-language benchmarks mainly test captioning and visual question answering, not production-oriented attribute extraction.

We could have mapped everything into one label space and reported one headline score. We considered it several times. We abandoned the idea because every mapping would embed a modeling decision. A system that performs well on flat product images but fails on street photography could still look good in an averaged result.

Instead, we built three tracks, three test sets, and three metrics. We do not average them into one number. A system is only as useful as its weakest important component.

## What the suite contains

The MODA General Attribute Suite is not a model. It is a reproducible way to evaluate models.

It has four parts.

**Track definitions.** Each track fixes the dataset version, test IDs, output vocabulary, acceptable aliases, rules for missing labels, metric, bootstrap procedure, and random seed. In other words, a track is a contract: this is what the model sees, and this is how its output will be judged.

| Track | Frozen test set | Input | Leakage unit |
|---|---:|---|---|
| Fashionpedia MODA-15 | 4,688 garment crops from 1,158 images | Garment crop supplied by the evaluator | Source image |
| Shopping100k-10 | 9,995 catalogue images; 61,384 eligible fields | Catalogue product image | Image |
| DFMM-18 | 5,000 images across 1,751 product groups | Full-body photo | Product group |

**Manifest builders.** Some source datasets are restricted, so we do not redistribute them. Instead, users download them from the official source under its terms. Our builders reconstruct the exact test split from record IDs and checksums. If a local dataset copy differs from ours, the build stops instead of silently evaluating against a different dataset.

**Scorers and uncertainty estimates.** Each track has a strict scorer and uses 10,000 paired bootstrap resamples, clustered at the correct leakage unit. This produces uncertainty intervals that account for related examples—for instance, multiple garment crops from one source image.

**Frozen evidence.** We publish the prediction files for every evaluated system, along with SHA-256 hashes and result JSONs. That includes systems that beat us and runs we lost. The point is to make the numbers checkable, not merely assertable.

There are 147 focused tests in the suite. Most exist because an earlier version of the evaluation failed to catch something important.

## Why the harness mattered

We did not begin with this level of discipline. Like many teams, we had models before we had a reliable benchmark.

Our first serious system used Florence-2 to produce a JSON object containing every attribute. It worked well enough to justify another training pass. After that pass, recall barely changed: from 0.6013 to 0.5885. On a blended score, that could easily look close enough to ship.

But precision fell from 0.5661 to 0.3158.

Per-field scoring showed what the aggregate hid: the model had started guessing much more often. Improvements on easy fields masked a collapse on harder ones. We were close to shipping a regression because a summary number made it look harmless.

That result changed two things.

First, it led to a technical diagnosis. A single autoregressive sequence that emits every field couples attributes that should be independent, crowds out rare labels, and gives the model no clean way to say that a field does not apply. We replaced that approach with conditional heads and an explicit applicability decision.

Second, we paused model work and built the evaluation suite.

The tracks were created incrementally, each to cover a blind spot in the last one. Fashionpedia came first because its expert mask-level annotations were the highest-quality available signal. It could not answer questions about colour or fit, so we added Shopping100k. Neither dataset distinguishes an absent or inapplicable attribute from a negative label, so we added DFMM, where N/A is a real class.

None of the tracks was planned upfront. Each exists because the earlier tracks could not answer a question our users needed answered.

## What happens when a model is scored

The workflow has five steps.

### 1. Build a fixed manifest

Before a model runs, the track builder pins the dataset version, checksums, records, groups, split roles, output vocabulary, aliases, labeling rules, metric, bootstrap method, and seed.

```
python -m suite.fullbody.build_manifest --source data/dfmm
# manifest: 5,000 test images, 1,751 product groups, 18 fields, sha256=...
```

This is deliberately unglamorous. The manifest prevents accidental changes to what is being measured.

### 2. Split and resample at the point where leakage occurs

Randomly splitting individual rows can inflate a fashion-model score. One Fashionpedia image may yield several garment crops; one DFMM product may appear in several photos. If related examples appear in both training and test data, the model is partly being tested on something it has effectively already seen.

The same logic applies to confidence intervals. Treating related crops as independent makes the interval too narrow because they tend to succeed and fail together.

Each track therefore splits and resamples at its natural unit: source image for Fashionpedia, image for Shopping100k, and product group for DFMM. The DFMM test is especially strict: its 5,000 images cover 1,751 product groups with no record or product-group overlap against any prior experiment.

Training selects weights. Calibration selects thresholds. Development selects checkpoints. The frozen test is read once, at the end.

### 3. Predict without labels, then commit the results

Inference receives images and the schema, but never the labels. When it finishes, the prediction file is hashed before scoring.

```
python -m suite.fullbody.commit --predictions preds.jsonl
# committed: preds.jsonl  sha256=9f3a...  rows=5000
```

The hash becomes part of the record. That prevents either side from swapping predictions after seeing the result.

### 4. Score strictly and account for observability

The scorer fails closed. A missing row, duplicate ID, unexpected record, unknown field value, or value the alias map cannot resolve is reported explicitly. It never quietly drops hard rows.

One rule is particularly important: an attribute that cannot be observed is not a negative example. If the waist is cropped out, predicting `waist: N/A` should not be treated as an error. Fashionpedia evaluates applicability separately for conditional fields; Shopping100k excludes unlabelled cells; and DFMM scores N/A as a class with its own F1.

That distinction prevents a model from getting credit for confidently inventing attributes on every garment.

### 5. Use paired bootstrap intervals and a conjunctive gate

Every score includes 10,000 paired bootstrap resamples clustered at the track's leakage unit. A promotion requires the lower bound of the improvement interval to clear zero on every required track. There is no weighted average where a weak result can hide behind strong ones elsewhere.

None of this is exotic. It is simply the collection of places where benchmarks tend to mislead their owners, closed one by one.

## What the harness has caught

The harness rarely produces dramatic revelations. More often, it prevents us from believing a convenient story.

It stopped an unnecessary backbone swap for the price of a coffee. We compared a general SigLIP-2 encoder with a fashion-pretrained encoder under the same development protocol. SigLIP-2 reached 0.6018 development micro-F1, versus 0.6163 for the fashion-pretrained option. The decision was quick, cheap, and did not depend on anyone's preference.

It also made us publish a loss. On the first full-body evaluation, FashionCLIP with matched supervised heads outperformed our standalone model, and a hybrid did better than both.

| System (earlier matched protocol) | Tier 1 | Tier 2 | Tier 3 |
|---|---:|---:|---:|
| MODA distilled, standalone | 0.5398 | 0.5260 | 0.4509 |
| FashionCLIP + linear heads | 0.5985 | 0.6237 | 0.5008 |
| MODA-FashionCLIP hybrid | **0.6492** | **0.6425** | **0.5353** |

We retained that result. We then selected the next standalone candidate using development data only, built a fresh test split with no overlapping product groups, and evaluated it once.

The suite also prevented us from overinterpreting a commercial VLM result. The model appeared to beat us on colour—0.730 versus 0.630—but the figure came from only 27 eligible rows. A 100-row checkpoint gate let us avoid paying for the remaining 900 calls after the model lost the broader comparison. We later tested colour on 400 balanced rows and the apparent advantage disappeared; the paired interval was entirely negative.

Twenty-seven rows are not a finding. They are a lead worth testing.

## Current results—and their limits

| Track / primary metric | MODA | Strongest comparator | Paired 95% CI |
|---|---:|---:|---:|
| Fashionpedia attribute micro-F1 | **0.6300** | FashionSigLIP 0.6245 | [+0.0014, +0.0097] |
| Shopping100k field-macro set F1 | **0.8292** | FashionCLIP 0.6657 | [+0.1595, +0.1676] |
| DFMM Tier-1 macro-F1 | **0.6917** | FashionCLIP 0.5943 | [+0.0891, +0.1053] |
| DFMM Tier-2 N/A-F1 | **0.6637** | FashionCLIP 0.6088 | [+0.0433, +0.0657] |
| DFMM Tier-3 visible macro-F1 | **0.5785** | FashionCLIP 0.4969 | [+0.0723, +0.0905] |

The DFMM Tier-1 result is the fresh rematch described above. The earlier loss remains in the record because it is part of the evidence that the later win was not selected after the fact.

The precise claim these results support is:

> MODA General is the best of the named open systems evaluated under the frozen public MODA General Attribute Suite, spanning localized garment crops, catalog product images, color and fit, and applicability-aware full-body attributes.

That is intentionally narrower than "world-best," "universal state of the art," "human-level," or "production ready." The harness helps us state exactly what the evidence supports—and no more.

Several fields still need work. On Fashionpedia, material achieves 0.4148 value-F1. On Shopping100k, fabric reaches 0.6693 and fit 0.6708. Collar style, neckline, and rare applicability values also lag. Passing every track means no track failed; it does not mean every attribute is ready for every production use case.

## The limitation the harness did not catch

The most important recent failure was outside the harness's reach.

On an independently human-labelled external set of 1,110 images that nobody on our team had handled, our system achieved 0.7169, compared with 0.6605 for FashionCLIP. The interval was clearly positive.

But the production composite route scored only 0.5764 on the same images—a drop of roughly 14 points.

The cause was not a model regression. The two datasets divided neckline categories differently, and our taxonomy mapping flattened a meaningful distinction. Internal evaluation could not expose the problem because the taxonomy was consistent within the internal protocol. Independent labels did.

That is the boundary of this harness: it can rigorously measure the system it defines, but it cannot prove that its taxonomy matches every real-world taxonomy. The next post will examine that gap in more detail.

## What comes next

We are improving the suite in four directions.

**Independent human gold labels on a public track.** We have committed 1,993 Fashionpedia image groups for independent annotation, targeting at least 4,000 garment rows across at least 1,000 groups. Predictions and thresholds will be committed before labels are opened. Until that work is complete, none of these results should be described as human-gold.

**More full-size comparisons.** Current external comparisons include FashionCLIP, FashionSigLIP, one open VLM, and one cost-gated commercial model. We want to evaluate at least five additional modern open VLMs at full row counts, not merely at a preliminary checkpoint.

**A fourth track.** We have built a streaming iMaterialist track for pattern and neckline. It fingerprints and deduplicates examples without persisting an image corpus. It will only be included if at least 2,000 deduplicated source URLs remain available; otherwise, we will drop it and report why.

**An independently administered holdout.** The strongest possible test set is one we neither build nor administer. That is the most direct way to address the concern that a benchmark author may design a test they pass.

We are also preparing a paper describing the suite, which will become the preferred citation when available.

## Why the benchmark is open while some model work is not

Our position is simple: open the ruler; sell what the ruler measures.

We will publish the protocol, splits, scorers, bootstrap code, frozen prediction files, and hashes—including the losing runs. A benchmark that others cannot run is just an opinion accompanied by a table. It becomes useful when others can inspect it, challenge it, and beat it.

What remains closed is the output of applying the measurement process to particular data: calibrated field-level thresholds, routing that chooses a head for a declared schema, and retailer-specific taxonomy mappings. These are accumulated, data-specific decisions. They are also the practical work customers are paying us to do.

Two of the three tracks are based on research-only datasets, so weights trained on them are non-commercial. We will respect those terms. Those exact weights are not part of our paid product, either. The components needed to reproduce the evaluation—protocols, scorers, splits, and prediction files—do not carry that restriction.

## Run your own model

Freeze a checkpoint. Generate predictions without labels on the same IDs. Commit the prediction hash. Score each track. Report the paired confidence interval against our published predictions.

Everything needed is in the repository.

If another team beats us under the protocol, we want to know. The alternative is learning from a customer after a taxonomy mismatch has quietly removed fourteen points from a production route. We have had that conversation with ourselves enough times to know that the evaluation harness is not overhead. It is how we keep the unknowns visible.

*Next: what a benchmark win does not tell you—independent human gold, the neckline taxonomy mismatch, and why our production system is calibrated on none of the public data.*
