# Why we tested the measurement system before the model

**MODA_NER, Part 1: the MODA General Attribute Suite**

*Series: Fashion attribute extraction from images*  
*Sibling series: [Building a fashion search engine from scratch](https://github.com/hopit-ai/Moda)*

When a computer looks at a fashion image, we want it to return useful facts: what the garment is, its colour, sleeve length, neckline, fit, pattern, and material.

That sounds simple. It is not.

Those facts feed product catalogues, search, recommendations, trend analysis, and other systems. So it is not enough for a model to give answers that look convincing in a demo. We need to know how often it is right, where it fails, and whether a new model is genuinely better than the last one.

This is why we are starting a series about fashion attribute extraction with the test system, not the model itself.

## The problem with a single benchmark score

We looked for an existing benchmark that could answer the questions we care about. We did not find one.

Fashion datasets differ in the images they contain, the attributes they label, and the words they use for those attributes. Their label systems do not neatly line up.

For example:

- Fashionpedia has detailed expert labels for garments, but does not cover colour or fit.
- Shopping100k includes colour and fit, but uses catalogue photos and cannot express that a field does not apply.
- DeepFashion-MultiModal (DFMM) includes full-body photos and has an explicit "not applicable" label, but uses its own fixed set of terms.
- General vision-language benchmarks usually test captioning or visual question answering. They do not test the structured product data a retailer needs.

We could have forced these datasets into one shared label system and published one impressive number. But that would hide important trade-offs. A model that performs well on clean product images but poorly on street photos could still look good after everything is averaged together.

So we built four separate tests instead. We report their results separately, because a good overall score should not hide a serious weakness.

## What we built

The MODA General Attribute Suite is not a model. It is a repeatable way to test models.

It has four tracks:

| Track | What it tests | Test set | What counts as a related example |
|---|---|---:|---|
| `crop` | Attribute extraction from a garment crop | 4,688 crops from 1,158 images | The original source image |
| `catalog` | Attribute extraction from catalogue images | 9,995 images; 61,384 labelled fields | The image |
| `fullbody` | Attribute extraction from full-body photos | 5,000 images across 1,751 product groups | The product group |
| `text` | Attribute extraction from titles and descriptions | 1,071 rows | The row |

The `crop` track covers category, shape, neck and collar, surface, and construction details. Its full label list is public. That matters: a benchmark cannot fairly judge a prediction if its allowed answers are secret. The `catalog` track includes ten fields, including colour and fit. The `fullbody` track has eighteen fields and includes an explicit N/A option. The `text` track identifies thirteen types of information in product titles and descriptions.

The suite has four main pieces.

**Fixed test definitions.** For every track, we fix the data version, test IDs, allowed answers, accepted alternate spellings, scoring rules, and random seed. This is a contract: it defines what a model will see and how its answer will be judged.

**Builders for the test data.** We do not ship gold labels, on any track. Users obtain each dataset from its original source under that source's terms, and our builders then recreate the exact test split using IDs and checksums. If a local copy does not match the expected version, the process stops instead of quietly producing a different test. What we do ship is the protocol, the scorers, and our own prediction files, which is what lets you recompute our numbers once you have the data.

The `text` track differs in one way: we created its labels, so instead of pointing at a third-party corpus its builder regenerates them from their public sources. We publish the test and our score on it, but not the model checkpoint behind that score.

**Strict scoring.** Each track has a scorer that checks every expected row and reports problems instead of ignoring them. We also calculate a range around every comparison, so readers can see whether an apparent improvement is likely to be real.

**Published evidence.** We publish the prediction files, their SHA-256 hashes, and the results for all systems we evaluate. That includes runs we lost. The goal is to let others check the numbers, not simply take our word for them.

The suite ships with its own tests, including one that re-scores our published predictions and fails if they stop reproducing the number we published. Most of the others exist because an earlier version of the evaluation exposed a way we could have fooled ourselves.

## Why this mattered to us

We did not start with a perfect evaluation process. Like many teams, we started with models.

Our first serious system used Florence-2 to produce a JSON object containing all of a garment's attributes. It was promising enough that we trained it again. Recall changed only a little, from 0.6013 to 0.5885. Looking only at a blended number, that might have seemed close enough to ship.

But precision dropped from 0.5661 to 0.3158.

In plain terms, the updated model had started guessing much more often. It still did well enough on easy fields that an overall score could hide the damage. A field-by-field view showed a real regression.

That taught us two things.

First, asking one autoregressive model to generate every field in a single sequence was a poor fit for the task. It linked attributes that should be independent, made rare labels easier to miss, and gave the model no clean way to say "this does not apply." We moved to separate conditional heads with an explicit applicability decision.

Second, we stopped improving the model for a while and improved the way we measured it.

The tracks grew one at a time. We began with Fashionpedia because its expert garment labels were strong. It could not answer questions about colour and fit, so we added Shopping100k. Neither of those datasets captures whether an attribute simply does not apply, so we added DFMM. The text track covers a related but distinct task: extracting structured attributes from product copy.

Each track exists because an earlier one left an important question unanswered.

## How a model is tested

The process has five steps.

### 1. Lock the test before running the model

Before an evaluation begins, we create a manifest: a file that records the dataset version, checksums, records, groups, allowed answers, rules, metric, and seed.

```
python -m suite.crop.build_manifest --source <your local copy>
# manifest: 4,688 garment crops, 1,158 source images, 15 fields, sha256=...
```

This is not glamorous, but it prevents the target from shifting while we are measuring it.

### 2. Avoid accidentally testing on near-duplicates

Fashion data often contains related examples. One source image can produce several garment crops. One product can appear in several photos. If related examples end up on both sides of a train/test split, the model may look better than it really is.

We split at the right level for each dataset: source image for `crop`, image for `catalog`, and product group for `fullbody`. The `fullbody` test is especially strict: it contains 5,000 images from 1,751 product groups, with no overlap with earlier experiments.

We follow the same rule when estimating uncertainty. Related images tend to succeed or fail together, so treating them as independent would make our confidence look stronger than it is.

Training chooses model weights. Calibration chooses decision thresholds. Development data helps select a checkpoint. The frozen test is saved for one final evaluation.

### 3. Make predictions before seeing the answers

The model receives the images and the expected schema, but not the labels. When its run finishes, we hash the prediction file before scoring it.

```
python -m suite.crop.commit --predictions preds.jsonl
# committed: preds.jsonl  sha256=9f3a...  rows=4688
```

This creates a record of the exact predictions that were scored. Neither we nor anyone else can swap in a better file after seeing the answers.

### 4. Score every row, including the awkward ones

The scorer stops or reports an error if a row is missing, duplicated, unexpected, or contains an unsupported value. It does not quietly remove difficult cases.

It also distinguishes "wrong" from "not visible." If a photograph cuts off a garment's waist, predicting `waist: N/A` should not count as an error. The tracks handle this in ways that fit their labels: `crop` evaluates applicability separately where needed, `catalog` ignores fields that were never labelled, and `fullbody` treats N/A as a real answer that can be scored.

Without this rule, a model could confidently invent attributes for every garment and appear better than it is.

### 5. Require improvements across the important tests

We compare models using 10,000 paired bootstrap resamples. This is a standard way to estimate how uncertain a difference is while comparing both models on the same examples.

To promote a model, the lower end of its improvement range must be above zero on every required track. We do not use an average that lets a bad result disappear behind good results elsewhere.

## What the suite has caught

Most of the value is quiet. The suite stops us from convincing ourselves of things that are convenient but not well supported.

For example, we compared a general SigLIP-2 encoder with a fashion-pretrained encoder under the same development setup. SigLIP-2 reached 0.6018 micro-F1, compared with 0.6163 for the fashion-pretrained encoder. That was enough to make the decision without a long debate or a costly experiment.

The suite also made us publish a loss. In our first full-body evaluation, FashionCLIP with matched supervised heads outperformed our standalone model. A hybrid of the two did better than both.

| System (earlier matched protocol) | Tier 1 | Tier 2 | Tier 3 |
|---|---:|---:|---:|
| MODA distilled, standalone | 0.5398 | 0.5260 | 0.4509 |
| FashionCLIP + linear heads | 0.5985 | 0.6237 | 0.5008 |
| MODA-FashionCLIP hybrid | **0.6492** | **0.6425** | **0.5353** |

We kept that result in the record. We used development data to choose the next standalone candidate, built a new test split with no overlapping product groups, and ran the final evaluation once.

The suite also stopped us from overreacting to a small sample. A commercial vision-language model initially looked better on colour: 0.730 versus our 0.630. But that result came from only 27 eligible rows. The model did not clear our 100-row checkpoint on the broader task, so we did not pay for another 900 calls. When we later tested colour on 400 balanced rows, the apparent advantage disappeared.

Twenty-seven rows are not a finding. They are a reason to investigate.

## Our current results

| Track / main metric | MODA | Strongest comparator | Paired 95% interval |
|---|---:|---:|---:|
| `crop` attribute micro-F1 | **0.6300** | Same heads with FashionSigLIP encoder: 0.6245 | [+0.0014, +0.0097] |
| `catalog` field-macro set F1 | **0.8292** | FashionCLIP 2.0 + matched heads: 0.6657 | [+0.1595, +0.1676] |
| `fullbody` Tier-1 macro-F1 | **0.6917** | FashionCLIP 2.0 + matched heads: 0.5943 | [+0.0891, +0.1053] |
| `fullbody` Tier-2 N/A-F1 | **0.6637** | FashionCLIP 2.0 + matched heads: 0.6088 | [+0.0433, +0.0657] |
| `fullbody` Tier-3 visible macro-F1 | **0.5785** | FashionCLIP 2.0 + matched heads: 0.4969 | [+0.0723, +0.0905] |

One important caveat: the `crop` comparison is an encoder experiment, not a win over another vendor. We used our own architecture, heads, and training release in both cases, changing only the encoder. The FashionSigLIP version reached 0.6245. The genuine third-party baselines on that track are zero-shot models: Qwen3-VL-8B at 0.1805 and calibrated FashionSigLIP text prototypes at 0.1817. Those models were not built specifically for this task, which is why we do not make much of the gap.

The fuller `crop` picture is two wins and two ties against the FashionSigLIP-encoder variant. We win attribute micro-F1 by 0.0056 and master-category accuracy by 0.0085. Field-macro F1 and category accuracy are ties because their uncertainty ranges include zero. Micro-F1 gives more weight to common attributes; field-macro F1 gives every field equal weight. Reporting both helps show whether gains come only from the easiest fields.

The `fullbody` Tier-1 result is a new rematch. We keep the earlier loss because it is evidence that the later result was not chosen after we saw the answer.

The claim these results support is deliberately narrow:

> MODA General is the best of the named open systems evaluated under the frozen public MODA General Attribute Suite, spanning localized garment crops, catalog product images, color and fit, and applicability-aware full-body attributes.

This does not mean we are world-best, universally state of the art, human-level, or ready for every production case. It means exactly what the benchmark measured.

Some fields are still weak. On `crop`, material reaches 0.4148 value-F1. On `catalog`, fabric reaches 0.6693 and fit 0.6708. Collar style, neckline, and rare "not applicable" cases also need work. Passing every track does not mean every attribute is production-ready.

## A limitation our own tests missed

The most important recent problem came from outside the suite.

On an independently human-labelled set of 1,110 images that nobody on our team had worked on, our model scored 0.7169. FashionCLIP scored 0.6605, and the difference was clearly positive.

But our production composite route scored only 0.5764 on the same images, about 14 points lower.

This was not a model regression. The external dataset divided neckline categories differently, and our mapping between the two label systems removed an important distinction. Our internal benchmark could not reveal this because its labels were internally consistent. Independent labels could.

This is the limit of any internal benchmark: it can measure the system it defines carefully, but it cannot prove that its label system matches every retailer's or every real-world use case.

## What we will improve next

We are working on four additions.

**Independent human labels on a public track.** We have committed 1,993 Fashionpedia image groups for outside annotation, aiming for at least 4,000 garment rows across 1,000 groups. We will commit predictions and thresholds before opening the labels. Until then, none of the results above should be called human-gold.

**More full-size model comparisons.** We currently compare against FashionCLIP, FashionSigLIP, one open vision-language model, and one cost-limited commercial model. We want to run at least five more modern open models on the full test sets.

**A fifth track.** We have built a streaming test for pattern and neckline. It deduplicates examples without storing an image corpus. It will only be released if at least 2,000 usable, distinct source URLs remain available. If that condition is not met, we will drop the track and say why.

**A holdout run by someone else.** The strongest test would be a set that we do not build or administer. That is the clearest answer to the fair concern that a benchmark author could create a test that favours their own system.

We are also preparing a paper about the suite. It will be the preferred citation when it is available.

## Why we open the test but not every model detail

Our approach is simple: open the ruler; sell what the ruler measures.

We will publish the protocol, splits, scorers, uncertainty code, prediction files, and hashes, including losing runs. A benchmark that no one else can run is just an opinion with a table beside it. It becomes useful when other people can inspect it, challenge it, and improve on it.

What stays closed is the work produced when we apply that process to a particular customer: field-level thresholds, model-routing logic, and mappings to a retailer's own taxonomy. Those are data-specific decisions and part of what a customer is paying us to build.

Two of the four tracks use research-only datasets. Any weights trained on those data are non-commercial, and we will respect that restriction. Those exact weights are not used in our paid product. The materials needed to reproduce the evaluation - protocols, scorers, splits, and prediction files - do not have the same restriction.

## What you can download

For `crop`, the published weights are the same checkpoint that produced the number in the table above, and they are MIT-licensed and usable commercially. Download it, score it, and you should get 0.6300.

For `catalog` and `fullbody`, the published weights are non-commercial, because the datasets they were evaluated against are research-only. That restriction binds us too: those exact weights are not part of our paid product. Where any published checkpoint differs from the one in the table, the model card reports both numbers, so trust the card over the table.

## Run your own model

Freeze your checkpoint. Create predictions without seeing the labels. Commit the prediction hash. Score each track. Report the paired uncertainty interval against our published predictions.

The suite opens one track at a time, starting with `crop`. The others follow as their builders and prediction files are published. Check the repository's availability table for what can be run today rather than assuming everything in this post is ready.

If another team beats us under this protocol, we want to know. The alternative is learning about a problem from a customer after a taxonomy mismatch has quietly removed fourteen points from a production route. We have had enough versions of that conversation to know that evaluation is not overhead. It is how we keep the unknowns visible.

*Next: what a benchmark win does not tell you - independent human gold, the neckline taxonomy mismatch, and why our production system is calibrated on none of the public data.*
