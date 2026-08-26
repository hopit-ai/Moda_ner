# Why we tested the measurement system before the model

**MODA_NER, Part 1: the MODA General Attribute Suite**

*Series: Fashion attribute extraction from images*
*Sibling series: [Building a fashion search engine from scratch](https://github.com/hopit-ai/Moda)*

When a computer looks at a fashion image, we want it to return useful facts: what the garment is, its colour, sleeve length, neckline, fit, pattern, and material.

That sounds simple. It is not.

Those facts feed product catalogues, search, recommendations, and trend analysis. So it is not enough for a model to give answers that look convincing in a demo. We need to know how often it is right, where it fails, and whether a new model is genuinely better than the last one.

This is why we are starting a series about fashion attribute extraction with the test system, not the model.

## The problem with a single benchmark score

We looked for an existing benchmark that could answer those questions. We did not find one.

Fashion datasets differ in the images they contain, the attributes they label, and the words they use for those attributes. Their label systems do not line up. Fashionpedia has detailed expert labels but no colour or fit. Shopping100k has colour and fit, but uses catalogue photos and cannot express that a field does not apply. DeepFashion-MultiModal has full-body photos and an explicit "not applicable" label, inside its own fixed vocabulary. General vision-language benchmarks test captioning or visual question answering, not structured product data.

We could have forced these into one shared label system and published a single impressive number. That would hide the trade-offs. A model that does well on clean product images and poorly on street photos would still look good after averaging.

So we built four separate tests and report them separately, because a good overall score should not hide a serious weakness.

| Track | What it tests | Test set | Related examples grouped by |
|---|---|---:|---|
| `crop` | Attributes from a garment crop | 4,688 crops from 1,158 images | Source image |
| `catalog` | Attributes from catalogue images | 9,995 images; 61,384 fields | Image |
| `fullbody` | Attributes from full-body photos | 5,000 images; 1,751 product groups | Product group |
| `text` | Attributes from titles and descriptions | 1,071 rows | Row |

## What we actually ask the model for

An F-score means nothing until you know what is being scored, so here is the `crop` schema in full. Fifteen fields, every one of them a closed vocabulary published with the track.

| Group | Fields | Example values |
|---|---|---|
| Category | `master_category`, `category`, `sub_category` | outerwear, coat, trench |
| Shape | `silhouette`, `hemline`, `waist_type` | a-line, asymmetrical, high-rise (21 silhouette values) |
| Sleeves | `sleeve_length`, `sleeve_shape` | long, three-quarter, bishop, balloon |
| Neck | `neckline`, `collar_presence`, `collar_style` | v-neck, cowl, halter, notched (21 neckline values) |
| Surface | `material`, `surface_treatment`, `pattern` | denim, distressed, floral |
| Construction | `closure_type` | zip-up, button, wrap |

`catalog` uses ten fields including colour and fit. `fullbody` uses eighteen with an explicit N/A option. `text` identifies thirteen entity types in product copy.

A benchmark cannot fairly judge a prediction if its allowed answers are secret, so all of that ships with the suite.

## What our numbers mean in practice

Our headline `crop` figure is attribute micro-F1 **0.6300**. In plain terms: pool every attribute decision across all fifteen fields, and about 63% of them match the label exactly. A more intuitive figure sits alongside it — on **49.66%** of garments we get *every one* of the fifteen fields right.

The average hides an enormous spread, so here is every field:

| Field | F1 | Gold cases |
|---|---:|---:|
| `master_category` | 0.9215 | 4,688 |
| `category` | 0.8825 | 4,688 |
| `pattern` | 0.8356 | 1,994 |
| `sleeve_length` | 0.8073 | 912 |
| `closure_type` | 0.7545 | 1,080 |
| `collar_presence` | 0.7508 | 294 |
| `sleeve_shape` | 0.6787 | 855 |
| `hemline` | 0.6428 | 1,729 |
| `sub_category` | 0.5966 | 1,594 |
| `silhouette` | 0.5535 | 1,994 |
| `waist_type` | 0.5002 | 1,102 |
| `neckline` | 0.4650 | 1,126 |
| `collar_style` | 0.4566 | 287 |
| `surface_treatment` | 0.4398 | 880 |
| `material` | 0.4148 | 146 |

Read that top to bottom before trusting any single number. Telling a coat from a dress is close to solved. Telling denim from twill is not: `material` sits at 0.4148 on only 146 labelled cases, which is both the hardest field and the thinnest evidence. If your pipeline depends on necklines or surface treatment, this model is not ready for you yet, and we would rather you learn that here than in production.

## Why this mattered to us

We did not start with a good evaluation process. Like many teams, we started with models.

Our first serious system used Florence-2 to produce a JSON object containing every attribute. It looked promising enough to train again. Recall barely moved, from 0.6013 to 0.5885 — on a blended number, close enough to ship.

But precision dropped from 0.5661 to 0.3158.

The updated model had started guessing far more often. It still did well enough on easy fields that an overall score hid the damage; a field-by-field view showed a clear regression.

That taught us two things. First, asking one autoregressive model to generate every field in a single sequence was a poor fit: it linked attributes that should be independent, made rare labels easier to miss, and gave the model no clean way to say "this does not apply." We moved to separate conditional heads with an explicit applicability decision. Second, we stopped improving the model for a while and improved how we measured it.

The tracks then grew one at a time. Fashionpedia first, for its expert garment labels. It could not answer colour or fit, so we added Shopping100k. Neither captures whether an attribute applies at all, so we added DFMM. Text covers a related but distinct task. Each track exists because an earlier one left an important question unanswered.

## How a model is tested

**Lock the test first.** Before any evaluation, a manifest records the dataset version, checksums, records, groups, allowed answers, rules, metric, and seed. Unglamorous, but it stops the target moving while we measure it.

```
python -m suite.crop.build_manifest --source <your local copy>
# manifest: 4,688 garment crops, 1,158 source images, 15 fields, sha256=...
```

**Avoid testing on near-duplicates.** One source image can yield several crops; one product can appear in several photos. If related examples land on both sides of a split, the model looks better than it is. We split at the right level per dataset — source image, image, product group — and estimate uncertainty the same way, because related images tend to succeed or fail together. Training chooses weights, calibration chooses thresholds, development selects a checkpoint, and the frozen test is saved for one final run.

**Predict before seeing the answers.** The model gets images and the schema, never the labels. We hash the prediction file before scoring it, so no better file can be swapped in afterwards.

```
python -m suite.crop.commit --predictions preds.jsonl
# committed: preds.jsonl  sha256=9f3a...  rows=4688
```

**Score every row, including the awkward ones.** The scorer stops or reports an error on a missing, duplicated, unexpected, or unsupported row rather than quietly dropping it. It also separates "wrong" from "not visible": if a photo cuts off the waist, predicting `waist: N/A` is not an error. Without that rule, a model could invent attributes for every garment and look better for it.

**Require improvement across the important tests.** We compare with 10,000 paired bootstrap resamples on the same examples. To promote a model, the lower end of its improvement range must clear zero on every required track. No average lets a bad result hide behind good ones.

## What the suite has caught

Most of the value is quiet: it stops us believing things that are convenient but poorly supported.

It made us publish a loss. In our first full-body evaluation, FashionCLIP with matched supervised heads beat our standalone model, and a hybrid beat both.

| System (earlier matched protocol) | Tier 1 | Tier 2 | Tier 3 |
|---|---:|---:|---:|
| MODA distilled, standalone | 0.5398 | 0.5260 | 0.4509 |
| FashionCLIP + linear heads | 0.5985 | 0.6237 | 0.5008 |
| MODA-FashionCLIP hybrid | **0.6492** | **0.6425** | **0.5353** |

We kept that in the record, used development data to pick the next candidate, built a new split with no overlapping product groups, and ran the final evaluation once.

It also stopped us overreacting to a small sample. A commercial vision-language model initially looked better on colour, 0.730 against our 0.630 — from 27 eligible rows. It did not clear our 100-row checkpoint on the broader task, so we did not buy another 900 calls. Tested later on 400 balanced rows, the advantage disappeared. Twenty-seven rows are not a finding. They are a reason to investigate.

## Our current results

| Track / main metric | MODA | Strongest comparator | Paired 95% interval |
|---|---:|---:|---:|
| `crop` attribute micro-F1 | **0.6300** | Same heads, FashionSigLIP encoder: 0.6245 | [+0.0014, +0.0097] |
| `catalog` field-macro set F1 | **0.8292** | FashionCLIP 2.0 + matched heads: 0.6657 | [+0.1595, +0.1676] |
| `fullbody` Tier-1 macro-F1 | **0.6917** | FashionCLIP 2.0 + matched heads: 0.5943 | [+0.0891, +0.1053] |
| `fullbody` Tier-2 N/A-F1 | **0.6637** | FashionCLIP 2.0 + matched heads: 0.6088 | [+0.0433, +0.0657] |
| `fullbody` Tier-3 visible macro-F1 | **0.5785** | FashionCLIP 2.0 + matched heads: 0.4969 | [+0.0723, +0.0905] |

One important caveat: the `crop` comparison is an encoder experiment, not a win over another vendor. Same architecture, same heads, same training release — only the encoder changed. The genuine third-party baselines there are zero-shot models, Qwen3-VL-8B at 0.1805 and calibrated FashionSigLIP text prototypes at 0.1817, and neither was built for this task, which is why we do not make much of the gap.

The fuller `crop` picture is two wins and two ties against the FashionSigLIP-encoder variant: we win micro-F1 by 0.0056 and master-category accuracy by 0.0085, while field-macro F1 and category accuracy are ties whose ranges include zero. Micro-F1 weights common attributes more heavily; field-macro weights every field equally. Reporting both shows whether gains come only from the easiest fields.

The claim these results support is deliberately narrow:

> MODA General is the best of the named open systems evaluated under the frozen public MODA General Attribute Suite, spanning localized garment crops, catalog product images, color and fit, and applicability-aware full-body attributes.

Not world-best, not universally state of the art, not human-level, not ready for every production case. Exactly what the benchmark measured.

## A limitation our own tests missed

The most important recent problem came from outside the suite.

On an independently human-labelled set of 1,110 images nobody on our team had touched, our model scored 0.7169 against FashionCLIP's 0.6605, clearly positive. But our production composite route scored 0.5764 on the same images, about 14 points lower.

That was not a model regression. The external dataset divided neckline categories differently, and our mapping between the two label systems removed a real distinction. Our internal benchmark could not reveal this, because its labels were internally consistent. Independent labels could.

This is the limit of any internal benchmark: it measures the system it defines, but cannot prove its label system matches every retailer's.

## What we open, and what we do not

Our approach: open the ruler; sell what the ruler measures.

We publish the protocol, splits, scorers, uncertainty code, prediction files, and hashes, including losing runs. A benchmark nobody else can run is just an opinion with a table beside it. We do not ship gold labels on any track — you obtain each dataset from its original source under that source's terms, and our builders rebuild the exact split from IDs and checksums.

What stays closed is what the process produces for a particular customer: field-level thresholds, model-routing logic, and mappings to a retailer's own taxonomy. Those are data-specific, and they are what a customer is paying us to build.

Two of the four tracks use research-only datasets. Weights trained on them are non-commercial and we respect that restriction, which binds us too: those exact weights are not in our paid product.

Next steps, briefly. We have committed 1,993 Fashionpedia image groups for independent human annotation, and until that lands nothing here should be called human-gold. We want at least five more modern open models run on the full test sets rather than at a checkpoint. A fifth track for pattern and neckline is built but will only ship if enough distinct source URLs survive. And the strongest test remains a holdout someone else builds and administers.

## Run your own model

Freeze your checkpoint. Predict without seeing the labels. Commit the hash. Score each track. Report the paired interval against our published predictions.

For `crop`, the published weights are the checkpoint that produced the number above, MIT-licensed and usable commercially — download it, score it, and you should get 0.6300. For `catalog` and `fullbody` the weights are non-commercial. Where any checkpoint differs from the table, the model card reports both numbers; trust the card.

The suite opens one track at a time, starting with `crop`. Check the repository's availability table for what runs today rather than assuming everything here is ready.

If another team beats us under this protocol, we want to know. The alternative is learning about a problem from a customer after a taxonomy mismatch has quietly removed fourteen points from a production route. We have had enough versions of that conversation to know evaluation is not overhead. It is how we keep the unknowns visible.

*Next: what a benchmark win does not tell you — independent human gold, the neckline taxonomy mismatch, and why our production system is calibrated on none of the public data.*
