# Why we built the test system before the model

**MODA_NER, Part 1: the MODA General Attribute Suite**

*Series: Fashion attribute extraction from images*  
*Sibling series: [Building a fashion search engine from scratch](https://github.com/hopit-ai/Moda)*

A fashion image contains useful product data: category, colour, sleeve length, neckline, fit, pattern, and material. Extracting that data well makes catalogues easier to build and improves search, recommendations, and trend analysis.

The difficult part is not getting a model to produce a plausible answer. The difficult part is knowing whether the answer is reliable.

Can the model handle product shots and full-body photos? Does it know when an attribute is not visible? Did a new training run really help, or did the score just move by chance?

Those questions are why we built the evaluation harness before building further models.

## One benchmark was not enough

We looked for a benchmark that measured the whole task. We could not find one.

Existing fashion datasets each cover part of the problem. Fashionpedia has strong expert labels but lacks colour and fit. Shopping100k includes colour and fit, but only on catalogue images and without a "not applicable" label. DeepFashion-MultiModal (DFMM) contains full-body images and supports N/A, but uses its own label set. General vision-language benchmarks focus on captions and questions, not structured product data.

We could have merged these datasets into one label system and reported a single score. That would have been misleading. A model that works on clean catalogue images but fails on full-body photos can still look good in an average.

Instead, we built four separate tracks and keep their results separate.

| Track | Input | Frozen test set | Why it exists |
|---|---|---:|---|
| `crop` | A cropped garment image | 4,688 crops from 1,158 images | Detailed garment attributes |
| `catalog` | A catalogue product image | 9,995 images; 61,384 labelled fields | Colour, fit, and retail-style images |
| `fullbody` | A full-body photo | 5,000 images across 1,751 product groups | Attributes that may not apply or be visible |
| `text` | Product title or description | 1,071 rows | Attributes in product copy |

Each track answers a different question. None should be used as a shortcut for the others.

## What the model is actually asked to do

Scores are only useful when the task is clear. In the `crop` track, the model predicts fifteen fields from a published set of allowed values.

| Group | Fields | Examples |
|---|---|---|
| Category | `master_category`, `category`, `sub_category` | outerwear, coat, trench |
| Shape | `silhouette`, `hemline`, `waist_type` | a-line, asymmetrical, high-rise |
| Sleeves | `sleeve_length`, `sleeve_shape` | long, three-quarter, bishop |
| Neck | `neckline`, `collar_presence`, `collar_style` | v-neck, cowl, halter |
| Surface | `material`, `surface_treatment`, `pattern` | denim, distressed, floral |
| Construction | `closure_type` | zip-up, button, wrap |

The full vocabulary ships with the benchmark. A test cannot fairly grade an answer if it does not say what answers are valid.

`catalog` uses ten fields, including colour and fit. `fullbody` uses eighteen fields, including N/A. `text` identifies thirteen kinds of information in product copy.

## The result that changed our approach

We did not start with a strong evaluation process. We started with a model.

Our first serious system used Florence-2 to produce a JSON object containing every attribute. It looked promising, so we trained it again. Recall barely changed: 0.6013 became 0.5885. An overall score could have made this look roughly unchanged.

But precision dropped from 0.5661 to 0.3158.

The model had started guessing more often. It still performed well on easy fields, which hid the problem in an aggregate score. Looking at each field separately showed a clear regression.

That result led to two changes. First, we replaced one long autoregressive output with separate conditional heads and an explicit "does this field apply?" decision. Second, we paused model work and built a better way to measure it.

The suite grew from that work: Fashionpedia for detailed expert labels, Shopping100k for colour and fit, DFMM for applicability, and a text track for product descriptions.

## How the harness keeps the test honest

The harness follows five rules.

**1. Freeze the test first.** Before a model runs, a manifest records the dataset version, exact examples, allowed values, scoring rules, and random seed.

```
python -m suite.crop.build_manifest --source <your local copy>
# manifest: 4,688 garment crops, 1,158 source images, 15 fields, sha256=...
```

This prevents the test from changing during the experiment.

**2. Keep related examples together.** One source photo can create several crops. One product can have several photos. If related examples appear in both training and test data, the model looks better than it should. We group the splits by source image, image, or product group depending on the track.

**3. Predict before seeing labels.** The model sees images and the schema, not the answers. We hash the prediction file before scoring it.

```
python -m suite.crop.commit --predictions preds.jsonl
# committed: preds.jsonl  sha256=9f3a...  rows=4688
```

The hash records exactly what was scored. It prevents predictions being replaced after the result is known.

**4. Score every row.** Missing rows, duplicate IDs, and unsupported values are reported. The scorer does not quietly remove difficult cases. It also separates "wrong" from "not visible." If the waist is outside the frame, predicting `waist: N/A` is not an error.

**5. Do not hide a weak result in an average.** We compare systems on the same examples and calculate an uncertainty range around the difference. A model must show a positive improvement on every required track; a strong result on one track cannot compensate for a weak one on another.

## What the model can do today

On the `crop` track, MODA's main score is attribute micro-F1 of **0.6300**. Put simply: across all fifteen attributes, about 63% of individual predictions exactly match the dataset labels. On **49.66%** of garments, the model gets all fifteen fields right.

Those summary numbers hide a large range in difficulty.

| Field | F1 | Labelled cases |
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

Category recognition is relatively strong. Material, neckline, collar style, and surface treatment are not. If a pipeline depends on those fields, this model is not ready for that use case. We would rather make that clear here than have users discover it in production.

## Results across the image tracks

| Track | What it measures | MODA | Comparator | Difference range (95%) | Can you check it today? |
|---|---|---:|---:|---:|---|
| `crop` | Exact attribute matches across 15 fields | **0.6300** | 0.6245 | [+0.0014, +0.0097] | **Yes** |
| `catalog` | Average score across 10 fields | **0.8292** | 0.6657 | [+0.1595, +0.1676] | Not yet |
| `fullbody`: overall | Average score across 18 fields | **0.6917** | 0.5943 | [+0.0891, +0.1053] | Not yet |
| `fullbody`: N/A | Correctly identifying inapplicable attributes | **0.6637** | 0.6088 | [+0.0433, +0.0657] | Not yet |
| `fullbody`: visible | Score where the attribute is visible | **0.5785** | 0.4969 | [+0.0723, +0.0905] | Not yet |

That last column matters more than it looks. Today only `crop` is fully open: its weights, its scorer and our own prediction files are published, so you can recompute 0.6300 yourself and tell us if we are wrong. The other rows are, for now, numbers you are taking on trust. The tracks open one at a time as their builders and prediction files are published, and we would rather show you which of our own claims are currently auditable than let the difference pass unmentioned.

Do not read across the rows either. `catalog` at 0.8292 is not "better" than `crop` at 0.6300: different metrics, different fields, different images. Ten catalogue fields on clean studio photos is an easier problem than fifteen on a cropped garment. Each row compares one system against one comparator on one track, and nothing more.

Do not compare numbers across rows. The tracks use different images, fields, and metrics. A 0.8292 on clean catalogue images is not directly comparable with 0.6300 on detailed garment crops. Each row is only a comparison with its own baseline.

For `catalog` and `fullbody`, the comparator is FashionCLIP 2.0 with matched supervised heads. The `crop` comparator needs more care: it is our own architecture and heads using a FashionSigLIP encoder instead of ours. It is an encoder comparison, not a claim that we beat another vendor's product.

Against that `crop` variant, we have two wins and two ties. We improve micro-F1 and master-category accuracy; field-macro F1 and category accuracy are statistically tied. We report both micro and macro scores because micro-F1 gives more weight to common fields, while macro-F1 gives every field equal weight.

The `text` track is not included in the comparative claim. We added it after that claim was frozen and have not yet evaluated another system on it under the same conditions.

The evidence supports this narrower statement:

> MODA General is the best of the named open systems evaluated under the frozen public MODA General Attribute Suite, spanning localized garment crops, catalog product images, color and fit, and applicability-aware full-body attributes.

It does not mean world-best, human-level, or production-ready for every attribute.

## What the harness has already saved us from

The harness made us publish a loss. On an early full-body evaluation, FashionCLIP with matched heads beat our standalone model, and a hybrid model beat both.

| System (earlier matched protocol) | Tier 1 | Tier 2 | Tier 3 |
|---|---:|---:|---:|
| MODA distilled, standalone | 0.5398 | 0.5260 | 0.4509 |
| FashionCLIP + linear heads | 0.5985 | 0.6237 | 0.5008 |
| MODA-FashionCLIP hybrid | **0.6492** | **0.6425** | **0.5353** |

We kept that result. We then chose a new candidate using development data, built a fresh test split with no overlapping product groups, and ran the final test once.

It also stopped us from overreacting to a small sample. A commercial model initially looked better on colour: 0.730 versus our 0.630. But that was based on 27 eligible rows. It did not pass our broader 100-row checkpoint, so we did not pay for another 900 calls. Later, on 400 balanced rows, the apparent advantage disappeared.

Twenty-seven rows are not a conclusion. They are a reason to look closer.

## The limitation we could not see internally

Our biggest recent issue was found outside the harness.

On an independent, human-labelled set of 1,110 images, our model scored 0.7169 against FashionCLIP's 0.6605. But the production composite route scored 0.5764 on the same images, roughly 14 points lower.

This was not a model regression. The external data divided neckline types differently, and our mapping between the two label systems removed a meaningful distinction. Our internal evaluation could not catch the issue because its labels were consistent with themselves.

That is the limit of an internal benchmark. It can measure a well-defined task carefully. It cannot prove that the task definition matches every retailer or external dataset.

## What is open, and what comes next

Our approach is: open the ruler; sell what the ruler measures.

We publish the protocol, splits, scorers, uncertainty code, prediction files, and hashes, including losing runs. We do not redistribute source datasets or their labels. Users download them from their original sources, and the builders recreate the frozen splits.

What remains closed is customer-specific work: decision thresholds, model-routing logic, and mappings to a retailer's taxonomy. Two of the four tracks are based on research-only data, so weights trained on them are non-commercial, and that restriction binds us too: those exact weights are not part of our paid product.

One thing we are not doing is holding back a better model. The `crop` weights published today are the same checkpoint that produced the 0.6300 above, not a weakened version of it. Download it and you should reproduce our number.

Next, we will add independent human labels on a public track, run more modern open models at full test size, and pursue a holdout built and administered by someone else. That last step is the strongest possible check on a benchmark we created ourselves.

## Run your own model

Freeze a checkpoint. Generate predictions without seeing labels. Commit the prediction hash. Score each available track and report the comparison range against our published predictions.

The suite opens one track at a time, starting with `crop`; check the repository's availability table for the current status. The published `crop` checkpoint is MIT-licensed, commercially usable, and is the checkpoint that produced the 0.6300 result above. The `catalog` and `fullbody` weights are non-commercial.

If another team beats us under this protocol, we want to know. Better to find a weakness in a reproducible test than through a customer's production pipeline.

*Next: what a benchmark win does not tell you - independent human gold, the neckline taxonomy mismatch, and why our production system is calibrated on none of the public data.*
