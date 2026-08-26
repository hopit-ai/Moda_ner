# Why we built the test system before the model

**MODA_NER, Part 1: the MODA General Attribute Suite**

*Series: Fashion attribute extraction from images*  
*Sibling series: [Building a fashion search engine from scratch](https://github.com/hopit-ai/Moda)*

Fashion images contain useful product data: category, colour, sleeve length, neckline, fit, pattern, and material. That data supports catalogues, search, recommendations, and trend analysis.

Getting a model to produce plausible answers is not the hard part. The hard part is knowing whether those answers are reliable.

Will the model work on both catalogue and full-body photos? Does it know when an attribute is not visible? Did a change genuinely improve the model? These are the questions we need to answer before trusting any result.

That is why this series begins with the test system rather than a model announcement.

## One benchmark was not enough

We looked for an existing benchmark that covered the whole problem. We did not find one.

Fashionpedia has detailed expert labels, but no colour or fit. Shopping100k includes colour and fit, but only catalogue images and no "not applicable" label. DeepFashion-MultiModal (DFMM) includes full-body photos and supports N/A, but uses a different vocabulary. General vision-language benchmarks focus on captions and questions, not structured product data.

Forcing these datasets into one label system and reporting one score would hide important trade-offs. A model could work well on clean product images and poorly on full-body photos, while still looking good in an average.

So we built four separate tracks.

| Track | Input | What it tests |
|---|---|---|
| `crop` | Cropped garment image | Fine-grained garment attributes |
| `catalog` | Catalogue product image | Colour, fit, and retail-style images |
| `fullbody` | Full-body photo | Attributes that may not apply or be visible |
| `text` | Product title or description | Attributes in product copy |

Each track answers a different question. A good result on one does not cover a weakness on another.

The disagreement between sources goes deeper than image type. Colour and fit, the two attributes merchandisers ask about first, appear in only two of the four tracks. Material appears in two. Exactly one field is present in all four: neckline.

That field is also the one that has cost us the most. We evaluated against an independent labelled set of 1,110 images and scored 0.7169. Our production route scored 0.5764 on the same images, about 14 points lower. Nothing had regressed. The two datasets divide neckline categories differently, and our mapping between the label systems lost a real distinction.

This is the trap in fashion attribute data. Two sources both have a field called neckline, both appear to mean the same thing, and nothing warns you that they do not. A single averaged score cannot show this. Four tracks with different label systems at least make the disagreement visible.

## What the model is asked to predict

The `crop` track covers fifteen fields from a published set of allowed values.

| Group | Fields | Examples |
|---|---|---|
| Category | `master_category`, `category`, `sub_category` | outerwear, coat, trench |
| Shape | `silhouette`, `hemline`, `waist_type` | a-line, asymmetrical, high-rise |
| Sleeves | `sleeve_length`, `sleeve_shape` | long, three-quarter, bishop |
| Neck | `neckline`, `collar_presence`, `collar_style` | v-neck, cowl, halter |
| Surface | `material`, `surface_treatment`, `pattern` | denim, distressed, floral |
| Construction | `closure_type` | zip-up, button, wrap |

`catalog` uses ten fields, including colour and fit. `fullbody` uses eighteen, with an explicit N/A option. `text` identifies thirteen entity types in product copy.

The vocabulary ships with the benchmark. A test cannot fairly judge an answer if it does not say what answers are valid.

## What happened before we had it

We did not start with this. We started with a model.

Our first serious system used Florence-2 to generate every attribute as one JSON response. We trained it again and watched recall: 0.6013 became 0.5885. On that number, nothing much had happened.

Precision had fallen from 0.5661 to 0.3158.

The model had begun guessing far more often. It still did well on easy fields, so the blended score stayed respectable while the system underneath got worse. Only a field-by-field view showed it. We came close to shipping that model.

A second example is smaller but makes the same point. A commercial model appeared to beat us on colour, 0.730 against our 0.630. That came from 27 eligible rows. It did not pass our broader 100-row checkpoint, so we stopped before paying for the remaining 900 calls. Tested later on 400 balanced rows, the advantage was gone. Twenty-seven rows are a reason to investigate, not a finding.

The harness has also made us publish results we would rather not have. In an early full-body evaluation, an external system with matched supervised heads beat our standalone model, and a hybrid of the two beat both. We kept that result, selected the next candidate on development data only, built a fresh test split, and evaluated once.

## How the harness works

Before a model runs, we create a manifest that fixes the dataset version, exact examples, allowed values, scoring rules, and random seed. This prevents the test from changing during an experiment.

We also keep related examples together. One source image can create several garment crops, and one product can appear in several photos. If related examples appear in both training and test data, the model can look better than it really is.

The model receives images and the required schema, not the labels. When prediction is complete, the output file is hashed before scoring. That creates a record of exactly what was evaluated.

The scorer checks every row. It reports missing rows, duplicate IDs, and unsupported values rather than quietly dropping them. It also distinguishes "wrong" from "not visible." If a garment's waist is outside the photo, `waist: N/A` is not an error.

Finally, we compare models on the same examples and calculate an uncertainty range around the difference. A model must improve on every required track. A strong result on one track cannot hide a weak result on another.

## Why this matters

We started with models before we had this test system. That is common, but it makes it easy to mistake a convenient story for a real improvement.

The suite was built to prevent that. It fixes the test before inference begins, prevents train/test leakage, records predictions before labels are opened, and scores every expected output.

The harness is not the product model. It is the way we make claims about any model, and make those claims checkable.

With the next post we publish the protocol, frozen splits, scorers, uncertainty code, prediction files, and hashes. We will not redistribute third-party datasets; users obtain them under the source dataset's terms and our builders recreate the frozen splits.

Next: the results, the released models, and what the numbers actually mean.
