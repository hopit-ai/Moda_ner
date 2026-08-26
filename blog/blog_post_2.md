# The numbers, the code, and what we are releasing

**MODA_NER, Part 2: results and release**

*Series: Fashion attribute extraction from images*
*Previous: [Why we built the test system before the model](blog_post_1.md)*

The previous post argued that a test system should come before the models it judges. This one puts our own models through it, publishes the code and the weights, and explains the design decisions that produced them.

Everything here arrives with the means to check it: the scorers, the frozen splits, and our own prediction files with their hashes. If you disagree with a number, you can recompute it.

## What the model is asked to do

The `crop` track uses fifteen fields: three category levels (`master_category`, `category`, `sub_category`), shape (`silhouette`, `hemline`, `waist_type`), sleeves (`sleeve_length`, `sleeve_shape`), neck and collar (`neckline`, `collar_presence`, `collar_style`), surface (`material`, `surface_treatment`, `pattern`), and `closure_type`.

Each has a fixed vocabulary published with the track: 21 possible silhouettes, 21 necklines, and so on. A test cannot fairly judge an answer whose allowed values are secret.

`catalog` uses ten fields including colour and fit. `fullbody` uses eighteen with an explicit not-applicable class. `text` identifies thirteen entity types in product copy.

## Results

| Track | What the score measures | MODA | Comparator | 95% range of the difference |
|---|---|---:|---:|---:|
| `crop` | Share of predictions matching exactly, across 15 fields | **0.6300** | 0.6245 | [+0.0014, +0.0097] |
| `catalog` | Score per field, averaged over 10 fields | **0.8292** | 0.6657 | [+0.1595, +0.1676] |
| `fullbody`: overall | Score per field, averaged over 18 fields | **0.6917** | 0.5943 | [+0.0891, +0.1053] |
| `fullbody`: knows when not to answer | How reliably it says an attribute is not there | **0.6637** | 0.6088 | [+0.0433, +0.0657] |
| `fullbody`: when visible | The 18-field average, on cases where the attribute is present | **0.5785** | 0.4969 | [+0.0723, +0.0905] |

Do not compare across rows. The tracks use different images, different fields and different scoring. `catalog` at 0.8292 is not "better" than `crop` at 0.6300: ten catalogue fields on clean studio photos is an easier problem than fifteen on a cropped garment. Each row says one thing only, which is how the model did against that track's comparator.

The comparator on `catalog` and `fullbody` is FashionCLIP 2.0 with matched supervised heads, a genuine external system. The `crop` comparator is not: it holds our architecture, heads and training data constant and changes only the encoder. That row is an encoder experiment, not a win over another company's product. The real third-party baselines on `crop` are zero-shot systems at 0.1805 and 0.1817, and neither was built for this task, which is why we do not make much of the gap.

The narrow claim this supports:

> MODA General is the best of the named open systems evaluated under the frozen public MODA General Attribute Suite, spanning localized garment crops, catalog product images, color and fit, and applicability-aware full-body attributes.

Not world-best, not human-level, not ready for every production case. The `text` track sits outside that sentence, because no other system has yet been evaluated on it under the same conditions.

## The number behind the number

On `crop`, 0.6300 means about 63% of individual attribute predictions match the label exactly. On 49.66% of garments, all fifteen fields are right.

The average hides a very wide spread.

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

Telling a coat from a dress is close to solved. Telling denim from twill is not: `material` sits at 0.4148, on only 146 labelled cases, so it is both the hardest field and the thinnest evidence. Read this table before trusting the headline.

## Architectural decisions

**One sequence became many heads.** The Florence-2 system generated every attribute in a single autoregressive response. That couples fields with nothing to do with each other, lets rare labels lose out to common ones, and gives the model no way to abstain. We replaced it with a frozen vision encoder and separate heads per field.

**Applicability is its own decision.** Each conditional field gets a binary head that answers "does this apply here?" before any value head runs. It is scored separately, which is the `fullbody` row above. A model that invents a neckline for a pair of trousers should not be rewarded for confidence.

**Thresholds are calibrated, not guessed**, on a split that never touches development or test. Multi-label fields such as material use an asymmetric focal loss, with balancing capped so that a common positive cannot swamp a rare one.

**The backbone was tested, not assumed.** We compared a general SigLIP-2 encoder against a fashion-pretrained one under an identical development protocol. SigLIP-2 reached 0.6018 against 0.6163. That settled it for about eleven cents of compute, before anyone had to argue.

**The encoder is ours, and this is worth being precise about.** An earlier ladder of checkpoints put our heads on a frozen third-party fashion encoder. We distilled that system into our own encoder, which is what serves today. The third-party encoder appears twice in this work: once as the distillation teacher during training, and once as the comparator on the `crop` row. It is never the model being served.

## What we are releasing

| Model | Input | Licence |
|---|---|---|
| **MODA_NER(V) Crop** | Cropped garment image | MIT |
| **MODA_NER(V) Catalog** | Catalogue product image | CC BY-NC 4.0 |
| **MODA_NER(V) Full-body** | Full-body photo | CC BY-NC 4.0 |

Two of the four tracks are built on research-only datasets. Weights trained on them are non-commercial, and that restriction binds us as well: those exact weights are not part of our paid product.

We are not holding back a better model. The published `crop` checkpoint is the same one that produced the 0.6300 above.

MODA_NER(T), the text model, is not distributed. Its benchmark is: we publish the track and our score on it, and keep the checkpoint.

Alongside the weights, the repository has the protocol, the frozen splits, the scorers, the uncertainty code and our own prediction files with their hashes, including the runs we lost. We do not redistribute source datasets. You obtain them from their original sources under their own terms, and the builders reconstruct the exact splits from IDs and checksums.

## Run your own model

Freeze a checkpoint. Predict without seeing labels. Commit the prediction hash. Score each track and report the comparison range against our published predictions.

If another team beats us under this protocol, we want to know. It is better to find a weakness in a reproducible test than in a customer's production pipeline.

*Next: what none of this tells you about your own catalogue.*
