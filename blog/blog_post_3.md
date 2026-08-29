# What none of this tells you about your catalogue

**MODA_NER, Part 3: using these models in a real fashion business**

*Series: Fashion attribute extraction from images*
*Previous: [The numbers, the code, and what we are releasing](blog_post_2.md)*

The last post published our scores. This one is about the distance between a benchmark number and a working catalogue pipeline, and how to cross it.

Everything below is written for someone who has to decide whether to put one of these models into production this quarter.

## Choose by input, not by score

The four tracks are not a leaderboard. They are a way of asking which model matches the pictures you actually have.

| If your images are | Use | Because |
|---|---|---|
| Cropped garments, one item per image | `crop` route | Fifteen fine-grained garment fields |
| Clean catalogue product shots | `catalog` route | Includes colour and fit, tuned to studio images |
| Street, editorial or full-body | `fullbody` route | Handles attributes that are absent or out of view |
| Titles and descriptions | text route | Extracts attributes from copy rather than pixels |

The highest number in the previous post is irrelevant if it was measured on images unlike yours. A model scoring 0.83 on clean catalogue photos will not hold that up on user-generated images, and nothing in the score warns you.

## Six fields you can automate, nine you cannot

This is the most useful thing we can tell you, and it does not appear in any headline metric.

Taking F1 0.75 as a rough line, meaning about one error in four:

**Safe to populate automatically:** `master_category` (0.9215), `category` (0.8825), `pattern` (0.8356), `sleeve_length` (0.8073), `closure_type` (0.7545), `collar_presence` (0.7508).

**Needs a review step:** `sleeve_shape` (0.6787), `hemline` (0.6428), `sub_category` (0.5966), `silhouette` (0.5535), `waist_type` (0.5002), `neckline` (0.4650), `collar_style` (0.4566), `surface_treatment` (0.4398), `material` (0.4148).

Six of fifteen automate cleanly. Nine do not.

There is a second number underneath that one. Field averages hide how errors land on
individual garments: on **49.66%** of garments every one of the fifteen fields is right. So
roughly half your catalogue comes through clean and half has at least one field to fix, which
is a very different planning input from a 0.63 average.

That is a workflow design problem, not a model problem, and it is solvable. Populate the six, route the nine to a review queue, and put your annotator effort where the errors actually are. A 0.63 headline does not mean 63% of your catalogue is fine. It means part of the job is done and part of it needs a person.

## Your taxonomy is not our taxonomy

We tested against an independent labelled set and scored 0.7169. Our production route scored 0.5764 on the same images, about 14 points lower, because the two label systems divide necklines differently and our mapping between them lost a real distinction.

Nothing had regressed. The vocabularies simply disagreed.

Expect the same. Your `product_type` values, your neckline names, your idea of where "midi" ends are yours, and no public checkpoint knows them. In our experience the mapping work is larger than the modelling work, and it is the part most often left out of a plan.

## Test the stupid baseline first

Our text model scores 0.8723 at extracting exact spans from product copy. That is a real number on a real task.

On a different task, mapping those spans onto a retailer's own `product_type` metadata, it loses to taking the last few words of the product title:

| Route | Exact accuracy | Coverage |
|---|---:|---:|
| Terminal title n-grams | **0.1333** | **100%** |
| Our text model | 0.1100 | 52% |

A 150-million-parameter model beaten by a string operation. Both numbers are low because the task is hard, but the ordering is the point: extraction and taxonomy mapping are different problems, and being good at the first does not make you good at the second.

Before you buy or deploy anything, run the dumb baseline on your data. Sometimes it wins. When it does, that is a result worth having.

## What a benchmark score leaves out

**Our labels are not human gold.** Every number in this series comes from dataset-native labels, not from independent annotators checking our specific claims. We have committed 1,993 image groups for independent annotation and it is not finished. Until it is, nothing here should be called gold.

**Model juries are not gold either.** Where we have used a language model to adjudicate labels, we say so and treat it as diagnostic. We do not convert model judgements into ground truth.

**Serving is its own constraint.** Warm, the routes answer at a p95 of 0.538 to 0.551 seconds across 975 requests. Cold is a different story: a container's first request took 17 to 46 seconds depending on the route, because the crop encoder alone is 794 MB. An earlier latency result failed its own one-second gate and we retracted it. If your pipeline enriches millions of items, throughput, cold-start behaviour and cost will matter as much as accuracy.

## We tried six ways to beat our own model, and lost six times

Over two days we ran every cheap idea we had for improving the full-body route without new
labels. All six lost, and the pattern in how they lost is more useful than any of them
individually.

We transferred the crop model's trunk to the full-body task, on the theory that attribute
supervision on one corpus would help another. It lost. We swapped the linear heads for
gradient-boosted trees and for an MLP; both lost badly to plain logistic regression. We gave
each field its own attention over the image's patch tokens instead of one pooled vector, since
the weakest fields are regional textures like fabric — that lost too. We trained one trunk
jointly on two corpora at once, the only attempt that added new supervision rather than
rearranging what we had; its training loss fell cleanly from 34.0 to 11.4, and it still lost by
0.14 F1, collapsing precisely on the conditional fields the published route handles with a
calibrated threshold.

The most instructive one looked like a win. Concatenating two encoders' features scored +0.0095
over the frozen baseline, which is small but real. Then we ran the control: instead of a second
encoder, we widened the original features to exactly the same dimension with a random
projection. That scored +0.0096. The gain was capacity, not complementarity, and we retracted
it.

Two things follow. The published route is well-built, and nothing cheap improves on it — which
is the strongest version we can offer of "we are not holding back a better model." And the
ceiling here is not architecture. It is data that looks like yours.

## So: off-the-shelf, or fine-tuned?

The answer falls out of the above rather than needing a pitch.

**Take the open weights** if your images match one of the contracts, you mainly need the six strong fields, and you already have a review process for the rest. That is a real amount of value for the price of a download.

**You need work on your own data** if you depend on the weak fields, your taxonomy differs from ours in ways that matter, or you need a guarantee about a specific field rather than an average. That work is mostly taxonomy alignment and calibration, not novel modelling, and it is what MODA_NER Pro is — the same extraction, trained on your products and mapped to your taxonomy rather than the benchmarks'. The models we published are the starting point, not the product.

Either way, the thing to insist on, from us or anyone else, is the same: a frozen test on your own data, predictions committed before labels are opened, and per-field numbers rather than an average. If a vendor will not give you that, the number they are quoting is not about your catalogue.

*This is the last post in the series. The suite, the scorers and the released weights are in the repository, and we would rather hear that we are wrong than not hear it.*
