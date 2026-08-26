# Why we built the test system before the model

**MODA_NER, Part 1: the MODA General Attribute Suite**

*Series: Fashion attribute extraction from images*
*Sibling series: [Building a fashion search engine from scratch](https://github.com/hopit-ai/Moda)*

Fashion images contain data retailers need: category, colour, sleeve length, neckline, fit, pattern, material. Extracting those attributes powers catalogues, search, recommendations and trend analysis.

The hard part is not making a model give a believable answer. The hard part is knowing whether that answer is reliable.

Does the model work on catalogue shots and street photos? Does it know when an attribute is not visible? Did a training change actually improve anything, or did a score just move?

This post is about the test system we built to answer those questions, and why we built it before building more models. The suite, the code and the model weights publish with the next post in this series. This one is about the reasoning.

## What happens when you skip this

We did not begin with a good evaluation process. We began with a model.

Our first serious system used Florence-2 to generate every attribute as one JSON response. We trained it again, and watched recall: 0.6013 became 0.5885. On that number alone, nothing much had happened.

Precision had fallen from 0.5661 to 0.3158.

The model had started guessing far more often. It still did well on easy fields, so the blended score stayed respectable while the system underneath got worse. Only a field-by-field view showed the regression. We came close to shipping it.

Two things followed. We changed the model design, because asking one autoregressive model to emit every field in a single sequence links attributes that should be independent and leaves no clean way to say "this does not apply." And we stopped building models for a while.

That is the honest answer to why the test system came first. Not discipline. We had been about to ship something broken and had no way to see it.

## Why one benchmark was not enough

We looked for an existing benchmark. There is no shortage of fashion datasets, but they disagree with each other in ways that matter.

Fashionpedia has detailed expert labels on garment regions, and no colour or fit. Shopping100k has colour and fit, on catalogue images only, with no way to express that a field does not apply. DeepFashion-MultiModal has full-body photos and an explicit "not applicable" class, inside its own fixed vocabulary. General vision-language benchmarks test captioning and question answering, which is a different job.

The disagreement is easier to see as a table. These are the four tracks we ended up building, and what each one can and cannot express:

| Track | Fields | Colour | Fit | "Not applicable" |
|---|---:|---|---|---|
| `crop` cropped garment | 15 | no | no | partial |
| `catalog` product image | 10 | yes | yes | no |
| `fullbody` full-body photo | 18 | no | no | yes, explicit |
| `text` title or description | 13 | yes | yes | not applicable |

Colour and fit, the two attributes a merchandiser asks about first, exist in only two of the four. Material exists in two.

And exactly one field appears in all four: **neckline**. It is also the field that has cost us the most.

## The same word, different meanings

We evaluated our system against an independent, human-labelled set of 1,110 images that nobody on our team had touched. It scored 0.7169 against a FashionCLIP baseline at 0.6605. A clear result.

Our production route scored 0.5764 on those same images. About 14 points lower.

The model had not regressed. The two datasets divide neckline categories differently, and our mapping between the label systems quietly collapsed a distinction that mattered. Every internal test we had was blind to it, because internally the taxonomy always agreed with itself.

This is the trap in fashion attribute data. Two datasets both have a field called "neckline". Both look like they mean the same thing. They do not, and nothing warns you.

A single averaged benchmark score cannot show you this. Four separate tracks with different label systems at least make the disagreement visible, which is why we report them separately and never average them. A good result on one track does not cancel a bad one on another.

## The five rules

The harness is not exotic infrastructure. It is five rules, each closing a way a benchmark can flatter the person who built it.

1. **Freeze the test before the model runs.** A manifest records the data version, the exact examples, the allowed answers, the scoring rules and the random seed. The target cannot move while we measure.
2. **Keep related images together.** One source photo yields several crops; one product appears in several photos. If related examples land on both sides of a split, the model looks better than it is. The same applies to uncertainty: related images succeed and fail together, so treating them as independent makes our confidence look stronger than it is.
3. **Predict before seeing the answers.** Every prediction file is hashed before scoring. Neither we nor anyone else can substitute a better one afterwards.
4. **Score every row.** Missing, duplicated and unsupported predictions are reported, not silently dropped. The scorer also separates "wrong" from "not visible": if a photo cuts off the waist, saying so is not an error. Without that rule, a model that invents an attribute for every garment looks better than one that knows when to abstain.
5. **Do not hide a weak track inside an average.** A model has to improve on every required track, not on the mean.

## What it caught

Most of what a test system does is quiet. It declines to let you believe convenient things.

It made us publish a loss. In an early full-body evaluation, FashionCLIP with matched supervised heads beat our standalone system, and a hybrid of the two beat both. We kept that in the record, chose the next candidate using development data only, built a fresh test split with no overlapping products, and evaluated once.

It stopped us buying a result we wanted. A commercial model looked better on colour, 0.730 against our 0.630. That came from 27 rows. It failed our broader 100-row checkpoint, so we did not pay for the remaining 900 calls. Tested later on 400 balanced rows, the advantage disappeared.

Twenty-seven rows are a reason to investigate. They are not a finding.

## What comes next

The next post has the numbers: how the models score on each track, what we release, and the architectural decisions that got us there. The code, the scorers, our own prediction files and the model weights publish alongside it, so the numbers arrive with the means to check them rather than ahead of it.

*Next: the results, the code, and what we are releasing.*
