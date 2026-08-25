# We couldn't find a benchmark, so we built one

**MODA_NER, part 1: the MODA General Attribute Suite**

*Series: Fashion attribute extraction from images*
*Sibling series: [Building a fashion search engine from scratch](https://github.com/hopit-ai/Moda)*

---

Say you have a photo of a dress and you need structured facts about it. Category, sleeve
length, neckline, color, fit, pattern, material. Not a caption, not a vibe: a row you can
put in a database, because a trend system or a catalog pipeline is going to consume it.

Which model should you use? We wanted a benchmark to answer that question and couldn't find
one. So we spent the summer building it, and this post is about what we built, how we run
it, and one result we'd rather not have gotten but published anyway.

## The result upfront

Three frozen public tracks, our system against the strongest open baseline we could find on
each. MODA wins every track, and every paired 95% confidence interval clears zero:

| Track / primary metric | MODA | Strongest comparator | Paired 95% CI |
|---|---:|---:|---:|
| Fashionpedia attribute micro-F1 | **0.6300** | FashionSigLIP 0.6245 | [+0.0014, +0.0097] |
| Shopping100k field-macro set F1 | **0.8292** | FashionCLIP 0.6657 | [+0.1595, +0.1676] |
| DFMM Tier-1 macro-F1 | **0.6917** | FashionCLIP 0.5943 | [+0.0891, +0.1053] |
| DFMM Tier-2 N/A-F1 | **0.6637** | FashionCLIP 0.6088 | [+0.0433, +0.0657] |
| DFMM Tier-3 visible macro-F1 | **0.5785** | FashionCLIP 0.4969 | [+0.0723, +0.0905] |

Here is the exact claim we allow ourselves:

> MODA General is the best of the named open systems evaluated under the frozen public MODA
> General Attribute Suite, spanning localized garment crops, catalog product images, color
> and fit, and applicability-aware full-body attributes.

That's it. Not world-best, not universal SOTA, not a public-leaderboard result, not
human-gold quality, not production readiness. The rest of this post is why the sentence is
worded that carefully.

## Why there was no benchmark

It isn't that fashion datasets don't exist. It's that each one covers a different slice
with a taxonomy that disagrees with the others.

Fashionpedia has expert-annotated attributes localized to garment masks, which is
wonderful, and it has no color and no fit, which are the two fields a merchandiser asks
about first. Shopping100k has color and fit as clean structured fields, on catalog images
only, with no concept of a field that doesn't apply to a garment. DeepFashion-MultiModal
has full-body photos, per-region labels, and an explicit N/A class, inside a closed
vocabulary. VLM benchmarks measure captioning and VQA, which is a different job.

The tempting move is to map everything into one label space and publish one number. We
tried sketching that mapping and stopped, because every mapping choice was quietly a
modeling decision, and the single number would have hidden which input contract a model
actually fails on. A model that's great on flat product shots and lost on street photos
would score "fine".

So the suite keeps three tracks, three test sets, three metrics, and refuses to average
them. A system is as good as its worst required track. That's the whole philosophy.

## The harness, start to finish

Everything above is just policy until you see what actually happens when a model gets
evaluated. Here's the walk.

**Step 1: build the manifest.** Each track has a builder that pins everything: dataset
version, checksums, record IDs, group IDs, split roles, the output vocabulary, alias maps,
missing-label rules, the metric, the bootstrap method, the seed. The output is a frozen
manifest, and it exists before any model runs.

```
python -m suite.dfmm18.build_manifest --source data/dfmm
# manifest: 5,000 test images, 1,751 product groups, 18 fields, sha256=...
```

We never redistribute the restricted datasets. You download them from their official
sources under their own terms; the builder reconstructs our exact splits from IDs and
checksums and refuses to continue if your copy doesn't match.

**Step 2: split at the unit where leakage actually happens.** This is the detail we'd fight
for. One Fashionpedia image yields several garment crops. One DFMM product appears in
several photos. If you split crops randomly, the same source image lands in train and test
and your score is quietly inflated. If you bootstrap crops independently, your confidence
interval is quietly too narrow, because crops from one image succeed and fail together.

So every track splits *and* resamples at its natural unit: source image for Fashionpedia,
image for Shopping100k, product group for DFMM. The DFMM test is the strictest thing we
have: 5,000 fresh images across 1,751 product groups, zero record overlap and zero
product-group overlap with every experiment we had ever run before it.

And the roles never blur. Training changes weights, calibration sets thresholds,
development picks checkpoints, and the frozen test is read once, at the end, for the final
comparison.

**Step 3: predict blind, then commit.** Your inference code gets images and a schema. It
never gets labels. When you're done, you hash the prediction file:

```
python -m suite.dfmm18.commit --predictions preds.jsonl
# committed: preds.jsonl  sha256=9f3a...  rows=5000
```

That hash goes in the record before scoring. After this point you can't swap the file, and
neither can we. Every prediction file in our results directory, including the ones from the
models we beat, sits next to its hash.

**Step 4: the scorer opens the labels, and it fails closed.** Missing row? It stops.
Duplicate ID? Stops. An ID that isn't in the manifest, a field outside the frozen
vocabulary, a hallucinated value that alias maps can't resolve? Reported, separately, where
everyone can see it. The one thing the scorer will never do is silently drop the rows a
model got wrong.

One scoring rule matters more than the rest: an unobservable field is not a negative
example. If the photo crops out the waist, "waist: N/A" isn't a mistake. Fashionpedia
scores applicability as its own judgment for conditional fields, Shopping100k excludes
unlabeled cells, and DFMM scores N/A as a real class with its own F1 (that's Tier 2 in the
table). A model that hallucinates every attribute it knows gets no credit for the habit.

**Step 5: paired bootstrap, then a gate with no averaging.** Scores come with 10,000
paired bootstrap resamples, clustered at the track's leakage unit, so the confidence
intervals in the table mean what they say. And promotion is conjunctive: the CI lower
bound has to clear zero on every required track. There's no weighted average for a bad
track to hide inside. If we'd failed DFMM Tier 2, we'd have failed, full stop.

That's the harness. It's not exotic. It's just every place we found where a benchmark can
lie to its owner, closed one at a time.

## The loss we kept

The first time we ran the full-body track, we lost it.

FashionCLIP with matched supervised heads beat our standalone model on the earlier
protocol, and a MODA–FashionCLIP hybrid beat both:

| System (earlier matched protocol) | Tier 1 | Tier 2 | Tier 3 |
|---|---:|---:|---:|
| MODA distilled, standalone | 0.5398 | 0.5260 | 0.4509 |
| FashionCLIP + linear heads | 0.5985 | 0.6237 | 0.5008 |
| MODA–FashionCLIP hybrid | **0.6492** | **0.6425** | **0.5353** |

We kept the result. We picked the next standalone candidate using development data only,
built the fresh product-group-disjoint shadow so the rematch couldn't be contaminated by
anything we'd learned, and evaluated once. That rematch is the 0.6917 in the headline
table. The earlier loss stays in the record because it's the reason you can believe the
win.

The same discipline ended a commercial-model comparison early. We started an image-only
Gemini 3.5 Flash evaluation with a 100-row checkpoint gate across all three tracks. It
scored well under MODA at the checkpoint — 0.3624 vs 0.6667 on the Fashionpedia rows,
0.5830 vs 0.8057 on Shopping100k, 0.3522 vs 0.4482 on DFMM Tier 1 — so the gate failed and
the remaining 900 paid calls were never made. Gemini did flash a promising color-only
signal on 27 eligible rows. On a 400-row balanced follow-up it didn't replicate. Cost
gates are how a small team affords honest baselines.

## What the numbers don't say

Some fields are still hard, and we'd rather you read it here than find it in your
pipeline. Material sits at 0.4148 value-F1 on Fashionpedia. Fabric is 0.6693 and fit
0.6708 on Shopping100k. Collar style, neckline, and rare applicability values all lag. A
conjunctive win means no track failed. It does not mean every attribute is
production-grade.

## Run your own model

The repo has the manifest builders, ID lists, vocabularies, scorers, bootstrap code, and
our frozen prediction files with their hashes — ours and the baselines'. Freeze your
checkpoint, predict blind on the same IDs, commit your hash, score each track, and report
the paired CI against our shipped predictions.

If your lower bound clears zero on every track, we genuinely want to hear from you.

*Next in the series: what a benchmark win doesn't tell you — independent human gold, a
taxonomy mismatch that cost 14 points, and why our production system is calibrated on none
of the data above.*
