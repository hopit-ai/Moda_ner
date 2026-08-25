# Attribute extraction for fashion — why the evaluation harness first?

**MODA_NER, part 1: the MODA General Attribute Suite**

*Series: Fashion attribute extraction from images*
*Sibling series: [Building a fashion search engine from scratch](https://github.com/hopit-ai/Moda)*

---

We spent most of a summer building a measuring device instead of a model.

Scroll the commit history for that period and the great majority of it is manifests,
scorers, hash commitments, bootstrap code, cost gates and integrity receipts. The model
that came out the other end is almost a byproduct. That looks like procrastination, and for
a while it felt like it.

This post is the argument for why it was the right order — and, at the end, what it decided
about which parts of the work we're giving away and which parts we're selling.

## The problem we actually had

Say you have a photo of a dress and you need structured facts about it. Category, sleeve
length, neckline, colour, fit, pattern, material. Not a caption, not a vibe: a row you can
put in a database, because a trend system or a catalogue pipeline is going to consume it.

Which model should you use? We went looking for a benchmark that would answer that and
couldn't find one. Not because fashion datasets don't exist — because each covers a
different slice with a taxonomy that disagrees with the others.

Fashionpedia has expert-annotated attributes localised to garment masks, which is
wonderful, and it has no colour and no fit, which are the two fields a merchandiser asks
about first. Shopping100k has colour and fit as clean structured fields, on catalogue
images only, with no concept of a field that doesn't apply to a garment.
DeepFashion-MultiModal has full-body photos, per-region labels and an explicit N/A class,
inside a closed vocabulary. VLM benchmarks measure captioning and VQA, which is a different
job.

The tempting move is to map all of it into one label space and publish one number. We
sketched that mapping and stopped, because every mapping choice was quietly a modelling
decision, and one number would have hidden which input contract a model actually fails on.
A model that's excellent on flat product shots and lost on street photos would score
"fine".

So we built three tracks instead, with three test sets and three metrics, and a rule
against averaging them. A system is as good as its worst required track.

## Three times the measurement caught a lie

The abstract case for evaluation is boring and everyone nods at it. Here are three
occasions where our own numbers were about to walk us off a cliff.

**The model that looked fine and wasn't.** We ran a corrective training pass on a
Florence-2 extraction model. Recall barely moved — 0.6013 to 0.5885, the kind of drift you
shrug at. Underneath, precision had fallen off a table: 0.5661 to 0.3158. A single blended
headline number would have read as "roughly flat, ship it". Per-field scoring showed a
model that had started guessing constantly and getting away with it on aggregate. That
failure is also what killed the architecture — one autoregressive sequence emitting every
field together couples attributes that have nothing to do with each other, and gives the
model no way to say a field doesn't apply. We replaced it with conditional heads and an
explicit applicability decision, which is what we run today.

**The signal that wasn't there.** A commercial VLM looked like it might beat us on colour
specifically: 0.730 against our 0.630. Genuinely interesting, and exactly the sort of
result that gets a slide. It was computed on 27 eligible rows. We ran a 400-row
label-balanced follow-up and the effect vanished — the paired interval came out entirely
negative. Twenty-seven rows is not a finding, it's a rumour, and the only reason we didn't
repeat it out loud is that the protocol made us check.

**The fourteen points hiding in a vocabulary.** Our system scored 0.7169 on an independent,
human-labelled external set — a real win over the FashionCLIP baseline at 0.6605, interval
clear of zero. Then the production composite route scored 0.5764 on the same images.
Fourteen points, gone. Not a modelling regression: the two datasets carve up necklines
differently, and our mapping silently dropped the difference on the floor. Nothing in our
internal numbers could have surfaced it, because internally the taxonomy always agreed with
itself.

Three different failure modes, one common thread. Each time the model was wrong and
confident, and the evaluation was the only thing in the room that wasn't.

Which is the part worth saying plainly: **bad evaluation is worse than none.** With no
evaluation you know you're guessing, so you stay cautious. With flattering evaluation you
confidently ship the precision collapse.

## The harness, start to finish

All of that is policy until you see what actually happens when a model gets evaluated.
Here's the walk.

**Step 1: build the manifest.** Each track has a builder that pins everything — dataset
version, checksums, record IDs, group IDs, split roles, output vocabulary, alias maps,
missing-label rules, metric, bootstrap method, seed. The manifest exists before any model
runs.

```
python -m suite.dfmm18.build_manifest --source data/dfmm
# manifest: 5,000 test images, 1,751 product groups, 18 fields, sha256=...
```

We never redistribute the restricted datasets. You download them from their official
sources under their own terms; the builder reconstructs our exact splits from IDs and
checksums, and refuses to continue if your copy doesn't match.

**Step 2: split where leakage actually happens.** One Fashionpedia image yields several
garment crops. One DFMM product appears in several photos. Split crops randomly and the
same source image lands in train and test, and your score is quietly inflated. Bootstrap
crops independently and your confidence interval is quietly too narrow, because crops from
one image succeed and fail together.

So every track splits *and* resamples at its natural unit: source image for Fashionpedia,
image for Shopping100k, product group for DFMM. The DFMM test is the strictest thing we
have — 5,000 fresh images across 1,751 product groups, with zero record overlap and zero
product-group overlap against every experiment we had ever run before it.

The roles never blur, either. Training changes weights, calibration sets thresholds,
development picks checkpoints, and the frozen test is read once, at the end.

**Step 3: predict blind, then commit.** Your inference code gets images and a schema, never
labels. When it's done, you hash the prediction file:

```
python -m suite.dfmm18.commit --predictions preds.jsonl
# committed: preds.jsonl  sha256=9f3a...  rows=5000
```

That hash goes into the record before scoring. After this point you can't swap the file,
and neither can we. Every prediction file in our results directory — including the ones
from models that beat us — sits next to its hash.

**Step 4: the scorer opens the labels, and it fails closed.** Missing row, it stops.
Duplicate ID, it stops. An ID not in the manifest, a field outside the frozen vocabulary, a
hallucinated value the alias map can't resolve — all reported separately, where anyone can
see them. The one thing the scorer will never do is silently drop the rows a model got
wrong.

One scoring rule matters more than the others: an unobservable field is not a negative
example. If the photo crops out the waist, "waist: N/A" isn't a mistake. Fashionpedia
scores applicability as its own judgment for conditional fields, Shopping100k excludes
unlabelled cells, and DFMM scores N/A as a real class with its own F1. A model that
hallucinates every attribute it knows gets no credit for the habit.

**Step 5: paired bootstrap, then a gate that can't be averaged around.** Scores come with
10,000 paired bootstrap resamples clustered at the track's leakage unit, so the intervals
mean what they say. Promotion is conjunctive: the lower bound has to clear zero on every
required track. There's no weighted average for a bad track to hide inside.

That's the harness. It isn't exotic. It's every place we found where a benchmark can lie to
its owner, closed one at a time.

## What it told us

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

Not world-best, not universal SOTA, not a public-leaderboard result, not human-gold
quality, not production readiness. The harness is what let us word it that precisely.

**The loss we kept.** The first time we ran the full-body track, we lost it. FashionCLIP
with matched supervised heads beat our standalone model, and a hybrid beat both:

| System (earlier matched protocol) | Tier 1 | Tier 2 | Tier 3 |
|---|---:|---:|---:|
| MODA distilled, standalone | 0.5398 | 0.5260 | 0.4509 |
| FashionCLIP + linear heads | 0.5985 | 0.6237 | 0.5008 |
| MODA–FashionCLIP hybrid | **0.6492** | **0.6425** | **0.5353** |

We kept it. We picked the next standalone candidate on development data only, built the
fresh product-group-disjoint shadow so the rematch couldn't be contaminated by anything
we'd learned, and evaluated once. That rematch is the 0.6917 above. The earlier loss stays
in the record because it's the reason you can believe the win.

**What the numbers don't say.** Material sits at 0.4148 value-F1 on Fashionpedia. Fabric is
0.6693 and fit 0.6708 on Shopping100k. Collar style, neckline and rare applicability values
all lag. A conjunctive win means no track failed. It does not mean every attribute is
production-grade, and we'd rather you read that here than find it in your pipeline.

## What this decides about open and closed

Here's the part that surprised us. Once you accept that the harness is the asset, the
open-source question mostly answers itself.

Open the ruler. Sell what the ruler measured.

The harness goes out, all of it — protocol, splits, scorers, bootstrap code, our frozen
prediction files with their hashes, including the runs we lose. A benchmark nobody else can
run isn't a benchmark, it's an opinion with a table in it. It only becomes worth something
if other people use it, and the people most motivated to attack it are exactly the ones
who'd enjoy beating us, which is adversarial review we could never afford to commission.
Publishing it is also the only honest evidence for what we sell: if we tell a retailer we
can build this on their catalogue and prove it worked, they should be able to inspect the
machinery that would do the proving.

What stays closed is what the ruler selected. Calibrated per-field thresholds. The routing
layer that takes a declared schema and picks the head. Taxonomies fitted to particular
retailers' catalogues. Those are accumulated measurement outcomes tuned to specific data;
they don't generalise the way the method does, and they're what a customer is actually
buying.

Which reframes the pitch in a way we find much easier to say out loud. We're not selling
weights. We're selling the loop — the ability to run this measurement process against your
data and tell you, with intervals, whether it worked. The open harness is the receipt that
the loop is real.

## Two objections, and they're both fair

**"You built the test you pass."** Yes, and that deserves suspicion, so the answer has to be
structural rather than indignant. Predictions are committed by hash before any label opens,
so we can't retro-fit. We picked the strongest available comparator on each track rather
than the most convenient — on the crop track that meant benchmarking against the model that
already held the lead. The scorer fails closed, so we can't quietly drop rows we got wrong.
And we published a track we lost. A benchmark whose author only ever wins on it is worth
what you'd expect; ours has our losses in it.

**"It's not really open — some weights are non-commercial."** True, and the datasets are
the reason rather than us. Two of the three tracks are built on research-only corpora whose
terms we're not going to quietly ignore, so weights trained on them ship non-commercial —
and that binds us as well as you: those exact weights aren't in our paid product either.
The part that matters for reproduction — protocol, scorers, splits, prediction files —
carries no such restriction.

## Run your own model

Freeze your checkpoint, predict blind on the same IDs, commit your hash, score each track,
report the paired CI against our shipped predictions. Everything you need is in the repo.

If a competitor reads all this, runs the harness and beats us honestly, we'd rather know.
That isn't magnanimity. The alternative is finding out from a customer, in production, with
a taxonomy mismatch quietly eating fourteen points and nothing in our own instrumentation
able to see it.

We've already had that meeting with ourselves three times this summer. Building the harness
first is how we keep the number finite.

*Next: what a benchmark win doesn't tell you — independent human gold, the neckline
mismatch in detail, and why our production system is calibrated on none of the public data.*
