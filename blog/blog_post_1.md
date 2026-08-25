# Attribute extraction for fashion - why the evaluation harness first?

**MODA_NER, part 1: the MODA General Attribute Suite**

*Series: Fashion attribute extraction from images*
*Sibling series: [Building a fashion search engine from scratch](https://github.com/hopit-ai/Moda)*

---

We spent most of a summer building a measuring device instead of a model.

Scroll the commit history for that period and the great majority of it is manifests,
scorers, hash commitments, bootstrap code, cost gates and integrity receipts. The model
that came out the other end is almost a byproduct. That looks like procrastination, and
for a while it felt like it.

I think it was the right order, though we did not arrive at it deliberately. We arrived
at it because a model nearly got past us. This post is that story, and at the end it
turns into the reason some of this work is public and some of it is not.

## The problem we actually had

Say you have a photo of a dress and you need structured facts about it. Category, sleeve
length, neckline, colour, fit, pattern, material. Not a caption, not a vibe: a row you
can put in a database, because a trend system or a catalogue pipeline is going to
consume it.

Which model should you use? We went looking for a benchmark that would answer that and
couldn't find one. Fashion datasets exist in quantity. The problem is that each covers a
different slice with a taxonomy that disagrees with every other one.

Fashionpedia has expert-annotated attributes localised to garment masks, which is
wonderful, and it has no colour and no fit, which are the two fields a merchandiser asks
about first. Shopping100k has colour and fit as clean structured fields, on catalogue
images only, with no concept of a field that doesn't apply to a garment.
DeepFashion-MultiModal has full-body photos, per-region labels and an explicit N/A
class, inside a closed vocabulary. VLM benchmarks measure captioning and VQA, which is a
different job.

The tempting move is to map all of it into one label space and publish one number. We
sketched that mapping and stopped, because every mapping choice was quietly a modelling
decision, and one number would have hidden which input contract a model actually fails
on. A model that's excellent on flat product shots and lost on street photos would score
"fine".

So we built three tracks instead, with three test sets and three metrics, and a rule
against averaging them. A system is as good as its worst required track.

## The model that nearly fooled us

We had models before we had a benchmark. That is the wrong order, and it is how most of
these projects actually go.

The first serious one was a Florence-2 extraction model: give it a garment image, get
back a JSON object with every field filled in. It worked well enough to keep going. Then
we ran a corrective training pass on it, scored the result, and recall had barely moved.
0.6013 to 0.5885. That is the kind of drift you shrug at.

Precision had fallen off a table. 0.5661 to 0.3158.

A single blended number would have read as roughly flat, ship it. What per-field scoring
showed instead was a model that had started guessing constantly and getting away with it
on average, because the wins on easy fields covered the collapse on hard ones.

Two things came out of that. The first was a diagnosis. One autoregressive sequence
emitting every field at once couples attributes that have nothing to do with each other,
lets rare labels get crowded out, and gives the model no way to say that a field does
not apply to this garment. We replaced it with conditional heads and an explicit
applicability decision, which is what we run today.

The second was less comfortable. We had come within one careless glance of shipping that
model, and nothing in how we were measuring would have stopped us. So we stopped
building models for a while and built the thing that could tell us when we were wrong.

## The harness, start to finish

All of that is policy until you see what happens when a model actually gets evaluated.

**Step 1: build the manifest.** Each track has a builder that pins everything: dataset
version, checksums, record IDs, group IDs, split roles, output vocabulary, alias maps,
missing-label rules, metric, bootstrap method, seed. The manifest exists before any
model runs.

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
crops independently and your confidence interval is quietly too narrow, because crops
from one image succeed and fail together.

So every track splits *and* resamples at its natural unit: source image for
Fashionpedia, image for Shopping100k, product group for DFMM. The DFMM test is the
strictest thing we have: 5,000 fresh images across 1,751 product groups, with zero
record overlap and zero product-group overlap against every experiment we had ever run
before it.

The roles never blur, either. Training changes weights, calibration sets thresholds,
development picks checkpoints, and the frozen test is read once, at the end.

**Step 3: predict blind, then commit.** Your inference code gets images and a schema,
never labels. When it's done, you hash the prediction file:

```
python -m suite.dfmm18.commit --predictions preds.jsonl
# committed: preds.jsonl  sha256=9f3a...  rows=5000
```

That hash goes into the record before scoring. After this point you can't swap the file,
and neither can we. Every prediction file in our results directory sits next to its
hash, including the ones from models that beat us.

**Step 4: the scorer opens the labels, and it fails closed.** Missing row, it stops.
Duplicate ID, it stops. An ID not in the manifest, a field outside the frozen
vocabulary, a hallucinated value the alias map can't resolve: all reported separately,
where anyone can see them. The one thing the scorer will never do is silently drop the
rows a model got wrong.

One scoring rule matters more than the others: an unobservable field is not a negative
example. If the photo crops out the waist, "waist: N/A" isn't a mistake. Fashionpedia
scores applicability as its own judgment for conditional fields, Shopping100k excludes
unlabelled cells, and DFMM scores N/A as a real class with its own F1. A model that
hallucinates every attribute it knows gets no credit for the habit.

**Step 5: paired bootstrap, then a gate that can't be averaged around.** Scores come
with 10,000 paired bootstrap resamples clustered at the track's leakage unit, so the
intervals mean what they say. Promotion is conjunctive: the lower bound has to clear
zero on every required track. There's no weighted average for a bad track to hide
inside.

That's the harness. It isn't exotic. It's every place we found where a benchmark can lie
to its owner, closed one at a time.

## What it caught next

With the harness in place, the pattern repeated. Not dramatically. Mostly it just kept
quietly declining to let us believe things.

It killed a backbone swap for the price of a coffee. We wanted to know whether a general
SigLIP-2 encoder could replace the fashion-pretrained one, so we ran both through the
development gate on identical rows and an identical schedule. SigLIP-2 came out at
0.6018 development micro-F1 against 0.6163. That decided it, for about eleven cents of
compute, without anyone needing to have an opinion.

Then it made us publish a loss. The first time we ran the full-body track, FashionCLIP
with matched supervised heads beat our standalone model, and a hybrid of the two beat
both:

| System (earlier matched protocol) | Tier 1 | Tier 2 | Tier 3 |
|---|---:|---:|---:|
| MODA distilled, standalone | 0.5398 | 0.5260 | 0.4509 |
| FashionCLIP + linear heads | 0.5985 | 0.6237 | 0.5008 |
| MODA-FashionCLIP hybrid | **0.6492** | **0.6425** | **0.5353** |

We kept that result rather than burying it. Then we picked the next standalone candidate
using development data only, built a fresh test split with no product group in common
with anything we had run before, and evaluated once.

It also stopped us buying a result we wanted. A commercial VLM looked like it might beat
us on colour specifically, 0.730 against our 0.630, which is exactly the sort of number
that ends up on a slide. It came from 27 eligible rows. We had put a 100-row checkpoint
gate in front of the full 1,000-row run precisely so that a small team could afford
honest baselines, and the model lost the general comparison at that gate, so the
remaining 900 paid calls were never made. We chased the colour result separately on 400
balanced rows and it evaporated. The paired interval came out entirely negative.
Twenty-seven rows is not a finding, it is a rumour.

## Where that leaves the numbers

| Track / primary metric | MODA | Strongest comparator | Paired 95% CI |
|---|---:|---:|---:|
| Fashionpedia attribute micro-F1 | **0.6300** | FashionSigLIP 0.6245 | [+0.0014, +0.0097] |
| Shopping100k field-macro set F1 | **0.8292** | FashionCLIP 0.6657 | [+0.1595, +0.1676] |
| DFMM Tier-1 macro-F1 | **0.6917** | FashionCLIP 0.5943 | [+0.0891, +0.1053] |
| DFMM Tier-2 N/A-F1 | **0.6637** | FashionCLIP 0.6088 | [+0.0433, +0.0657] |
| DFMM Tier-3 visible macro-F1 | **0.5785** | FashionCLIP 0.4969 | [+0.0723, +0.0905] |

The 0.6917 on the first line of the DFMM rows is that rematch. The earlier loss stays in
the record because it is the reason you can believe the win.

Here is the exact claim we allow ourselves:

> MODA General is the best of the named open systems evaluated under the frozen public MODA
> General Attribute Suite, spanning localized garment crops, catalog product images, color
> and fit, and applicability-aware full-body attributes.

Not world-best, not universal SOTA, not a public-leaderboard result, not human-gold
quality, not production readiness. We could only word it that tightly because the
harness told us where the edges were.

Some fields are still weak, and we would rather you read it here than find it in your
pipeline. Material sits at 0.4148 value-F1 on Fashionpedia. Fabric is 0.6693 and fit
0.6708 on Shopping100k. Collar style, neckline and rare applicability values all lag. A
conjunctive win means no track failed. It does not mean every attribute is
production-grade.

## The one it caught last

The most recent catch is the one that still bothers me, because our own harness could
not have found it.

We ran the system against an independent, human-labelled external set, 1,110 images that
nobody on our side had touched. It scored 0.7169 against the FashionCLIP baseline at
0.6605, interval clear of zero. A genuine win.

Then the production composite route scored 0.5764 on those same images. Fourteen points,
gone.

It was not a modelling regression. The two datasets carve up necklines differently, and
our mapping between them quietly dropped the difference on the floor. Nothing in our
internal numbers could have surfaced that, because internally the taxonomy always agrees
with itself. It took labels produced by people with no connection to us to make the gap
visible, which is a fairly pointed argument for external evaluation over more of your
own.

That one gets its own post, because the details matter more than the headline.

## What this decides about open and closed

The part that surprised us is that once you accept the harness is the asset, the
open-source question mostly answers itself.

Open the ruler. Sell what the ruler measured.

The harness goes out, all of it: protocol, splits, scorers, bootstrap code, our frozen
prediction files with their hashes, including the runs we lose. A benchmark nobody else
can run is just an opinion with a table in it. It only becomes worth something if other
people use it, and the people most motivated to attack it are exactly the ones who'd
enjoy beating us, which is adversarial review we could never afford to commission.
Publishing it is also the only honest evidence for what we sell: if we tell a retailer
we can build this on their catalogue and prove it worked, they should be able to inspect
the machinery that would do the proving.

What stays closed is what the ruler selected. Calibrated per-field thresholds. The
routing layer that takes a declared schema and picks the head. Taxonomies fitted to
particular retailers' catalogues. Those are accumulated measurement outcomes tuned to
specific data; they don't generalise the way the method does, and they're what a
customer is actually buying.

That reframes the pitch in a way we find easier to say out loud. We aren't selling
weights. We're selling the loop: the ability to run this measurement process against
your data and tell you, with intervals, whether it worked. Publishing the harness is how
you can check that the loop exists before you pay for it.

## Two objections, and they're both fair

**"You built the test you pass."** Yes, and that deserves suspicion, so the answer has
to be structural rather than indignant. Predictions are committed by hash before any
label opens, so we can't retro-fit. We picked the strongest available comparator on each
track rather than the most convenient. On the crop track that meant benchmarking against
the model that already held the lead. The scorer fails closed, so we can't quietly drop
rows we got wrong. And we published a track we lost. A benchmark whose author only ever
wins on it is worth what you'd expect; ours has our losses in it.

**"It's not really open, some of the weights are non-commercial."** True, and the
datasets are the reason rather than us. Two of the three tracks are built on
research-only corpora whose terms we're not going to quietly ignore, so weights trained
on them ship non-commercial, and that binds us as well as you: those exact weights
aren't in our paid product either. The part that matters for reproduction, meaning
protocol, scorers, splits and prediction files, carries no such restriction.

## Run your own model

Freeze your checkpoint, predict blind on the same IDs, commit your hash, score each
track, report the paired CI against our shipped predictions. Everything you need is in
the repo.

If a competitor reads all this, runs the harness and beats us honestly, we'd rather
know. That isn't magnanimity. The alternative is finding out from a customer, in
production, with a taxonomy mismatch quietly eating fourteen points and nothing in our
own instrumentation able to see it.

We have had that meeting with ourselves several times since June. Building the harness
first is how we keep the number small.

*Next: what a benchmark win doesn't tell you. Independent human gold, the neckline
mismatch in detail, and why our production system is calibrated on none of the public
data.*
