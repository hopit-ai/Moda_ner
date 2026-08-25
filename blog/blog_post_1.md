# Attribute extraction for fashion — why the evaluation harness first?

**MODA_NER, part 1: the MODA General Attribute Suite**

*Series: Fashion attribute extraction from images*
*Sibling series: [Building a fashion search engine from scratch](https://github.com/hopit-ai/Moda)*

---

For those of you who have worked in AI long enough, it shouldn't come as a surprise that
as we start our new series on fashion attribute extraction, we talk about the evaluation
framework in our first post. For those of you who have worked in fashion or ecommerce
long enough, it shouldn't come as a surprise that we are doing a full series on
attribute extraction: the harness, our closed models and our open models. In some senses
it might be the most unfashionable thing to do a series like this, but it might also be
the most useful for the community at large. I don't remember who this was, but there was
a famous AI researcher who said he would spend 90% of his time building the evaluation
framework and making it airtight, and build the model in the remaining 10%.

So, what's the problem we are solving here?

Say you have a photo of a dress and you need structured facts about it. Category, sleeve
length, neckline, colour, fit, pattern, material. An entry in your database, because all
your downstream systems depend on it, from cataloguing to trends to search and
personalisation. Which model should you use? Do the general large vision models do a
good job? What's the cost tradeoff? Can they be used at runtime?

We tried to find a benchmark. We were fairly unsuccessful, and having talked to our
customers, we identified this as a sharp customer need. There are an innumerable number
of fashion datasets, but each of them covers a different part of the taxonomy and these
taxonomies don't agree with each other.

Fashionpedia has expert-annotated attributes localised to garment masks, but it has no
colour and no fit. The merchandiser cares about those fields more than anything.
Shopping100k has colour and fit as structured fields, on catalogue images only, with no
concept of a field that doesn't apply to a garment. DeepFashion-MultiModal has full-body
photos, per-region labels and an explicit N/A class, inside a closed vocabulary. VLM
benchmarks measure captioning and VQA. The mess is real.

The easy thing for us to do was to map it all into one label space and publish a magic
winning number. We sketched that plan multiple times over, and abandoned it every time,
because each mapping choice is itself a modelling decision. A model that works on flat
product shots but loses on street photos would have scored well. What we ended up doing
was three tracks, three test sets and three metrics, with a rule that we will not
average them. A system is as good as its weakest component.

## What's actually in the harness

The suite is four things, and none of them are a model.

**Track definitions.** Three of them, each pinning a dataset version, the record and group
IDs in the frozen test, the output vocabulary, the alias map that resolves near-miss
values, the missing-label rule, the metric, the bootstrap method and the seed. A track
is a contract about what a model will be asked and how the answer will be judged.

| Track | Frozen test | Input contract | Leakage unit |
|---|---|---|---|
| Fashionpedia MODA-15 | 4,688 garment crops from 1,158 images | oracle garment crop | source image |
| Shopping100k-10 | 9,995 catalogue images, 61,384 eligible cells | catalogue product image | image |
| DFMM-18 | 5,000 images across 1,751 product groups | full-body photo | product group |

**Manifest builders.** Restricted datasets are never redistributed. You download them from
their official sources under their own terms, and a builder reconstructs our exact
splits from IDs and checksums. If your copy doesn't match ours byte for byte, it refuses
to continue rather than silently scoring you on different data.

**Scorers and the bootstrap.** One strict scorer per track, plus 10,000-sample paired
bootstrap resampling clustered at that track's leakage unit. The scorers are the part we
were most careful with, and the part we most expect people to attack.

**The frozen evidence.** Every prediction file from every system we evaluated, ours and the
baselines', each next to its SHA-256 commitment, along with the result JSONs. This is
what makes the numbers in this post checkable rather than assertable. The runs we lose
are in there too.

There are 147 focused tests across the suite. Most of them exist because something went
wrong once.

## How it got built, in the order it happened

We had models before we had a benchmark. That's the wrong order, and it's how most of
these projects actually go.

The first serious one was a Florence-2 extraction model: give it a garment image, get
back a JSON object with every field filled in. It worked well enough to keep going. Then
we ran a corrective training pass on it, scored the result, and recall had barely moved.
0.6013 to 0.5885. That's the kind of drift you shrug at.

Precision had fallen off a table. 0.5661 to 0.3158.

A single blended number would have read as roughly flat, ship it. What per-field scoring
showed instead was a model that had started guessing constantly and getting away with it
on average, because wins on easy fields covered a collapse on hard ones. We had come
within one careless glance of shipping it.

Two things came out of that. One was a diagnosis: a single autoregressive sequence
emitting every field at once couples attributes that have nothing to do with each other,
lets rare labels get crowded out, and gives the model no way to say a field doesn't
apply to this garment. We replaced it with conditional heads and an explicit
applicability decision, which is what we run today.

The other was that we stopped building models for a while.

The first track we built was Fashionpedia, because expert mask-level annotation was the
highest-quality signal available. Almost immediately it couldn't answer the question our
customers actually asked, which was about colour and fit, so Shopping100k became the
second track. Then both of them turned out to share a blind spot: neither has any
concept of a field that doesn't apply, so a model that hallucinates an attribute for
every garment looks identical to one that knows when to abstain. DFMM became the third
track because it scores N/A as a real class.

Each track exists because the previous ones couldn't answer something. None of them was
planned.

## The five things that happen when you score a model

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

## What it caught

Once the harness existed, the pattern repeated. Not dramatically. Mostly it just kept
quietly declining to let us believe things.

It killed a backbone swap for the price of a coffee. We wanted to know whether a general
SigLIP-2 encoder could replace the fashion-pretrained one, so both went through the
development gate on identical rows and an identical schedule. SigLIP-2 came out at
0.6018 development micro-F1 against 0.6163. Decided, for about eleven cents of compute,
without anyone needing to have an opinion about it.

Then it made us publish a loss. The first time we ran the full-body track, FashionCLIP
with matched supervised heads beat our standalone model, and a hybrid of the two beat
both:

| System (earlier matched protocol) | Tier 1 | Tier 2 | Tier 3 |
|---|---:|---:|---:|
| MODA distilled, standalone | 0.5398 | 0.5260 | 0.4509 |
| FashionCLIP + linear heads | 0.5985 | 0.6237 | 0.5008 |
| MODA-FashionCLIP hybrid | **0.6492** | **0.6425** | **0.5353** |

We kept that result rather than burying it, picked the next standalone candidate using
development data only, built a fresh test split with no product group in common with
anything we had run before, and evaluated once.

It also stopped us buying a result we wanted. A commercial VLM looked like it might beat
us on colour specifically, 0.730 against our 0.630, which is exactly the sort of number
that ends up on a slide. It came from 27 eligible rows. We had put a 100-row checkpoint
gate in front of the full 1,000-row run precisely so a small team could afford honest
baselines, the model lost the general comparison at that gate, and the remaining 900
paid calls were never made. We chased the colour result separately on 400 balanced rows
and it evaporated, with the paired interval entirely negative. Twenty-seven rows is not
a finding, it's a rumour.

## Where that leaves the numbers

| Track / primary metric | MODA | Strongest comparator | Paired 95% CI |
|---|---:|---:|---:|
| Fashionpedia attribute micro-F1 | **0.6300** | FashionSigLIP 0.6245 | [+0.0014, +0.0097] |
| Shopping100k field-macro set F1 | **0.8292** | FashionCLIP 0.6657 | [+0.1595, +0.1676] |
| DFMM Tier-1 macro-F1 | **0.6917** | FashionCLIP 0.5943 | [+0.0891, +0.1053] |
| DFMM Tier-2 N/A-F1 | **0.6637** | FashionCLIP 0.6088 | [+0.0433, +0.0657] |
| DFMM Tier-3 visible macro-F1 | **0.5785** | FashionCLIP 0.4969 | [+0.0723, +0.0905] |

The DFMM Tier-1 number is that rematch. The earlier loss stays in the record because
it's the reason you can believe the win.

Here is the exact claim the harness lets us make:

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

## The one it couldn't catch

The most recent catch is the one that still bothers me, because our own harness could
not have found it.

We ran the system against an independent, human-labelled external set, 1,110 images
nobody on our side had touched. It scored 0.7169 against the FashionCLIP baseline at
0.6605, interval clear of zero. A genuine win.

Then the production composite route scored 0.5764 on those same images. Fourteen points,
gone.

It wasn't a modelling regression. The two datasets carve up necklines differently, and
our mapping between them quietly dropped the difference on the floor. Nothing in our
internal numbers could have surfaced that, because internally the taxonomy always agrees
with itself. It took labels produced by people with no connection to us to make the gap
visible.

That's the current boundary of what this harness can do, and it's why the next section
is about where it goes rather than what it has done.

## Where the harness goes next

Four things, roughly in order of how much they'd change what we can claim.

**Human gold on a public track.** Every label in the suite today is dataset-native. That's
legitimate, but it means our numbers inherit whatever each dataset's annotators decided.
We have 1,993 Fashionpedia image groups committed for independent annotation, targeting
at least 4,000 garment rows across at least 1,000 groups, with predictions and
thresholds committed before any label opens. Until that lands, no number here is
human-gold and we won't describe one as such.

**More comparators, at full size.** The current external set is FashionCLIP, FashionSigLIP,
one open VLM and one cost-gated commercial model. That's the thinnest part of the story.
The harness makes each addition mostly compute and discipline rather than new code, so
the fix is to run five or more modern open VLMs at full row counts rather than at a
checkpoint gate.

**A fourth track.** Pattern and neckline on iMaterialist, built as a streaming pass that
fingerprints and deduplicates without ever persisting an image corpus. It's built and
unexecuted, gated behind an environment lock, and it will only count if enough of the
source URLs survive to clear a hard floor of 2,000 deduplicated representatives. If they
don't, we drop it and say so.

**Somebody else's holdout.** The strongest version of this is a test set we didn't build,
administered by someone who isn't us. Everything above still has our fingerprints on the
protocol. A genuinely external evaluation is the only thing that fully answers the
objection below, and we'd rather walk toward it than argue about it.

There's also a paper in preparation describing the suite. When it lands we'll add it as
the preferred citation.

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
