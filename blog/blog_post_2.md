# Attribute extraction for fashion — why the evaluation harness first?

**MODA_NER, part 2**

*Series: Fashion attribute extraction from images*
*Previous: [We couldn't find a benchmark, so we built one](blog_post_1.md)*

---

We spent most of a summer building a measuring device instead of a model.

If you scroll the commit history for that period, the great majority of it is
manifests, scorers, hash commitments, bootstrap code, cost gates and integrity
receipts. The model that came out the other end is almost a byproduct. That looks
like procrastination, and for a while it felt like it. This post is the argument
for why it was the right order, and what it ended up deciding about which parts of
the work we give away and which parts we sell.

## Three times the measurement caught a lie

The abstract case for evaluation is boring and everyone nods at it. The concrete
case is better, so here are three occasions where our own numbers were about to
walk us off a cliff.

**The model that looked fine and wasn't.** We ran a corrective training pass on a
Florence-2 extraction model. Recall barely moved: 0.6013 to 0.5885, the kind of
drift you shrug at. Underneath, precision had fallen off a table — 0.5661 to
0.3158. A single blended headline number would have read as "roughly flat, ship
it." Per-field scoring showed a model that had started guessing constantly and
getting away with it on aggregate. That failure is also what killed the
architecture: one autoregressive sequence emitting every field together couples
attributes that have nothing to do with each other, and gives the model no way to
say a field doesn't apply. We replaced it with conditional heads and an explicit
applicability decision, which is the design we run today.

**The signal that wasn't there.** A commercial VLM looked like it might beat us on
colour specifically — 0.730 against our 0.630. Genuinely interesting, and exactly
the sort of result that gets a slide. It was computed on 27 eligible rows. We ran
a 400-row label-balanced follow-up and the effect vanished; the paired interval
came out entirely negative. Twenty-seven rows is not a finding, it's a rumour, and
the only reason we didn't repeat it publicly is that the protocol made us check.

**The fourteen points hiding in a vocabulary.** Our system scored 0.7169 on an
independent, human-labelled external set — a real win over the FashionCLIP
baseline at 0.6605, interval clear of zero. Then the production composite route
scored 0.5764 on the same images. Fourteen points, gone. Not a modelling
regression: the two datasets carve up necklines differently, and our mapping
silently dropped the difference on the floor. Nothing in our internal numbers
could have surfaced that, because internally the taxonomy always agreed with
itself.

Three different failure modes, one common thread. In each case the model was
wrong and confident, and the evaluation was the only thing in the room that
wasn't.

## Why first, and not later

The usual sequence is to build the thing, get it working, then figure out how to
measure it. That's fine when you already know what good looks like. In fashion
attribute extraction we didn't, because no benchmark covered the fields anyone
actually asks for. Fashionpedia has no colour or fit. Shopping100k has no notion
of a field that doesn't apply. Everything else is captioning under a different
hat.

So every decision after the first one was a selection problem: which architecture,
which checkpoint, whether the frozen fashion encoder beats the general one,
whether the vendor model is worth its bill, whether we're done. Selection is only
as good as the ruler you select with. Build the model first and you spend months
choosing between options using a measurement you haven't earned the right to
trust — and you won't feel it going wrong, because a flattering evaluation feels
exactly like progress.

Which is the part worth saying plainly: **bad evaluation is worse than none.** No
evaluation, and you know you're guessing, so you stay cautious. Flattering
evaluation, and you confidently ship the precision collapse.

It's an unglamorous investment. It demos badly, it produces mostly bad news about
your own work, and when a matched SigLIP-2 backbone lost to the domain-pretrained
one at the development gate, all that expensive apparatus bought us was the right
to not run the expensive experiment. That's the job. Most of what a good harness
returns is experiments you didn't have to pay for and claims you didn't get to
make.

## What that decides about open and closed

Here's the part that surprised us: once you accept that the harness is the asset,
the open-source question mostly answers itself.

Open the ruler. Sell what the ruler measured.

The harness goes out, all of it — protocol, splits, scorers, bootstrap code, our
frozen prediction files with their hashes, including the runs where we lose. A
benchmark nobody else can run is not a benchmark, it's an opinion with a table in
it. It only becomes worth anything if other people use it, and the people most
motivated to attack it are precisely the ones who'd enjoy beating us, which is
adversarial review we could never afford to commission. Publishing it is also the
only honest evidence for what we sell: if we tell a retailer we can build this on
their catalogue and prove it worked, they should be able to inspect the machinery
that would do the proving.

What stays closed is everything the ruler selected. Calibrated per-field
thresholds. The routing layer that takes a declared schema and picks the head.
Taxonomies fitted to specific retailers' catalogues. Those are accumulated
measurement outcomes tuned to particular data, they don't generalise the way the
method does, and they're the part a customer is actually buying.

Which reframes the commercial pitch in a way we find much easier to say out loud.
We're not selling weights. We're selling the loop — the ability to run this
measurement process against your data and tell you, with intervals, whether it
worked. The open harness is the receipt that the loop is real.

## Two objections, and they're both fair

**"You built the test you pass."** Yes, and that's a legitimate thing to be
suspicious of, so the answer has to be structural rather than indignant. We commit
prediction files by hash before any label is opened, so we can't retro-fit. We
picked the strongest available comparator on each track rather than the most
convenient one — on the crop track that meant benchmarking against the model that
already held the lead. The scorer fails closed, so we can't quietly drop rows we
got wrong. And we published a track we lost, where FashionCLIP beat our standalone
model and a hybrid beat both, before a later candidate took it back on a fresh
product-disjoint split. A benchmark whose author only ever wins on it is worth
what you'd expect. Ours has our losses in it.

**"It's not really open — some of the weights are non-commercial."** True, and the
reason is the datasets rather than us. Two of the three tracks are built on
research-only corpora whose terms we're not going to quietly ignore, so the
weights trained on them ship non-commercial, and that binds us as well as you:
those exact weights aren't in our paid product either. The part that matters for
reproduction — protocol, scorers, splits, prediction files — carries no such
restriction.

## The uncomfortable version

If a competitor reads all of this, runs our harness, and beats our numbers
honestly, we'd rather know. That's not magnanimity. It's that the alternative is
finding out from a customer, in production, with a taxonomy mismatch quietly
eating fourteen points and nothing in our own instrumentation able to see it.

We've already had that meeting with ourselves three times this summer. Building
the harness first is how we keep the number of times finite.

*Next: what a benchmark win doesn't tell you — independent human gold, the
neckline mismatch in detail, and why our production system is calibrated on none
of the public data.*
