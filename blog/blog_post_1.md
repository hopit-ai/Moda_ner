# We couldn't find a benchmark, so we built one

**MODA_NER, part 1: the MODA General Attribute Suite**

*Series: Fashion attribute extraction from images*
*Sibling series: [Building a fashion search engine from scratch](https://github.com/hopit-ai/Moda)*

---

## The result upfront

We evaluated fashion attribute extraction across three frozen public tracks and compared our
system against the strongest open baseline we could find on each one. MODA wins every track,
and every paired 95% confidence interval excludes zero:

| Track / primary metric | MODA | Strongest comparator | Paired 95% CI |
|---|---:|---:|---:|
| Fashionpedia attribute micro-F1 | **0.6300** | FashionSigLIP 0.6245 | [+0.0014, +0.0097] |
| Shopping100k field-macro set F1 | **0.8292** | FashionCLIP 0.6657 | [+0.1595, +0.1676] |
| DFMM Tier-1 macro-F1 | **0.6917** | FashionCLIP 0.5943 | [+0.0891, +0.1053] |
| DFMM Tier-2 N/A-F1 | **0.6637** | FashionCLIP 0.6088 | [+0.0433, +0.0657] |
| DFMM Tier-3 visible macro-F1 | **0.5785** | FashionCLIP 0.4969 | [+0.0723, +0.0905] |

The exact claim, and its boundary:

> MODA General is the best of the named open systems evaluated under the frozen public MODA
> General Attribute Suite, spanning localized garment crops, catalog product images, color
> and fit, and applicability-aware full-body attributes.

Not claimed: world-best, universal SOTA, a public-leaderboard result, human-gold quality, or
production readiness. This post explains why the benchmark had to exist before the claim
could, and why we kept a loss in the record on the way.

## The gap

Say you need structured attributes from a fashion image — category, silhouette, sleeve
length, neckline, color, fit, pattern, material — because a trend system or a catalog
enrichment pipeline consumes them. Which public benchmark tells you which model to use?

None of them, it turns out:

- **Fashionpedia** has expert-annotated, mask-localized attributes — but no color and no
  fit, two of the fields any commercial consumer cares about most.
- **Shopping100k** has color and fit as structured fields — but clean catalog images only,
  and no notion of a field that does not apply.
- **DeepFashion-MultiModal** has full-body images with per-region labels and explicit N/A —
  but a closed vocabulary.
- General VLM benchmarks measure captioning and VQA, not schema-faithful extraction.

Each dataset covers a different slice with an incompatible taxonomy. The tempting move is
to map everything into one label space and average. We think that is how you fool yourself:
the mapping choices become invisible modeling decisions, and one aggregate number hides
which contract your model actually fails.

So the suite keeps three tracks separate, each with its own frozen test set, input
contract, and metric. A system is only as good as its worst required track.

## The design choices that matter

**Three different leakage units.** A Fashionpedia image yields several garment crops; a
DFMM product appears in several photos. Split or bootstrap at the wrong unit and your
confidence intervals are fiction. We split and resample at the natural unit per track:
source image (Fashionpedia), image (Shopping100k), product group (DFMM). The DFMM test is
the strictest: 5,000 fresh images across 1,751 product groups with zero record overlap and
zero product-group overlap against every earlier experiment we ran.

**Four-way data separation.** Training changes weights. Calibration selects thresholds.
Development selects architectures and checkpoints. The frozen test is touched once, for the
final comparison. Nothing else ever reads it.

**Commit before you score.** Prediction files are generated without access to labels and
SHA-256 committed. Only then does a strict scorer load the gold labels — and it fails
closed on any missing, duplicate, or extra ID. You cannot quietly drop the rows you got
wrong.

**An unobservable field is not a negative example.** If a photo does not show the waist,
"waist type: none" is not an error the model made. Fashionpedia scores applicability
separately for conditional fields; Shopping100k excludes unlabeled cells; DFMM scores N/A
as an explicit class. Systems that hallucinate every possible attribute get no credit for
it here.

**A conjunctive gate.** Promotion requires the 95% CI lower bound above zero on every
required track, computed from 10,000 paired bootstrap samples at the track's leakage unit.
There is no weighted average for a failed track to hide inside.

## The loss we kept

The first time we ran the full-body track, our standalone model lost. FashionCLIP with
matched supervised heads beat it on the earlier protocol, and a MODA–FashionCLIP hybrid
beat both:

| System (earlier matched protocol) | Tier 1 | Tier 2 | Tier 3 |
|---|---:|---:|---:|
| MODA distilled (standalone) | 0.5398 | 0.5260 | 0.4509 |
| FashionCLIP + linear heads | 0.5985 | 0.6237 | 0.5008 |
| MODA–FashionCLIP hybrid | **0.6492** | **0.6425** | **0.5353** |

We published that result internally, kept it in the record, selected the next standalone
candidate using development data only, and evaluated it once on the fresh
product-group-disjoint shadow. That rematch is the 0.6917 in the headline table. The
earlier loss is not an embarrassment; it is the reason you can believe the win.

The same discipline cut a commercial-model comparison short: an image-only Gemini 3.5 Flash
run started with a 100-row checkpoint gate across all three tracks. It scored well below
MODA on the checkpoint (0.3624 vs 0.6667 on Fashionpedia rows, 0.5830 vs 0.8057 on
Shopping100k, 0.3522 vs 0.4482 on DFMM Tier-1), so the gate failed and the remaining 900
paid calls were never purchased. A promising color-only signal on 27 eligible rows did not
replicate on a 400-row balanced follow-up. Cost gates are how a small team affords honest
baselines.

## What the numbers do not say

Weak fields, stated plainly: material (0.4148 value-F1 on Fashionpedia), fabric (0.6693)
and fit (0.6708) on Shopping100k, collar style, neckline, and rare applicability values. A
conjunctive benchmark win does not make every attribute production-grade, and we would
rather you read that here than discover it in your pipeline.

## Run your own model

Everything needed to evaluate a new system is in the repo: manifest builders, ID lists,
vocabularies, fail-closed scorers, bootstrap code, and our frozen prediction files with
their hashes. Restricted datasets are never redistributed — you rebuild gold labels from
each dataset's official source, then:

1. Freeze your checkpoint, inference code, and thresholds.
2. Predict on the identical test IDs without loading labels; commit your prediction hashes.
3. Score each track independently with the shipped scorer.
4. Report the paired 95% CI against our shipped predictions.

If your lower bound clears zero on every track, we want to hear about it.

*Next in the series: what a benchmark win doesn't tell you — independent human gold, a
taxonomy mismatch that cost 14 points, and why our production system is calibrated on none
of the data above.*
