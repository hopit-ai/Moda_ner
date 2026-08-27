<h1 align="center">MODA_NER</h1>
<p align="center"><b>Open fashion attribute extraction: benchmark suite and models.</b><br>
Given a fashion image and a declared schema, extract the attributes — measured across three
frozen public tracks, competitors included, losses shown.</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/Code-MIT-yellow.svg" alt="MIT"></a>
  <a href="https://huggingface.co/HopitAI"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Models-HopitAI-blue" alt="Hugging Face"></a>
  <a href="https://hopit-ai.github.io/Moda_ner/"><img src="https://img.shields.io/badge/Benchmarks-four%20tracks-0f7b4a" alt="Benchmarks"></a>
  <a href="https://github.com/hopit-ai/Moda"><img src="https://img.shields.io/badge/Sibling-MODA%20retrieval-blue" alt="MODA"></a>
</p>

> **Status: pre-launch.** The suite, results, and model cards land here ahead of the public
> flip. Nothing in this repo overrides the frozen receipts in the private benchmark package.

## The claim, exactly

> MODA General is the best of the named open systems evaluated under the frozen public MODA
> General Attribute Suite, spanning localized garment crops, catalog product images, color and
> fit, and applicability-aware full-body attributes.

Not claimed: world-best, universal SOTA, public-leaderboard SOTA, human-gold quality,
production readiness. We publish the losses next to the wins.

## Three tracks, one question

No single public dataset covers fashion attributes at production breadth, so the suite keeps
three frozen tracks separate rather than averaging incompatible taxonomies:

| Track | Frozen test | Input contract | Covers | Known gap |
|---|---|---|---|---|
| **crop** | 4,688 garment crops / 1,158 images | oracle garment crop | category hierarchy, silhouette, sleeve, neckline, collar, closure, hemline, waist, pattern, material, surface treatment | no color/fit; localization supplied |
| **catalog** | 9,995 catalogue images / 61,384 cells | catalogue product image | category, collar, **color**, fabric, fastening, **fit**, neckline, pattern, pocket, sleeve length | no applicability/accessories |
| **fullbody** | 5,000 images / 1,751 product groups | full-body image | region fabric/pattern, neckline, sleeve, lengths, accessories, explicit **N/A** | closed vocabulary |
| **text** | 1,071 rows (765 standard + 306 hard) | product title or description | exact character spans for 13 entity types | labels are silver, not human gold |

Protocol, in one paragraph: splits and bootstrap resampling at each track's natural leakage
unit (source image / image / product group); training, calibration, development, and test
strictly separated; predictions generated label-blind and SHA-256-committed before a
fail-closed scorer opens labels; 10,000-sample paired clustered bootstrap; promotion requires
the 95% CI lower bound above zero on **every** track — no aggregate can hide a failed track.

## Frozen headline results

| Track / metric | MODA | Strongest comparator | Paired 95% CI |
|---|---:|---:|---:|
| Fashionpedia attribute micro-F1 | **0.6300** | FashionSigLIP 0.6245 | [+0.0014, +0.0097] |
| Shopping100k field-macro set F1 | **0.8292** | FashionCLIP 0.6657 | [+0.1595, +0.1676] |
| DFMM Tier-1 macro-F1 | **0.6917** | FashionCLIP 0.5943 | [+0.0891, +0.1053] |
| DFMM Tier-2 N/A-F1 | **0.6637** | FashionCLIP 0.6088 | [+0.0433, +0.0657] |
| DFMM Tier-3 visible macro-F1 | **0.5785** | FashionCLIP 0.4969 | [+0.0723, +0.0905] |

Weak fields, stated plainly: material 0.4148 (Fashionpedia value-F1), fabric 0.6693 and fit
0.6708 (Shopping100k), collar style, neckline, rare applicability values. A benchmark win
does not make every attribute production-perfect.

## Released models

Two families. **MODA_NER(V)** works on images, **MODA_NER(T)** on product text. Names describe
the input contract, never the corpus a model was measured against.

Release tiers:

```
*     open code + open weights
**    open weights only, code closed
***   closed code + closed weights, benchmarks published
```

No model currently ships at `**`. The tier is defined because we expect to use it, not
because anything sits there today.

| Model | Tier | Input | Weights | Best internal |
|---|---|---|---|---|
| **MODA_NER(V) — Crop** `moda-ner-v-crop` | `*` | garment crop | MIT | 0.6300 micro-F1 |
| **MODA_NER(V) — Catalog** `moda-ner-v-catalog` | `*` | catalogue image | CC BY-NC 4.0 | 0.8292 set-F1 |
| **MODA_NER(V) — Full-body** `moda-ner-v-fullbody` | `*` | full-body photo | CC BY-NC 4.0 | 0.6917 T1 macro-F1 |
| **MODA_NER(T)** | `***` | product text | not distributed | 0.8723 strict-span F1 |
| **MODA_NER Pro** | `***` | hosted | not distributed | benchmarks published |

MODA_NER Pro is the hosted tier, and MODA_NER(T) is a text model we have built but are not
distributing. In both cases the benchmark numbers are published alongside everything else;
the systems themselves are not. In practice a Pro engagement is a model fine-tuned
on the customer's own catalogue, which is both more accurate on their taxonomy and free of any
research-licence dependency. [Talk to us](https://hopit.ai).

Weights are heads and adapters on our own already-public MIT encoder,
[`HopitAI/moda-fashion-distilled`](https://huggingface.co/HopitAI/moda-fashion-distilled).
No third-party encoder is loaded at inference. FashionSigLIP appears in this work as a
distillation teacher and as the baseline we measure against, never as the served model —
each card's Provenance section says which.
Where a published checkpoint is not our strongest, the model card states both figures.

**On the non-commercial routes.** Two tracks are evaluated against research-only corpora whose
terms do not permit commercial use of models trained on them. We honour that, and it binds us
too: those weights are not in Hopit's hosted product. For production we fine-tune on the
customer's own catalogue, which raises accuracy on their taxonomy and yields a model with no
dependency on research-licensed data.

## What's available, and when

The suite opens one complete track at a time. A track is independently reproducible the day
it lands — manifest builder, scorer, bootstrap, and our frozen prediction files with their
hashes, including the baselines we lose to. We'd rather ship one track you can actually run
than three you can't.

| Track | Code | Weights | Status |
|---|---|---|---|
| crop | `suite/crop/` | `moda-ner-v-crop` (MIT) | ready — reproduces 0.6300 |
| catalog | `suite/catalog/` | `moda-ner-v-catalog` (CC BY-NC 4.0) | ready — reproduces 0.829235 |
| fullbody | `suite/fullbody/` | `moda-ner-v-fullbody` (CC BY-NC 4.0) | ready — reproduces all six scores and three intervals |
| text | `suite/text/` | none — model not distributed | ready — builder regenerates the splits |

No gold labels are committed here, on any track. Each track's builder reconstructs its split
from the corpus's official source, under that source's own terms, so you obtain the data and
we supply the protocol, the scorers and our predictions. The text track is the odd one out in
one respect: no model ships alongside it. We publish the benchmark and our number on it, and
keep the checkpoint.

Until the image tracks land, the conjunctive claim above is only partly checkable from this
repo.
The frozen results for every track are published regardless — what arrives on the dates
above is your ability to verify them yourself.

## Reproduce

Restricted corpora are never redistributed here. `suite/` ships manifest builders, ID lists,
vocabularies, scorers, and our frozen prediction files with SHA-256 commitments; gold labels
are rebuilt by you from each corpus's official source under its own terms. The sources are
identified in `REPRODUCE.md`, which is the one place we name them, because a builder cannot
work without knowing what to build from.

## Citing and credit

`CITATION.cff` in this repository has the machine-readable entry; GitHub's "Cite
this repository" button reads it directly. Every scorer also stamps its output
with the suite name and version, so a result file carries its own provenance:

```json
{"micro_f1": 0.6300, "suite": {"name": "MODA General Attribute Suite",
  "version": "v1", "frozen": "2026-08-24", "track": "fashionpedia-2020-val-moda-oracle-crop-v1"}}
```

If you report a number produced by this suite, cite it. Both weight licences we use carry an
attribution term (MIT requires the notice; CC BY-NC requires credit), and every scorer stamps
the suite name and version into its output so a result stays attributable after it is copied
out of context.

A paper describing the suite is in preparation, and will become the preferred citation.

## Compare your model

1. Freeze your checkpoint, inference code, and thresholds before scoring.
2. Predict on the identical test IDs without loading labels; commit prediction hashes.
3. Score each track independently with the shipped scorer, same grouping unit.
4. Report the paired 95% CI against the shipped predictions.

## Related

[`hopit-ai/Moda`](https://github.com/hopit-ai/Moda) — the retrieval sibling: benchmark,
harness, and models for fashion search. Text extraction (FashionNER) releases separately.
