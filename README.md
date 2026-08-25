<h1 align="center">MODA_NER</h1>
<p align="center"><b>Open fashion attribute extraction: benchmark suite and models.</b><br>
Given a fashion image and a declared schema, extract the attributes — measured across three
frozen public tracks, competitors included, losses shown.</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/Code-MIT-yellow.svg" alt="MIT"></a>
  <a href="https://huggingface.co/HopitAI"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Models-HopitAI-blue" alt="Hugging Face"></a>
  <a href="https://github.com/hopit-ai/Moda"><img src="https://img.shields.io/badge/Sibling-MODA%20retrieval-0f7b4a" alt="MODA"></a>
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
| **Fashionpedia MODA-15** | 4,688 garment crops / 1,158 images | oracle garment crop | category hierarchy, silhouette, sleeve, neckline, collar, closure, hemline, waist, pattern, material, surface treatment | no color/fit; localization supplied |
| **Shopping100k-10** | 9,995 catalog images / 61,384 cells | catalog product image | category, collar, **color**, fabric, fastening, **fit**, neckline, pattern, pocket, sleeve length | no applicability/accessories |
| **DFMM-18 fresh shadow** | 5,000 images / 1,751 product groups | full-body image | region fabric/pattern, neckline, sleeve, lengths, accessories, explicit **N/A** | closed vocabulary |

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

Weights are heads/adapters on the already-public MIT encoder
[`HopitAI/moda-fashion-distilled`](https://huggingface.co/HopitAI/moda-fashion-distilled).
The released checkpoints are strong open baselines; our best internal checkpoints stay in
the hosted tier, and each model card states BOTH numbers so nothing is implied.

| Route | HF model | Weight license | Released ckpt score | Best internal |
|---|---|---|---|---|
| Fashionpedia MODA-15 | `HopitAI/moda-ner-fashionpedia15` | MIT | see card | 0.6300 micro-F1 |
| Shopping100k-10 | `HopitAI/moda-ner-shopping100k10` | CC BY-NC 4.0 | see card | 0.8292 set-F1 |
| DFMM-18 | `HopitAI/moda-ner-dfmm18` | CC BY-NC 4.0 | see card | 0.6917 T1 macro-F1 |

CC BY-NC routes: the underlying datasets (Shopping100k, DeepFashion-MultiModal) are
non-commercial research resources; the weight license honors that, and it binds us too —
these weights are not part of Hopit's hosted product.

## What's available, and when

The suite opens one complete track at a time. A track is independently reproducible the day
it lands — manifest builder, scorer, bootstrap, and our frozen prediction files with their
hashes, including the baselines we lose to. We'd rather ship one track you can actually run
than three you can't.

| Track | Code | Weights | Status |
|---|---|---|---|
| Fashionpedia MODA-15 | `suite/fashionpedia_moda15/` | `moda-ner-fashionpedia15` (MIT) | **5 Sep 2026** |
| Shopping100k-10 | `suite/shopping100k_10/` | `moda-ner-shopping100k10` (CC BY-NC 4.0) | **10 Sep 2026** |
| DFMM-18 | `suite/dfmm18/` | `moda-ner-dfmm18` (CC BY-NC 4.0) | **12 Sep 2026** |

Until all three land, the conjunctive claim above is only partly checkable from this repo.
The frozen results for every track are published regardless — what arrives on the dates
above is your ability to verify them yourself.

## Reproduce

Restricted datasets are never redistributed here. `suite/` ships manifest builders, ID
lists, vocabularies, scorers, and our frozen prediction files with SHA-256 commitments;
gold labels are rebuilt from each dataset's official source by you. See `REPRODUCE.md`.

## Citing

`CITATION.cff` in this repository has the machine-readable entry; GitHub's "Cite
this repository" button reads it directly. Every scorer also stamps its output
with the suite name and version, so a result file carries its own provenance:

```json
{"micro_f1": 0.6300, "suite": {"name": "MODA General Attribute Suite",
  "version": "v1", "frozen": "2026-08-24", "track": "fashionpedia-2020-val-moda-oracle-crop-v1"}}
```

A paper describing the suite is in preparation.

## Compare your model

1. Freeze your checkpoint, inference code, and thresholds before scoring.
2. Predict on the identical test IDs without loading labels; commit prediction hashes.
3. Score each track independently with the shipped scorer, same grouping unit.
4. Report the paired 95% CI against the shipped predictions.

## Related

[`hopit-ai/Moda`](https://github.com/hopit-ai/Moda) — the retrieval sibling: benchmark,
harness, and models for fashion search. Text extraction (FashionNER) releases separately.
