"""Suite identity, stamped onto every scored result.

Any scorer in this repository prints the banner and stamps its result payload.
The point is propagation: a number pasted out of a result JSON should still say
which suite and which version produced it, months later and three hands away.

VERSIONING RULE (from the frozen protocol): any change to data, split, taxonomy,
metric, comparator, or router creates a NEW suite version. Bump SUITE_VERSION in
the same commit as the change; never redefine an existing version in place.
"""

from __future__ import annotations

SUITE_NAME = "MODA General Attribute Suite"
SUITE_VERSION = "v1"
SUITE_FROZEN = "2026-08-24"
SUITE_URL = "https://github.com/hopit-ai/Moda_ner"

# Track identifiers. The Fashionpedia id matches the frozen internal release
# name; the other two are reconciled against their frozen release names during
# the suite port and must not drift afterwards.
TRACKS = {
    "fashionpedia_moda15": "fashionpedia-2020-val-moda-oracle-crop-v1",
    "shopping100k_10": "shopping100k-10-blind-v1",
    "dfmm18": "dfmm18-fresh-shadow-v1",
}

CITATION = (
    f"{SUITE_NAME} ({SUITE_VERSION}), Hopit AI, {SUITE_FROZEN}. {SUITE_URL}"
)


def banner(track: str | None = None) -> str:
    """One-line provenance header. Scorers print this to stderr so that stdout
    stays clean for machine-readable output."""
    head = f"{SUITE_NAME} {SUITE_VERSION} (frozen {SUITE_FROZEN})"
    if track:
        head += f" | track: {TRACKS.get(track, track)}"
    return f"{head}\n{SUITE_URL}"


def stamp(payload: dict, track: str | None = None) -> dict:
    """Return `payload` with a suite provenance block attached.

    Applied at the end of scoring, so the identity travels with the numbers
    wherever the JSON is copied. Existing keys are never overwritten.
    """
    block = {
        "name": SUITE_NAME,
        "version": SUITE_VERSION,
        "frozen": SUITE_FROZEN,
        "url": SUITE_URL,
        "citation": CITATION,
    }
    if track:
        block["track"] = TRACKS.get(track, track)
    out = dict(payload)
    out.setdefault("suite", block)
    return out


def markdown_footer(track: str | None = None) -> str:
    """Attribution line appended under any table a scorer renders."""
    t = f" — {TRACKS.get(track, track)}" if track else ""
    return f"_Scored with [{SUITE_NAME} {SUITE_VERSION}]({SUITE_URL}){t}._"
