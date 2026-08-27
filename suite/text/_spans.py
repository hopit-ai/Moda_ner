"""Span alignment and BIO encoding for fashion NER."""

from __future__ import annotations

import re
from typing import Any

from ._labels import LABEL2ID

Entity = dict[str, Any]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def find_span(text: str, phrase: str) -> tuple[int, int] | None:
    """Find first case-insensitive span of phrase in text."""
    if not phrase or not text:
        return None
    text_n = text.lower()
    phrase_n = phrase.strip().lower()
    if not phrase_n:
        return None
    idx = text_n.find(phrase_n)
    if idx >= 0:
        return idx, idx + len(phrase)
    # Try matching significant tokens in order (e.g. "sage green" from title words)
    tokens = [t for t in re.split(r"[\s\-/,]+", phrase_n) if len(t) > 1]
    if len(tokens) < 2:
        return None
    positions: list[tuple[int, int]] = []
    search_from = 0
    for tok in tokens:
        pos = text_n.find(tok, search_from)
        if pos < 0:
            return None
        positions.append((pos, pos + len(tok)))
        search_from = pos + len(tok)
    start = positions[0][0]
    end = positions[-1][1]
    return start, end


def entities_to_bio_tags(
    entities: list[Entity],
    offset_mapping: list[tuple[int, int]],
    word_ids: list[int | None],
) -> list[int]:
    """Map character spans to per-token label ids (-100 for subword continuations)."""
    labels = [-100] * len(offset_mapping)
    span_labels: list[str | None] = [None] * len(offset_mapping)
    sorted_entities = sorted(entities, key=lambda e: (e["start"], -(e["end"] - e["start"])))

    for ent in sorted_entities:
        start, end = int(ent["start"]), int(ent["end"])
        label = ent["label"]
        b_label = f"B-{label}"
        i_label = f"I-{label}"
        first = True
        for i, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start == tok_end == 0:
                continue
            if tok_end <= start or tok_start >= end:
                continue
            if span_labels[i] is not None:
                continue
            span_labels[i] = b_label if first else i_label
            first = False

    prev_word: int | None = None
    for i, word_id in enumerate(word_ids):
        if word_id is None:
            labels[i] = -100
            continue
        if word_id != prev_word:
            tag = span_labels[i] or "O"
            labels[i] = LABEL2ID[tag]
        else:
            labels[i] = -100
        prev_word = word_id

    return labels


def merge_non_overlapping(entities: list[Entity]) -> list[Entity]:
    """Drop overlapping spans (keep earlier / longer-first sorted)."""
    if not entities:
        return []
    out: list[Entity] = []
    for ent in sorted(entities, key=lambda e: (e["start"], -(e["end"] - e["start"]))):
        if out and ent["start"] < out[-1]["end"]:
            continue
        out.append(ent)
    return out


def align_phrases_to_text(
    text: str,
    phrases: list[tuple[str, str]],
) -> list[Entity]:
    """Align (phrase, label) pairs to non-overlapping spans in text."""
    entities: list[Entity] = []
    for phrase, label in phrases:
        span = find_span(text, phrase)
        if span is None:
            continue
        start, end = span
        entities.append(
            {
                "start": start,
                "end": end,
                "text": text[start:end],
                "label": label,
            }
        )
    return merge_non_overlapping(entities)
