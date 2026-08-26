"""Conditional-attribute evaluation + label normalization.

Fashion attributes are mostly *conditional*: ``sleeve_shape`` only means something
on a sleeved garment, ``waist_type`` only on a waisted one. A plain macro-F1 over
rare classes conflates three different failures:

  1. the model couldn't decide the attribute *applies* (applicability),
  2. the model picked the wrong *value* when it does apply (discrimination), and
  3. the predicted value is a synonym/surface-variant of the gold (measurement).

This module separates them and normalizes label vocabulary first, so the reported
numbers reflect real ability rather than scoring artifacts.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any

from .metrics import fuzzy_match

# Field-specific surface-form synonyms -> canonical value (applied after generic
# cleanup). Keep small and high-precision.
_SYNONYMS: dict[str, dict[str, str]] = {
    "color_palette_primary": {
        "navy blue": "navy",
        "dark blue": "navy",
        "grey": "gray",
        "off white": "white",
        "off-white": "white",
        "cream": "beige",
    },
    "neckline": {
        "vneck": "v-neck",
        "v neck": "v-neck",
        "round neck": "round",
        "crew neck": "crew",
        "scoop neck": "scoop",
    },
    "collar_presence": {
        "collared": "present",
        "with collar": "present",
        "collarless": "absent",
        "no collar": "absent",
        "without collar": "absent",
    },
    "pattern": {
        "solid": "plain",
        "floral print": "floral",
        "striped": "stripe",
        "dotted": "dot",
        "polka dot": "dot",
    },
}

# Trailing type-tag words to strip per field (e.g. "set-in sleeve" -> "set-in").
_TRAILING_TAGS: dict[str, tuple[str, ...]] = {
    "sleeve_shape": ("sleeve",),
    "sleeve_length": ("length",),
    "hemline": ("length",),
    "collar_style": ("collar",),
}


# Field-name aliases the model invents under clean-JSON decoding.
_KEY_ALIASES: dict[str, str] = {
    "sub-category": "sub_category",
    "subcategory": "sub_category",
    "closure_style": "closure_type",
    "closure": "closure_type",
    "waist_length": "waist_type",
    "colour_palette_primary": "color_palette_primary",
    "color": "color_palette_primary",
    "sleeve-shape": "sleeve_shape",
    "sleeve-length": "sleeve_length",
}


def normalize_key(key: str) -> str:
    """Canonicalize an attribute field name (snake_case, lowercase, de-alias)."""
    k = key.strip().lower().replace(" ", "").replace("-", "_")
    while "__" in k:
        k = k.replace("__", "_")
    return _KEY_ALIASES.get(k, k)


def normalize_keys(attrs: dict) -> dict:
    """Apply key normalization to an attribute dict, keeping first-seen value."""
    out: dict = {}
    for k, v in attrs.items():
        nk = normalize_key(k)
        if nk not in out:
            out[nk] = v
    return out


# Closed-vocabulary fields that should be snapped to training enums.
# Open-vocab and multi-value fields are intentionally excluded.
_SNAP_FIELDS: frozenset[str] = frozenset(
    {
        "master_category",
        "silhouette",
        "hemline",
        "neckline",
        "collar_presence",
        "collar_style",
        "waist_type",
        "closure_type",
        "sleeve_length",
        "sleeve_shape",
        "pattern",
    }
)

# Module-level enum cache: field -> set of normalized enum values.
_ENUM_CACHE: dict[str, set[str]] | None = None
_ENUM_CACHE_PATH: str | None = None


def _load_enums(path: str) -> dict[str, set[str]]:
    """Load enum vocab file and return field -> set-of-normalized-values."""
    with open(path) as f:
        raw = json.load(f)
    return {field: set(raw[field]) for field in raw}


def get_enums() -> dict[str, set[str]] | None:
    """Return enum cache, loading lazily from FASHION_ENUM_PATH if available."""
    global _ENUM_CACHE, _ENUM_CACHE_PATH
    env_path = os.environ.get("FASHION_ENUM_PATH", "data/fashion_vision_enums.json")
    if env_path == "/dev/null":
        return None
    if _ENUM_CACHE is not None and _ENUM_CACHE_PATH == env_path:
        return _ENUM_CACHE
    if not os.path.exists(env_path):
        return None
    _ENUM_CACHE = _load_enums(env_path)
    _ENUM_CACHE_PATH = env_path
    return _ENUM_CACHE


def snap_to_enum(
    field: str,
    value: str,
    enums: dict[str, set[str]],
    threshold: float = 0.80,
) -> str:
    """Normalize value then snap to nearest enum member if score ≥ threshold.

    Returns the snapped enum string (already normalized) on a hit, or the
    normalized input value unchanged on a miss.
    """
    if field not in _SNAP_FIELDS or field not in enums:
        return value
    norm = normalize_value(field, value)
    if norm in enums[field]:
        return norm
    best_score = 0.0
    best = norm
    for candidate in enums[field]:
        score = SequenceMatcher(None, norm, candidate).ratio()
        if score > best_score:
            best_score = score
            best = candidate
    return best if best_score >= threshold else norm


def normalize_value(field: str, value: str) -> str:
    """Normalize a single attribute value for fair matching."""
    v = value.lower().strip()
    v = re.sub(r"\(.*?\)", "", v)  # drop parentheticals: "round (neck)" -> "round"
    v = re.sub(r"[^a-z0-9 \-]", "", v)  # drop stray punctuation
    v = re.sub(r"\s+", " ", v).strip(" -")
    for tag in _TRAILING_TAGS.get(field, ()):  # "set-in sleeve" -> "set-in"
        if v.endswith(" " + tag):
            v = v[: -(len(tag) + 1)].strip()
    v = _SYNONYMS.get(field, {}).get(v, v)
    return v


def _values(field: str, val: Any) -> list[str]:
    """Extract normalized string values from an attribute entry."""
    out: list[str] = []
    if isinstance(val, dict):
        v = val.get("value")
        if v is not None:
            out.append(str(v))
    elif isinstance(val, list):
        for item in val:
            if isinstance(item, dict) and item.get("value") is not None:
                out.append(str(item["value"]))
            elif isinstance(item, str):
                out.append(item)
    elif isinstance(val, str):
        out.append(val)
    return [normalize_value(field, x) for x in out if x]


def conditional_report(
    preds: list[dict[str, Any]],
    golds: list[dict[str, Any]],
    fuzzy_threshold: float = 0.85,
) -> dict[str, dict[str, float]]:
    """Per-attribute applicability F1 + value-accuracy-when-applicable.

    - ``applicability_f1``: treat "field present" as a binary detection task
      (does the model emit the field exactly when the gold has it?).
    - ``value_acc``: among samples where the gold HAS the field, fraction where a
      predicted value matches (normalized, fuzzy). This is pure discrimination,
      independent of how rare the attribute is.
    - ``support``: number of gold-applicable samples.
    """
    stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"app_tp": 0, "app_fp": 0, "app_fn": 0, "val_correct": 0, "support": 0}
    )
    fields: set[str] = set()
    for p, g in zip(preds, golds):
        fields.update(p.keys())
        fields.update(g.keys())

    for p, g in zip(preds, golds):
        for field in fields:
            pv = _values(field, p.get(field))
            gv = _values(field, g.get(field))
            gold_has = len(gv) > 0
            pred_has = len(pv) > 0

            if gold_has and pred_has:
                stats[field]["app_tp"] += 1
            elif pred_has and not gold_has:
                stats[field]["app_fp"] += 1
            elif gold_has and not pred_has:
                stats[field]["app_fn"] += 1

            if gold_has:
                stats[field]["support"] += 1
                matched = any(fuzzy_match(a, b, fuzzy_threshold) for a in pv for b in gv)
                if matched:
                    stats[field]["val_correct"] += 1

    report: dict[str, dict[str, float]] = {}
    for field, s in stats.items():
        tp, fp, fn = s["app_tp"], s["app_fp"], s["app_fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        app_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        support = int(s["support"])
        val_acc = s["val_correct"] / support if support else 0.0
        report[field] = {
            "applicability_f1": round(app_f1, 4),
            "value_acc": round(val_acc, 4),
            "support": support,
        }
    return report
