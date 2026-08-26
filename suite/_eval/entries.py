"""Read sparse attributes off a benchmark or prediction row.

Vendored from the training package so that evaluation carries no dependency on
model or training code. Behaviour is unchanged.
"""

from __future__ import annotations

import json
from typing import Any


def attributes_from_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Return sparse confidence-free attributes from legacy or v4 release rows."""
    raw = entry.get("attributes_json", entry.get("attributes", {}))
    attrs = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(attrs, dict):
        raise ValueError("attributes/attributes_json must be a JSON object")
    supervised = entry.get("supervised_attributes")
    if isinstance(supervised, list):
        allowed = {str(field) for field in supervised}
        attrs = {key: value for key, value in attrs.items() if key in allowed}
    return attrs
