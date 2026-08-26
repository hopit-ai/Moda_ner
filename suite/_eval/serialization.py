"""Robust wire format for Florence fashion-attribute generation.

Florence-2-base learns visual values quickly but is unreliable at balancing
nested JSON punctuation. The model therefore emits a simple line protocol and
the application deterministically converts it to canonical sparse JSON.
"""

from __future__ import annotations

from typing import Any

from .schema import (
    MULTI_LABEL_ATTRIBUTES,
    SINGLE_LABEL_ATTRIBUTES,
)

START_MARKER = "<attributes>"
END_MARKER = "</attributes>"
JUDGMENT_START_MARKER = "<attribute_judgments>"
JUDGMENT_END_MARKER = "</attribute_judgments>"
INDEXED_JUDGMENT_START_MARKER = "<j>"
INDEXED_JUDGMENT_END_MARKER = "</j>"
NOT_VISIBLE_VALUE = "<not_visible>"
NOT_APPLICABLE_VALUE = "<not_applicable>"
INDEXED_NOT_VISIBLE_VALUE = "?"
INDEXED_NOT_APPLICABLE_VALUE = "-"
MULTI_VALUE_SEPARATOR = " || "
FIELD_ORDER = SINGLE_LABEL_ATTRIBUTES + MULTI_LABEL_ATTRIBUTES
KNOWN_FIELDS = frozenset(FIELD_ORDER)
MULTI_FIELDS = frozenset(MULTI_LABEL_ATTRIBUTES)
FIELD_ALIASES = {
    # Bounded decoder variants observed in the A10 pilot. These are key repairs,
    # never value guesses, and therefore cannot manufacture an attribute value.
    "sleeveless_shape": "sleeve_shape",
    "swaist_type": "waist_type",
}


def _clean_value(value: Any) -> str:
    cleaned = " ".join(str(value).replace("\r", " ").replace("\n", " ").split())
    return cleaned.replace(MULTI_VALUE_SEPARATOR, " ").strip()


def serialize_attribute_sequence(attributes: dict[str, Any]) -> str:
    """Serialize sparse attributes into the canonical line protocol."""
    lines = [START_MARKER]
    for field in FIELD_ORDER:
        if field not in attributes:
            continue
        value = attributes[field]
        if field in MULTI_FIELDS:
            raw_values = value if isinstance(value, list) else [value]
            values = [_clean_value(item) for item in raw_values]
            values = [item for item in values if item]
            if values:
                lines.append(f"{field}={MULTI_VALUE_SEPARATOR.join(dict.fromkeys(values))}")
        else:
            cleaned = _clean_value(value)
            if cleaned:
                lines.append(f"{field}={cleaned}")
    lines.append(END_MARKER)
    if len(lines) == 2:
        raise ValueError("Cannot serialize an empty attribute target")
    return "\n".join(lines)


def serialize_attribute_judgment_sequence(
    attributes: dict[str, Any],
    attribute_states: dict[str, str],
) -> str:
    """Serialize all fields with explicit visibility/applicability states."""
    missing = KNOWN_FIELDS - set(attribute_states)
    extra = set(attribute_states) - KNOWN_FIELDS
    if missing or extra:
        raise ValueError(
            f"Complete judgment states mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    lines = [JUDGMENT_START_MARKER]
    for field in FIELD_ORDER:
        state = attribute_states[field]
        if state == "value":
            if field not in attributes:
                raise ValueError(f"State is value but {field} has no attribute value")
            value = attributes[field]
            if field in MULTI_FIELDS:
                raw_values = value if isinstance(value, list) else [value]
                values = [_clean_value(item) for item in raw_values]
                values = [item for item in values if item]
                if not values:
                    raise ValueError(f"State is value but {field} has no usable value")
                rendered = MULTI_VALUE_SEPARATOR.join(dict.fromkeys(values))
            else:
                rendered = _clean_value(value)
                if not rendered:
                    raise ValueError(f"State is value but {field} has no usable value")
        elif state == "not_visible":
            rendered = NOT_VISIBLE_VALUE
        elif state == "not_applicable":
            rendered = NOT_APPLICABLE_VALUE
        else:
            raise ValueError(f"Unsupported judgment state for {field}: {state!r}")
        lines.append(f"{field}={rendered}")
    lines.append(JUDGMENT_END_MARKER)
    return "\n".join(lines)


def serialize_indexed_attribute_judgment_sequence(
    attributes: dict[str, Any],
    attribute_states: dict[str, str],
) -> str:
    """Serialize the complete contract with compact, recoverable field indexes."""
    missing = KNOWN_FIELDS - set(attribute_states)
    extra = set(attribute_states) - KNOWN_FIELDS
    if missing or extra:
        raise ValueError(
            f"Complete judgment states mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    lines = [INDEXED_JUDGMENT_START_MARKER]
    for index, field in enumerate(FIELD_ORDER, start=1):
        state = attribute_states[field]
        if state == "value":
            if field not in attributes:
                raise ValueError(f"State is value but {field} has no attribute value")
            value = attributes[field]
            if field in MULTI_FIELDS:
                raw_values = value if isinstance(value, list) else [value]
                values = [_clean_value(item) for item in raw_values]
                values = [item for item in values if item]
                if not values:
                    raise ValueError(f"State is value but {field} has no usable value")
                rendered = MULTI_VALUE_SEPARATOR.join(dict.fromkeys(values))
            else:
                rendered = _clean_value(value)
                if not rendered:
                    raise ValueError(f"State is value but {field} has no usable value")
        elif state == "not_visible":
            rendered = INDEXED_NOT_VISIBLE_VALUE
        elif state == "not_applicable":
            rendered = INDEXED_NOT_APPLICABLE_VALUE
        else:
            raise ValueError(f"Unsupported judgment state for {field}: {state!r}")
        lines.append(f"{index:02d}={rendered}")
    lines.append(INDEXED_JUDGMENT_END_MARKER)
    return "\n".join(lines)


def parse_indexed_attribute_judgment_sequence(
    text: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Parse compact indexed judgments without allowing row shifts."""
    candidate = text.strip()
    start = candidate.find(INDEXED_JUDGMENT_START_MARKER)
    if start >= 0:
        candidate = candidate[start + len(INDEXED_JUDGMENT_START_MARKER) :]
    end = candidate.find(INDEXED_JUDGMENT_END_MARKER)
    if end >= 0:
        candidate = candidate[:end]

    attributes: dict[str, Any] = {}
    states: dict[str, str] = {}
    for raw_line in candidate.splitlines():
        line = raw_line.strip()
        if "=" not in line:
            continue
        raw_index, raw_value = line.split("=", 1)
        try:
            index = int(raw_index.strip())
        except ValueError:
            continue
        if index < 1 or index > len(FIELD_ORDER):
            continue
        field = FIELD_ORDER[index - 1]
        if field in states:
            continue
        cleaned = _clean_value(raw_value)
        if cleaned == INDEXED_NOT_VISIBLE_VALUE:
            states[field] = "not_visible"
        elif cleaned == INDEXED_NOT_APPLICABLE_VALUE:
            states[field] = "not_applicable"
        elif cleaned and cleaned.lower() != "unknown":
            states[field] = "value"
            if field in MULTI_FIELDS:
                values = [_clean_value(item) for item in raw_value.split("||")]
                values = [item for item in values if item]
                if values:
                    attributes[field] = list(dict.fromkeys(values))
            else:
                attributes[field] = cleaned
    return attributes, states


def parse_attribute_judgment_sequence(
    text: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Parse a complete sequence into sparse values and explicit states."""
    if INDEXED_JUDGMENT_START_MARKER in text:
        return parse_indexed_attribute_judgment_sequence(text)
    candidate = text.strip()
    start = candidate.find(JUDGMENT_START_MARKER)
    if start >= 0:
        candidate = candidate[start + len(JUDGMENT_START_MARKER) :]
    end = candidate.find(JUDGMENT_END_MARKER)
    if end >= 0:
        candidate = candidate[:end]

    attributes: dict[str, Any] = {}
    states: dict[str, str] = {}
    for raw_line in candidate.splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        field, raw_value = line.split("=", 1)
        field = "".join(field.split())
        field = FIELD_ALIASES.get(field, field)
        if field not in KNOWN_FIELDS or field in states:
            continue
        cleaned = _clean_value(raw_value)
        if cleaned == NOT_VISIBLE_VALUE:
            states[field] = "not_visible"
        elif cleaned == NOT_APPLICABLE_VALUE:
            states[field] = "not_applicable"
        elif cleaned and cleaned.lower() != "unknown":
            states[field] = "value"
            if field in MULTI_FIELDS:
                values = [_clean_value(item) for item in raw_value.split("||")]
                values = [item for item in values if item]
                if values:
                    attributes[field] = list(dict.fromkeys(values))
            else:
                attributes[field] = cleaned
    return attributes, states


def parse_attribute_sequence(text: str) -> dict[str, Any]:
    """Parse a generated line protocol into sparse canonical attributes."""
    if JUDGMENT_START_MARKER in text or INDEXED_JUDGMENT_START_MARKER in text:
        attributes, _states = parse_attribute_judgment_sequence(text)
        return attributes
    candidate = text.strip()
    start = candidate.find(START_MARKER)
    if start >= 0:
        candidate = candidate[start + len(START_MARKER) :]
    end = candidate.find(END_MARKER)
    if end >= 0:
        candidate = candidate[:end]

    attributes: dict[str, Any] = {}
    for raw_line in candidate.splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        field, raw_value = line.split("=", 1)
        field = "".join(field.split())
        field = FIELD_ALIASES.get(field, field)
        if field not in KNOWN_FIELDS or field in attributes:
            continue
        if field in MULTI_FIELDS:
            values = [_clean_value(item) for item in raw_value.split("||")]
            values = [item for item in values if item and item.lower() != "unknown"]
            if values:
                attributes[field] = list(dict.fromkeys(values))
        else:
            value = _clean_value(raw_value)
            if value and value.lower() != "unknown":
                attributes[field] = value
    if "collar_style" in attributes and attributes.get("collar_presence") != "present":
        attributes.pop("collar_style")
    return attributes
