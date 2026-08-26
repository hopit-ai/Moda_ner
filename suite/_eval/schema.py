"""Task prompts for Florence-2 fashion attribute extraction."""

from __future__ import annotations

# Fashion Florence recipe — natural-language prompt (NOT a Florence task token).
# Using <EXTRACT_ATTRIBUTES> causes the model to emit <loc_*> spatial tokens.
FASHION_FLORENCE_PROMPT = (
    "Inspect only the target fashion product in this image and extract visible attributes."
)

SINGLE_LABEL_ATTRIBUTES: tuple[str, ...] = (
    "master_category",
    "category",
    "sub_category",
    "silhouette",
    "fit",
    "hemline",
    "sleeve_length",
    "sleeve_shape",
    "neckline",
    "collar_presence",
    "collar_style",
    "lapel_style",
    "cuff_style",
    "waist_type",
    "styling_state",
    "heel_style",
    "toe_shape",
    "bag_size_style",
)

MULTI_LABEL_ATTRIBUTES: tuple[str, ...] = (
    "material",
    "color_palette_primary",
    "surface_treatment",
    "hardware_details",
    "decorative_elements",
    "pattern",
    "pocket_style",
    "fabric_construction",
    "closure_type",
    "opacity",
    "color_finish",
    "print_scale",
)

INDEXED_ATTRIBUTE_MAP = ",".join(
    f"{index:02d}:{field}"
    for index, field in enumerate(SINGLE_LABEL_ATTRIBUTES + MULTI_LABEL_ATTRIBUTES, start=1)
)

HOPIT_SCHEMA_SUFFIX = f"""
Return one compact JSON object. Omit hidden, uncertain, unknown, or non-applicable fields;
never emit confidence scores. Single-value fields map to a string:
{",".join(SINGLE_LABEL_ATTRIBUTES)}.
Multi-value fields map to arrays of every independently visible coexisting value:
{",".join(MULTI_LABEL_ATTRIBUTES)}.
Multiple values are simultaneous evidence, not alternative guesses or synonyms. Inspect only the
target item, not the person, background, inner layers, or styling accessories. Emit
collar_presence="absent" only when the full neck edge is visible, and emit collar_style only with
collar_presence="present"."""

ATTRIBUTE_SEQUENCE_SUFFIX = f"""
Return only this compact attribute sequence, with one visible field per line:
<attributes>
field=value
</attributes>
Use only these single-value fields: {",".join(SINGLE_LABEL_ATTRIBUTES)}.
Use only these multi-value fields: {",".join(MULTI_LABEL_ATTRIBUTES)}.
For a multi-value field, join simultaneous visible values with || on the same line. Omit hidden,
uncertain, unknown, and non-applicable fields. Never emit confidence scores. Never describe the
person, background, inner layers, or styling accessories. Emit collar_presence=absent only when
the full neck edge is visible, and collar_style only with collar_presence=present."""

ATTRIBUTE_JUDGMENT_SEQUENCE_SUFFIX = f"""
Return only this complete attribute-judgment sequence with all 30 fields exactly once:
<attribute_judgments>
field=value
</attribute_judgments>
Use <not_visible> when a field applies to the target product but cannot be determined from the
pixels. Use <not_applicable> only when the field is irrelevant to this product type. Use only
these single-value fields: {",".join(SINGLE_LABEL_ATTRIBUTES)}.
Use only these multi-value fields: {",".join(MULTI_LABEL_ATTRIBUTES)}.
For a visible multi-value field, join simultaneous values with ||. Inspect only the target item,
not the person, background, inner layers, or styling accessories. Never infer fibre composition
or hidden construction. Emit collar_style as a value only with collar_presence=present."""

INDEXED_ATTRIBUTE_JUDGMENT_SUFFIX = f"""
Return only this complete 30-line indexed judgment block:
<j>
01=value
02=value
...
30=value
</j>
Use ? when the indexed field applies but is not visible. Use - only when it is not applicable to
the target product type. The fixed index order is:
{INDEXED_ATTRIBUTE_MAP}.
For a visible multi-value field, join simultaneous values with ||. Inspect only the target item,
not the person, background, inner layers, or styling accessories. Never infer fibre composition
or hidden construction. Emit a collar-style value only when collar presence is present."""

TIER2_SUPPLEMENT_PROMPT = ""  # Retained for API compatibility; v3 always exposes all fields.


def get_training_prompt(output_format: str = "json") -> str:
    """Prompt for v3 sparse, confidence-free LoRA training and evaluation."""
    if output_format == "attribute_sequence":
        return FASHION_FLORENCE_PROMPT + ATTRIBUTE_SEQUENCE_SUFFIX
    if output_format == "attribute_judgment_sequence":
        return FASHION_FLORENCE_PROMPT + ATTRIBUTE_JUDGMENT_SEQUENCE_SUFFIX
    if output_format == "indexed_attribute_judgment_sequence":
        return FASHION_FLORENCE_PROMPT + INDEXED_ATTRIBUTE_JUDGMENT_SUFFIX
    if output_format != "json":
        raise ValueError(f"Unsupported Florence output format: {output_format}")
    return build_extraction_prompt(include_tier2=True, include_schema_hint=True)


def build_extraction_prompt(include_tier2: bool = False, include_schema_hint: bool = True) -> str:
    """Build inference prompt for attribute extraction."""
    prompt = FASHION_FLORENCE_PROMPT
    if include_schema_hint:
        prompt += HOPIT_SCHEMA_SUFFIX
    if include_tier2 and TIER2_SUPPLEMENT_PROMPT:
        prompt += TIER2_SUPPLEMENT_PROMPT
    return prompt
