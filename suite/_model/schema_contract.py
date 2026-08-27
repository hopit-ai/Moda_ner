"""Schema and sparse-output contracts for MODA General inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class ContractError(ValueError):
    """Raised when a schema or prediction violates the frozen contract."""


@dataclass(frozen=True)
class FieldSpec:
    """Closed vocabulary and cardinality for one output field."""

    name: str
    values: tuple[str, ...]
    multi_value: bool = False
    not_applicable_values: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, name: str, value: Mapping[str, Any]) -> FieldSpec:
        values = tuple(str(item) for item in value.get("values", ()))
        if not values or len(values) != len(set(values)):
            raise ContractError(f"Field {name!r} has an empty or duplicate vocabulary")
        not_applicable = tuple(
            str(item) for item in value.get("not_applicable_values", ())
        )
        if not set(not_applicable) <= set(values):
            raise ContractError(
                f"Field {name!r} has unknown not-applicable vocabulary values"
            )
        return cls(
            name=name,
            values=values,
            multi_value=bool(value.get("multi_value", False)),
            not_applicable_values=not_applicable,
        )


@dataclass(frozen=True)
class AttributeSchema:
    """Caller-declared output schema for one supported image contract."""

    schema_id: str
    input_contract: str
    fields: dict[str, FieldSpec]
    sparse_output: bool = True

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AttributeSchema:
        schema_id = str(value.get("schema_id") or "")
        input_contract = str(value.get("input_contract") or "")
        raw_fields = value.get("fields")
        if not schema_id or not input_contract or not isinstance(raw_fields, Mapping):
            raise ContractError("Schema ID, input contract, and fields are required")
        fields = {
            str(name): FieldSpec.from_dict(str(name), specification)
            for name, specification in raw_fields.items()
            if isinstance(specification, Mapping)
        }
        if len(fields) != len(raw_fields) or not fields:
            raise ContractError(f"Schema {schema_id!r} contains malformed fields")
        return cls(
            schema_id=schema_id,
            input_contract=input_contract,
            fields=fields,
            sparse_output=bool(value.get("sparse_output", True)),
        )

    def normalize(
        self,
        attributes: Mapping[str, Any],
        *,
        include_not_applicable: bool = False,
    ) -> tuple[dict[str, list[str]], tuple[str, ...]]:
        """Validate and normalize one route prediction into list-valued sparse output."""
        unknown = set(attributes) - set(self.fields)
        if unknown:
            raise ContractError(
                f"Schema {self.schema_id!r} received unknown fields: {sorted(unknown)}"
            )
        normalized: dict[str, list[str]] = {}
        not_applicable_fields: list[str] = []
        for field_name, raw in attributes.items():
            field = self.fields[field_name]
            values = _string_values(raw, field_name)
            if not values:
                continue
            unknown_values = set(values) - set(field.values)
            if unknown_values:
                raise ContractError(
                    f"Field {field_name!r} received unknown values: {sorted(unknown_values)}"
                )
            na_values = [value for value in values if value in field.not_applicable_values]
            visible_values = [
                value for value in values if value not in field.not_applicable_values
            ]
            if na_values and visible_values:
                raise ContractError(
                    f"Field {field_name!r} mixes visible and not-applicable values"
                )
            if na_values:
                not_applicable_fields.append(field_name)
                if include_not_applicable:
                    normalized[field_name] = na_values
                continue
            if not field.multi_value and len(visible_values) != 1:
                raise ContractError(f"Field {field_name!r} accepts exactly one value")
            normalized[field_name] = visible_values
        return normalized, tuple(sorted(not_applicable_fields))


def _string_values(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw = list(value)
    else:
        raise ContractError(f"Field {field!r} must contain a string or string list")
    if not all(isinstance(item, str) for item in raw):
        raise ContractError(f"Field {field!r} contains a non-string value")
    stripped = [item.strip() for item in raw if item.strip()]
    if len(stripped) != len(set(stripped)):
        raise ContractError(f"Field {field!r} contains duplicate values")
    return stripped


class SchemaRegistry:
    """Immutable registry keyed only by caller-declared schema ID."""

    def __init__(self, schemas: Mapping[str, AttributeSchema]) -> None:
        self._schemas = dict(schemas)
        if not self._schemas or any(key != value.schema_id for key, value in self._schemas.items()):
            raise ContractError("Schema registry keys must match non-empty schema IDs")

    @property
    def schema_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._schemas))

    def require(self, schema_id: str) -> AttributeSchema:
        try:
            return self._schemas[schema_id]
        except KeyError as exc:
            raise ContractError(
                f"Unknown schema {schema_id!r}; expected one of {self.schema_ids}"
            ) from exc


@dataclass(frozen=True)
class ModaGeneralPrediction:
    """Stable, confidence-free public response contract."""

    package_id: str
    schema_id: str
    attributes: dict[str, list[str]]
    not_applicable_fields: tuple[str, ...] = ()
    abstained_fields: tuple[str, ...] = ()

    def to_dict(self, *, include_abstentions: bool = False) -> dict[str, Any]:
        """Serialize a prediction, optionally making sparse abstentions explicit.

        The default remains byte-compatible with the original sparse contract. Production
        callers can opt into ``abstained_fields`` so an unobservable/low-confidence field is not
        confused with a negative label or a schema-defined not-applicable state.
        """
        result: dict[str, Any] = {
            "package_id": self.package_id,
            "schema_id": self.schema_id,
            "attributes": self.attributes,
        }
        if self.not_applicable_fields:
            result["not_applicable_fields"] = list(self.not_applicable_fields)
        if include_abstentions and self.abstained_fields:
            result["abstained_fields"] = list(self.abstained_fields)
        return result
