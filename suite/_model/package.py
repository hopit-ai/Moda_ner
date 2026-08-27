"""Integrity verification and schema discovery for a local MODA General package."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema_contract import AttributeSchema, ContractError, SchemaRegistry


class PackageIntegrityError(RuntimeError):
    """Raised when package bytes do not match the immutable manifest."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ModaGeneralPackage:
    """Verified package metadata and schema registry."""

    root: Path
    package_id: str
    manifest: dict[str, Any]
    router: dict[str, Any]
    schemas: SchemaRegistry

    @classmethod
    def load(cls, root: str | Path, *, verify: bool = True) -> ModaGeneralPackage:
        package_root = Path(root)
        manifest_path = package_root / "manifest.json"
        commitment_path = package_root / "PACKAGE_COMMITMENT.json"
        router_path = package_root / "router.json"
        for path in (manifest_path, commitment_path, router_path):
            if not path.is_file():
                raise PackageIntegrityError(f"Package is missing {path.name}")
        manifest = _read_object(manifest_path)
        commitment = _read_object(commitment_path)
        router = _read_object(router_path)
        package_id = str(manifest.get("package_id") or "")
        if not package_id or commitment.get("package_id") != package_id:
            raise PackageIntegrityError("Package and commitment IDs do not match")
        if router.get("package_id") != package_id:
            raise PackageIntegrityError("Router package ID does not match manifest")
        if commitment.get("manifest_sha256") != sha256_file(manifest_path):
            raise PackageIntegrityError("Manifest SHA-256 does not match commitment")
        if commitment.get("router_sha256") != sha256_file(router_path):
            raise PackageIntegrityError("Router SHA-256 does not match commitment")
        _validate_router(router)
        if verify:
            _verify_artifacts(package_root, manifest)
        routes = router.get("routes")
        if not isinstance(routes, dict) or not routes:
            raise PackageIntegrityError("Router contains no schema routes")
        schemas: dict[str, AttributeSchema] = {}
        for schema_id, relative in routes.items():
            schema_path = package_root / str(relative) / "schema.json"
            if not schema_path.is_file():
                raise PackageIntegrityError(
                    f"Route {schema_id!r} is missing canonical schema.json"
                )
            try:
                schema = AttributeSchema.from_dict(_read_object(schema_path))
            except ContractError as exc:
                raise PackageIntegrityError(
                    f"Route {schema_id!r} has an invalid canonical schema"
                ) from exc
            if schema.schema_id != schema_id:
                raise PackageIntegrityError(
                    f"Router key {schema_id!r} does not match canonical schema ID"
                )
            schemas[schema_id] = schema
        return cls(
            root=package_root,
            package_id=package_id,
            manifest=manifest,
            router=router,
            schemas=SchemaRegistry(schemas),
        )


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise PackageIntegrityError(f"Expected a JSON object in {path}")
    return value


def _validate_router(router: dict[str, Any]) -> None:
    if router.get("router_input") != "caller_declared_schema_id_only":
        raise PackageIntegrityError("Router is not caller-schema-only")
    forbidden = (
        "dataset_identity_allowed",
        "image_content_routing_allowed",
        "test_score_routing_allowed",
    )
    if any(router.get(field) is not False for field in forbidden):
        raise PackageIntegrityError("Router permits a forbidden test-guided signal")
    if router.get("unknown_schema_policy") != "reject":
        raise PackageIntegrityError("Unknown schema policy must be reject")


def _verify_artifacts(root: Path, manifest: dict[str, Any]) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise PackageIntegrityError("Manifest contains no artifacts")
    relative_paths: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise PackageIntegrityError("Manifest contains a malformed artifact")
        relative = str(artifact.get("relative_path") or "")
        if not relative or relative in relative_paths:
            raise PackageIntegrityError("Manifest artifact paths are empty or duplicated")
        relative_paths.add(relative)
        path = root / relative
        if not path.is_file():
            raise PackageIntegrityError(f"Package artifact is missing: {relative}")
        if path.stat().st_size != int(artifact.get("bytes", -1)):
            raise PackageIntegrityError(f"Package artifact size changed: {relative}")
        if sha256_file(path) != artifact.get("sha256"):
            raise PackageIntegrityError(f"Package artifact SHA-256 changed: {relative}")
