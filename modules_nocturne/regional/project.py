"""Portable project files and bounded Regional metadata payloads."""

import base64
import binascii
import json
import os
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from modules_nocturne.regional.errors import PlanError
from modules_nocturne.regional.model import CURRENT_SCHEMA, RegionalGenerationPlan
from modules_nocturne.regional.serialization import MAX_JSON_BYTES, canonical_json, load_plan, plan_hash, plan_to_dict

PROJECT_SUFFIX = ".nocturne-region.json"
SIDECAR_SUFFIX = ".nocturne.json"
EMBEDDED_DATA_KEY = "Nocturne Regional Data"
MAX_DECOMPRESSED_METADATA_BYTES = MAX_JSON_BYTES


@dataclass(frozen=True, slots=True)
class MetadataBundle:
    fields: Mapping[str, str]
    sidecar_document: Mapping[str, Any]
    sidecar_required: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))
        object.__setattr__(self, "sidecar_document", MappingProxyType(dict(self.sidecar_document)))


@dataclass(frozen=True, slots=True)
class RestoredRegionalMetadata:
    plan: RegionalGenerationPlan
    requested_engine: str
    selected_engine: str | None
    adapter_id: str | None
    accepted_fallbacks: tuple[str, ...]


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def save_project(path: str | Path, plan: RegionalGenerationPlan) -> Path:
    destination = Path(path)
    if not destination.name.endswith(PROJECT_SUFFIX):
        raise PlanError("project.extension.invalid", "$", f"Project filename must end with {PROJECT_SUFFIX}")
    _atomic_write_text(destination, canonical_json(plan) + "\n")
    return destination


def load_project(path: str | Path) -> RegionalGenerationPlan:
    source = Path(path)
    if not source.name.endswith(PROJECT_SUFFIX):
        raise PlanError("project.extension.invalid", "$", f"Project filename must end with {PROJECT_SUFFIX}")
    if source.stat().st_size > MAX_JSON_BYTES:
        raise PlanError("project.input_too_large", "$", f"Project exceeds {MAX_JSON_BYTES} bytes")
    return load_plan(source.read_bytes())


def sidecar_path_for(image_path: str | Path) -> Path:
    source = Path(image_path)
    return source.with_name(source.name + SIDECAR_SUFFIX)


def _metadata_document(
    plan: RegionalGenerationPlan,
    *,
    selected_engine: str | None,
    adapter_id: str | None,
    accepted_fallbacks: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "metadata_schema": "nocturne.regional.metadata/v1",
        "plan_schema": plan.schema,
        "plan_hash": plan_hash(plan),
        "requested_engine": plan.engine.requested,
        "selected_engine": selected_engine,
        "adapter_id": adapter_id,
        "accepted_fallbacks": list(accepted_fallbacks),
        "plan": plan_to_dict(plan),
    }


def _metadata_json(document: Mapping[str, Any]) -> bytes:
    return json.dumps(dict(document), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _summary(plan: RegionalGenerationPlan, selected_engine: str | None) -> str:
    enabled = [region for region in plan.regions if region.enabled]
    names = ", ".join(region.name for region in enabled[:4])
    if len(enabled) > 4:
        names += f", +{len(enabled) - 4} more"
    engine = selected_engine or plan.engine.requested
    return f"Regional: {len(enabled)}/{len(plan.regions)} enabled; engine={engine}; regions={names or 'none'}"


def build_metadata(
    plan: RegionalGenerationPlan,
    *,
    selected_engine: str | None = None,
    adapter_id: str | None = None,
    accepted_fallbacks: tuple[str, ...] = (),
    embedded_limit_bytes: int = 256 * 1024,
) -> MetadataBundle:
    document = _metadata_document(
        plan,
        selected_engine=selected_engine,
        adapter_id=adapter_id,
        accepted_fallbacks=accepted_fallbacks,
    )
    encoded = base64.b64encode(zlib.compress(_metadata_json(document), level=9)).decode("ascii")
    sidecar_required = len(encoded) > embedded_limit_bytes
    fields = {
        "Nocturne Regional Schema": CURRENT_SCHEMA,
        "Nocturne Regional Hash": document["plan_hash"],
        "Nocturne Regional Summary": _summary(plan, selected_engine),
        "Nocturne Regional Requested Engine": plan.engine.requested,
        "Nocturne Regional Selected Engine": selected_engine or "",
        "Nocturne Regional Adapter": adapter_id or "",
    }
    if not sidecar_required:
        fields[EMBEDDED_DATA_KEY] = encoded
    return MetadataBundle(fields=fields, sidecar_document=document, sidecar_required=sidecar_required)


def save_sidecar(image_path: str | Path, bundle: MetadataBundle) -> Path:
    destination = sidecar_path_for(image_path)
    _atomic_write_text(destination, _metadata_json(bundle.sidecar_document).decode("utf-8") + "\n")
    return destination


def _decode_embedded(encoded: str) -> dict[str, Any]:
    if len(encoded) > MAX_JSON_BYTES * 2:
        raise PlanError("metadata.encoded_too_large", f"$.{EMBEDDED_DATA_KEY}", "Embedded metadata payload is too large")
    try:
        compressed = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise PlanError("metadata.invalid_base64", f"$.{EMBEDDED_DATA_KEY}", "Embedded metadata is not valid base64") from error

    decompressor = zlib.decompressobj()
    try:
        decoded = decompressor.decompress(compressed, MAX_DECOMPRESSED_METADATA_BYTES + 1)
        remaining = MAX_DECOMPRESSED_METADATA_BYTES + 1 - len(decoded)
        if remaining > 0:
            decoded += decompressor.flush(remaining)
    except zlib.error as error:
        raise PlanError("metadata.invalid_compression", f"$.{EMBEDDED_DATA_KEY}", "Embedded metadata is not valid zlib data") from error
    if len(decoded) > MAX_DECOMPRESSED_METADATA_BYTES or decompressor.unconsumed_tail:
        raise PlanError("metadata.decoded_too_large", f"$.{EMBEDDED_DATA_KEY}", "Decoded metadata exceeds the safety limit")

    try:
        document = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PlanError("metadata.invalid_json", f"$.{EMBEDDED_DATA_KEY}", "Embedded metadata is not valid JSON") from error
    if not isinstance(document, dict):
        raise PlanError("metadata.object_required", f"$.{EMBEDDED_DATA_KEY}", "Embedded metadata must contain an object")
    return document


def restore_metadata(
    fields: Mapping[str, str],
    *,
    sidecar_document: Mapping[str, Any] | None = None,
) -> RestoredRegionalMetadata:
    if EMBEDDED_DATA_KEY in fields:
        document = _decode_embedded(fields[EMBEDDED_DATA_KEY])
    elif sidecar_document is not None:
        document = dict(sidecar_document)
    else:
        raise PlanError("metadata.canonical_missing", "$", "Canonical Regional metadata is not available")

    if document.get("metadata_schema") != "nocturne.regional.metadata/v1":
        raise PlanError("metadata.schema.unsupported", "$.metadata_schema", "Unsupported Regional metadata schema")
    plan = load_plan(document.get("plan"))
    if document.get("plan_hash") != plan_hash(plan):
        raise PlanError("metadata.plan_hash_mismatch", "$.plan_hash", "Regional metadata plan hash does not match")

    fallbacks = document.get("accepted_fallbacks", [])
    if not isinstance(fallbacks, list) or not all(isinstance(item, str) for item in fallbacks):
        raise PlanError("metadata.fallbacks.invalid", "$.accepted_fallbacks", "Accepted fallbacks must be strings")

    requested_engine = document.get("requested_engine")
    selected_engine = document.get("selected_engine")
    adapter_id = document.get("adapter_id")
    if not isinstance(requested_engine, str):
        raise PlanError("metadata.requested_engine.invalid", "$.requested_engine", "Requested engine must be a string")
    if requested_engine != plan.engine.requested:
        raise PlanError(
            "metadata.requested_engine.mismatch",
            "$.requested_engine",
            "Requested engine does not match the canonical Regional plan",
        )
    if selected_engine is not None and not isinstance(selected_engine, str):
        raise PlanError("metadata.selected_engine.invalid", "$.selected_engine", "Selected engine must be a string or null")
    if adapter_id is not None and not isinstance(adapter_id, str):
        raise PlanError("metadata.adapter.invalid", "$.adapter_id", "Adapter ID must be a string or null")

    return RestoredRegionalMetadata(
        plan=plan,
        requested_engine=requested_engine,
        selected_engine=selected_engine,
        adapter_id=adapter_id,
        accepted_fallbacks=tuple(fallbacks),
    )
