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
from uuid import UUID

from modules_nocturne.regional.errors import PlanError
from modules_nocturne.regional.model import CURRENT_SCHEMA, RegionalGenerationPlan
from modules_nocturne.regional.seeds import ResolvedSeedBatch, ResolvedSeedPlan
from modules_nocturne.regional.serialization import MAX_JSON_BYTES, canonical_json, load_plan, plan_hash, plan_to_dict

PROJECT_SUFFIX = ".nocturne-region.json"
SIDECAR_SUFFIX = ".nocturne.json"
EMBEDDED_DATA_KEY = "Nocturne Regional Data"
MAX_DECOMPRESSED_METADATA_BYTES = MAX_JSON_BYTES
MAX_METADATA_IMAGE_BYTES = 128 * 1024 * 1024


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
    resolved_seeds: tuple[ResolvedSeedPlan, ...] = ()


@dataclass(frozen=True, slots=True)
class RestoredMetadataFile:
    metadata: RestoredRegionalMetadata
    source_kind: str


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
    resolved_seeds: tuple[ResolvedSeedPlan, ...],
) -> dict[str, Any]:
    return {
        "metadata_schema": "nocturne.regional.metadata/v1",
        "plan_schema": plan.schema,
        "plan_hash": plan_hash(plan),
        "requested_engine": plan.engine.requested,
        "selected_engine": selected_engine,
        "adapter_id": adapter_id,
        "accepted_fallbacks": list(accepted_fallbacks),
        "resolved_seeds": [
            {
                "requested_base_seed": item.requested_base_seed,
                "image_seed": item.image_seed,
                "batch_index": item.batch_index,
                "region_seeds": {str(region_id): seed for region_id, seed in sorted(item.region_seeds.items(), key=lambda pair: str(pair[0]))},
            }
            for item in resolved_seeds
        ],
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
    resolved_seeds: ResolvedSeedBatch | tuple[ResolvedSeedPlan, ...] = (),
    embedded_limit_bytes: int = 256 * 1024,
) -> MetadataBundle:
    seed_records = resolved_seeds.images if isinstance(resolved_seeds, ResolvedSeedBatch) else tuple(resolved_seeds)
    if len(seed_records) > 10_000 or not all(isinstance(item, ResolvedSeedPlan) for item in seed_records):
        raise PlanError("metadata.resolved_seeds.invalid", "$.resolved_seeds", "Resolved seeds must contain at most 10000 seed records")
    document = _metadata_document(
        plan,
        selected_engine=selected_engine,
        adapter_id=adapter_id,
        accepted_fallbacks=accepted_fallbacks,
        resolved_seeds=seed_records,
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

    raw_seed_records = document.get("resolved_seeds", [])
    if not isinstance(raw_seed_records, list) or len(raw_seed_records) > 10_000:
        raise PlanError("metadata.resolved_seeds.invalid", "$.resolved_seeds", "Resolved seeds must be a bounded array")
    expected_region_ids = {region.id for region in plan.regions}
    restored_seed_records = []
    for index, raw_record in enumerate(raw_seed_records):
        path = f"$.resolved_seeds[{index}]"
        if not isinstance(raw_record, dict):
            raise PlanError("metadata.resolved_seed.invalid", path, "Resolved seed record must be an object")
        integers = {}
        for name in ("requested_base_seed", "image_seed", "batch_index"):
            value = raw_record.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PlanError("metadata.resolved_seed.invalid", f"{path}.{name}", f"{name} must be a non-negative integer")
            if name != "batch_index" and value > (1 << 32) - 1:
                raise PlanError(
                    "metadata.resolved_seed.invalid",
                    f"{path}.{name}",
                    f"{name} must be a 32-bit unsigned integer",
                )
            integers[name] = value
        raw_region_seeds = raw_record.get("region_seeds")
        if not isinstance(raw_region_seeds, dict):
            raise PlanError("metadata.region_seeds.invalid", f"{path}.region_seeds", "Region seeds must be an object")
        region_seeds = {}
        for raw_region_id, seed in raw_region_seeds.items():
            try:
                region_id = UUID(raw_region_id)
            except (TypeError, ValueError) as error:
                raise PlanError("metadata.region_seed.id_invalid", f"{path}.region_seeds", "Region seed key must be a UUID") from error
            if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= (1 << 32) - 1:
                raise PlanError("metadata.region_seed.invalid", f"{path}.region_seeds.{raw_region_id}", "Region seed must be a 32-bit unsigned integer")
            region_seeds[region_id] = seed
        if set(region_seeds) != expected_region_ids:
            raise PlanError(
                "metadata.region_seeds.mismatch",
                f"{path}.region_seeds",
                "Resolved region seed IDs do not match the canonical plan",
            )
        restored_seed_records.append(ResolvedSeedPlan(region_seeds=region_seeds, **integers))

    return RestoredRegionalMetadata(
        plan=plan,
        requested_engine=requested_engine,
        selected_engine=selected_engine,
        adapter_id=adapter_id,
        accepted_fallbacks=tuple(fallbacks),
        resolved_seeds=tuple(restored_seed_records),
    )


def restore_metadata_file(path: str | Path) -> RestoredMetadataFile:
    """Restore canonical metadata from a bounded sidecar or PNG text fields."""

    source = Path(path)
    if not source.is_file():
        raise PlanError("metadata.input.missing", "$", "Choose a PNG image or Regional sidecar first")

    if source.suffix.lower() == ".json":
        if source.stat().st_size > MAX_JSON_BYTES:
            raise PlanError("metadata.input_too_large", "$", f"Metadata sidecar exceeds {MAX_JSON_BYTES} bytes")
        try:
            document = json.loads(source.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PlanError("metadata.invalid_json", "$", "Metadata sidecar is not valid JSON") from error
        if not isinstance(document, dict):
            raise PlanError("metadata.object_required", "$", "Metadata sidecar must contain an object")
        return RestoredMetadataFile(restore_metadata({}, sidecar_document=document), "sidecar")

    if source.suffix.lower() != ".png":
        raise PlanError("metadata.input.unsupported", "$", "Metadata restoration accepts PNG images or JSON sidecars")
    if source.stat().st_size > MAX_METADATA_IMAGE_BYTES:
        raise PlanError("metadata.image.too_large", "$", "PNG metadata source exceeds the 128 MiB safety limit")

    try:
        from PIL import Image, UnidentifiedImageError

        with Image.open(source) as image:
            if image.format != "PNG":
                raise PlanError("metadata.image.unsupported", "$", "Metadata restoration accepts PNG images")
            fields = {str(key): value for key, value in image.info.items() if isinstance(value, str)}
    except UnidentifiedImageError as error:
        raise PlanError("metadata.image.invalid", "$", "The selected file is not a readable PNG image") from error
    except OSError as error:
        raise PlanError("metadata.image.invalid", "$", "The selected PNG image could not be read") from error
    if sum(len(key) + len(value) for key, value in fields.items()) > MAX_JSON_BYTES * 2:
        raise PlanError("metadata.fields.too_large", "$", "PNG text metadata exceeds the safety limit")

    sidecar_document = None
    companion = sidecar_path_for(source)
    if EMBEDDED_DATA_KEY not in fields and companion.is_file():
        if companion.stat().st_size > MAX_JSON_BYTES:
            raise PlanError("metadata.input_too_large", "$", f"Metadata sidecar exceeds {MAX_JSON_BYTES} bytes")
        try:
            sidecar_document = json.loads(companion.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PlanError("metadata.invalid_json", "$", "Companion metadata sidecar is not valid JSON") from error
        if not isinstance(sidecar_document, dict):
            raise PlanError("metadata.object_required", "$", "Companion metadata sidecar must contain an object")

    if EMBEDDED_DATA_KEY not in fields and sidecar_document is None:
        if any(key.startswith("Nocturne Regional ") for key in fields):
            raise PlanError(
                "metadata.summary_only",
                "$",
                "This image contains only a Regional summary; exact geometry cannot be restored without its sidecar",
            )
        raise PlanError("metadata.canonical_missing", "$", "This PNG does not contain restorable Regional metadata")

    source_kind = "embedded PNG metadata" if EMBEDDED_DATA_KEY in fields else "companion sidecar"
    return RestoredMetadataFile(restore_metadata(fields, sidecar_document=sidecar_document), source_kind)
