"""Bounded parsing, canonical serialisation, hashing and migrations."""

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable
from uuid import UUID

from modules_nocturne.regional.errors import PlanError
from modules_nocturne.regional.model import (
    CURRENT_SCHEMA,
    Canvas,
    Composition,
    EngineSelection,
    GlobalPrompt,
    GuidanceSchedule,
    OverlapPolicy,
    PassPolicy,
    Point,
    PolygonGeometry,
    RasterMaskGeometry,
    RectGeometry,
    Region,
    RegionalGenerationPlan,
    SeedMode,
    SeedPolicy,
    UncoveredPolicy,
)

MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 100_000
MAX_STRING_LENGTH = 8 * 1024 * 1024
TRANSIENT_KEYS = frozenset({"ui", "ui_state", "_ui", "selected_region_id", "preview"})


class MigrationRegistry:
    def __init__(self, current_schema: str):
        self.current_schema = current_schema
        self._migrations: dict[str, tuple[str, Callable[[dict[str, Any]], dict[str, Any]]]] = {}

    def register(self, source: str, target: str, migration: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        if source in self._migrations:
            raise ValueError(f"Migration already registered for {source}")
        self._migrations[source] = (target, migration)

    def migrate(self, document: dict[str, Any]) -> dict[str, Any]:
        schema = document.get("schema")
        visited: set[str] = set()

        while schema != self.current_schema:
            if not isinstance(schema, str):
                raise PlanError("schema.missing", "$.schema", "A string schema identifier is required")
            if schema in visited:
                raise PlanError("schema.migration_cycle", "$.schema", f"Migration cycle detected at {schema}")
            visited.add(schema)

            migration_entry = self._migrations.get(schema)
            if migration_entry is None:
                raise PlanError(
                    "schema.unsupported_version",
                    "$.schema",
                    f"Unsupported Regional schema {schema!r}; this build supports {self.current_schema!r}",
                )

            target, migration = migration_entry
            document = migration(dict(document))
            document["schema"] = target
            schema = target

        return document


MIGRATIONS = MigrationRegistry(CURRENT_SCHEMA)


def _reject_constant(value: str) -> None:
    raise PlanError("json.non_finite_number", "$", f"Non-finite JSON number {value!r} is not supported")


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PlanError("json.duplicate_key", "$", f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


def _bounded_json(value: Any, *, path: str = "$", depth: int = 0, counter: list[int] | None = None) -> None:
    if counter is None:
        counter = [0]
    counter[0] += 1
    if counter[0] > MAX_JSON_NODES:
        raise PlanError("json.too_many_values", path, f"JSON exceeds {MAX_JSON_NODES} values")
    if depth > MAX_JSON_DEPTH:
        raise PlanError("json.too_deep", path, f"JSON nesting exceeds {MAX_JSON_DEPTH} levels")
    if isinstance(value, str):
        if len(value) > MAX_STRING_LENGTH:
            raise PlanError("json.string_too_large", path, f"String exceeds {MAX_STRING_LENGTH} characters")
        return
    if value is None or isinstance(value, (bool, int, float)):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _bounded_json(item, path=f"{path}.{key}", depth=depth + 1, counter=counter)
        return
    if isinstance(value, Sequence):
        for index, item in enumerate(value):
            _bounded_json(item, path=f"{path}[{index}]", depth=depth + 1, counter=counter)
        return
    raise PlanError("json.unsupported_value", path, f"Unsupported value type {type(value).__name__}")


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PlanError("type.object_required", path, "Expected a JSON object")
    return dict(value)


def _sequence(value: Any, path: str) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        raise PlanError("type.array_required", path, "Expected a JSON array")
    return list(value)


def _string(value: Any, path: str, default: str | None = None) -> str:
    if value is None and default is not None:
        return default
    if not isinstance(value, str):
        raise PlanError("type.string_required", path, "Expected a string")
    return value


def _boolean(value: Any, path: str, default: bool | None = None) -> bool:
    if value is None and default is not None:
        return default
    if not isinstance(value, bool):
        raise PlanError("type.boolean_required", path, "Expected a boolean")
    return value


def _integer(value: Any, path: str, default: int | None = None) -> int:
    if value is None and default is not None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanError("type.integer_required", path, "Expected an integer")
    return value


def _number(value: Any, path: str, default: float | None = None) -> float:
    if value is None and default is not None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlanError("type.number_required", path, "Expected a number")
    result = float(value)
    if not math.isfinite(result):
        raise PlanError("number.not_finite", path, "Expected a finite number")
    return result


def _extra(document: Mapping[str, Any], known: set[str]) -> dict[str, Any]:
    return {key: value for key, value in document.items() if key not in known}


def _enum(enum_type, value: Any, path: str, default):
    if value is None:
        return default
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        choices = ", ".join(item.value for item in enum_type)
        raise PlanError("enum.invalid", path, f"Expected one of: {choices}") from error


def _parse_geometry(value: Any, path: str):
    document = _mapping(value, path)
    geometry_type = _string(document.get("type"), f"{path}.type")

    if geometry_type == "rect":
        known = {"type", "x", "y", "width", "height"}
        return RectGeometry(
            x=_number(document.get("x"), f"{path}.x"),
            y=_number(document.get("y"), f"{path}.y"),
            width=_number(document.get("width"), f"{path}.width"),
            height=_number(document.get("height"), f"{path}.height"),
            extra=_extra(document, known),
        )

    if geometry_type == "polygon":
        known = {"type", "points"}
        points = []
        for index, raw_point in enumerate(_sequence(document.get("points"), f"{path}.points")):
            point_path = f"{path}.points[{index}]"
            point = _mapping(raw_point, point_path)
            points.append(Point(x=_number(point.get("x"), f"{point_path}.x"), y=_number(point.get("y"), f"{point_path}.y")))
        return PolygonGeometry(points=tuple(points), extra=_extra(document, known))

    if geometry_type in {"raster_mask", "raster-alpha-mask"}:
        known = {"type", "png_base64", "data", "width", "height", "sha256"}
        encoded = document.get("png_base64", document.get("data"))
        sha256 = document.get("sha256")
        return RasterMaskGeometry(
            png_base64=_string(encoded, f"{path}.png_base64"),
            width=_integer(document.get("width"), f"{path}.width"),
            height=_integer(document.get("height"), f"{path}.height"),
            sha256=None if sha256 is None else _string(sha256, f"{path}.sha256"),
            extra=_extra(document, known),
        )

    raise PlanError("geometry.unsupported_type", f"{path}.type", f"Unsupported geometry type {geometry_type!r}")


def _parse_region(value: Any, index: int) -> Region:
    path = f"$.regions[{index}]"
    document = _mapping(value, path)
    known = {
        "id",
        "name",
        "enabled",
        "locked",
        "hidden",
        "positive",
        "negative",
        "inherit_global_positive",
        "inherit_global_negative",
        "geometry",
        "weight",
        "priority",
        "feather_px",
        "guidance",
        "seed",
    }

    raw_id = _string(document.get("id"), f"{path}.id")
    try:
        region_id = UUID(raw_id)
    except ValueError as error:
        raise PlanError("region.id.invalid", f"{path}.id", "Expected a valid UUID") from error

    guidance_document = _mapping(document.get("guidance", {}), f"{path}.guidance")
    seed_document = _mapping(document.get("seed", {}), f"{path}.seed")
    seed_value = seed_document.get("value")

    return Region(
        id=region_id,
        name=_string(document.get("name"), f"{path}.name"),
        enabled=_boolean(document.get("enabled"), f"{path}.enabled", True),
        locked=_boolean(document.get("locked"), f"{path}.locked", False),
        hidden=_boolean(document.get("hidden"), f"{path}.hidden", False),
        positive=_string(document.get("positive"), f"{path}.positive", ""),
        negative=_string(document.get("negative"), f"{path}.negative", ""),
        inherit_global_positive=_boolean(document.get("inherit_global_positive"), f"{path}.inherit_global_positive", True),
        inherit_global_negative=_boolean(document.get("inherit_global_negative"), f"{path}.inherit_global_negative", True),
        geometry=_parse_geometry(document.get("geometry"), f"{path}.geometry"),
        weight=_number(document.get("weight"), f"{path}.weight", 1.0),
        priority=_integer(document.get("priority"), f"{path}.priority", 0),
        feather_px=_number(document.get("feather_px"), f"{path}.feather_px", 0.0),
        guidance=GuidanceSchedule(
            start=_number(guidance_document.get("start"), f"{path}.guidance.start", 0.0),
            end=_number(guidance_document.get("end"), f"{path}.guidance.end", 1.0),
            extra=_extra(guidance_document, {"start", "end"}),
        ),
        seed=SeedPolicy(
            mode=_enum(SeedMode, seed_document.get("mode"), f"{path}.seed.mode", SeedMode.DERIVED),
            offset=_integer(seed_document.get("offset"), f"{path}.seed.offset", 0),
            value=None if seed_value is None else _integer(seed_value, f"{path}.seed.value"),
            extra=_extra(seed_document, {"mode", "offset", "value"}),
        ),
        extra=_extra(document, known),
    )


def plan_from_dict(value: Mapping[str, Any]) -> RegionalGenerationPlan:
    document = MIGRATIONS.migrate(dict(value))
    _bounded_json(document)

    canvas_document = _mapping(document.get("canvas", {}), "$.canvas")
    global_document = _mapping(document.get("global", {}), "$.global")
    composition_document = _mapping(document.get("composition", {}), "$.composition")
    engine_document = _mapping(document.get("engine", {}), "$.engine")
    passes_document = _mapping(document.get("passes", {}), "$.passes")

    regions = tuple(_parse_region(item, index) for index, item in enumerate(_sequence(document.get("regions", []), "$.regions")))

    return RegionalGenerationPlan(
        schema=CURRENT_SCHEMA,
        canvas=Canvas(
            width=_integer(canvas_document.get("width"), "$.canvas.width", 1024),
            height=_integer(canvas_document.get("height"), "$.canvas.height", 1024),
            extra=_extra(canvas_document, {"width", "height"}),
        ),
        global_prompt=GlobalPrompt(
            positive=_string(global_document.get("positive"), "$.global.positive", ""),
            negative=_string(global_document.get("negative"), "$.global.negative", ""),
            background_policy=_string(global_document.get("background_policy"), "$.global.background_policy", "global"),
            extra=_extra(global_document, {"positive", "negative", "background_policy"}),
        ),
        regions=regions,
        composition=Composition(
            overlap_policy=_enum(
                OverlapPolicy,
                composition_document.get("overlap_policy"),
                "$.composition.overlap_policy",
                OverlapPolicy.NORMALIZED,
            ),
            uncovered_policy=_enum(
                UncoveredPolicy,
                composition_document.get("uncovered_policy"),
                "$.composition.uncovered_policy",
                UncoveredPolicy.GLOBAL,
            ),
            mask_antialias=_boolean(composition_document.get("mask_antialias"), "$.composition.mask_antialias", True),
            extra=_extra(composition_document, {"overlap_policy", "uncovered_policy", "mask_antialias"}),
        ),
        engine=EngineSelection(
            requested=_string(engine_document.get("requested"), "$.engine.requested", "auto"),
            options=_mapping(engine_document.get("options", {}), "$.engine.options"),
            extra=_extra(engine_document, {"requested", "options"}),
        ),
        passes=PassPolicy(
            base=_string(passes_document.get("base"), "$.passes.base", "regional"),
            hires=_string(passes_document.get("hires"), "$.passes.hires", "recompile"),
            refiner=_string(passes_document.get("refiner"), "$.passes.refiner", "preserve_when_supported"),
            extra=_extra(passes_document, {"base", "hires", "refiner"}),
        ),
        extra=_extra(document, {"schema", "canvas", "global", "regions", "composition", "engine", "passes"}),
    )


def load_plan(value: str | bytes | bytearray | Mapping[str, Any]) -> RegionalGenerationPlan:
    if isinstance(value, Mapping):
        return plan_from_dict(value)

    raw = value.encode("utf-8") if isinstance(value, str) else bytes(value)
    if len(raw) > MAX_JSON_BYTES:
        raise PlanError("json.input_too_large", "$", f"Input exceeds {MAX_JSON_BYTES} bytes")

    try:
        document = json.loads(
            raw,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except PlanError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PlanError("json.invalid", "$", f"Invalid JSON: {error}") from error

    return plan_from_dict(_mapping(document, "$"))


def _plain(value: Any, *, include_transient: bool) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _plain(item, include_transient=include_transient)
            for key, item in value.items()
            if include_transient or key not in TRANSIENT_KEYS
        }
    if isinstance(value, tuple):
        return [_plain(item, include_transient=include_transient) for item in value]
    return value


def _merge_extra(extra: Mapping[str, Any], known: Mapping[str, Any], *, include_transient: bool) -> dict[str, Any]:
    result = _plain(extra, include_transient=include_transient)
    result.update(known)
    return result


def plan_to_dict(plan: RegionalGenerationPlan, *, include_transient: bool = True) -> dict[str, Any]:
    def geometry_to_dict(geometry) -> dict[str, Any]:
        if isinstance(geometry, RectGeometry):
            return _merge_extra(
                geometry.extra,
                {"type": geometry.type, "x": geometry.x, "y": geometry.y, "width": geometry.width, "height": geometry.height},
                include_transient=include_transient,
            )
        if isinstance(geometry, PolygonGeometry):
            return _merge_extra(
                geometry.extra,
                {"type": geometry.type, "points": [{"x": point.x, "y": point.y} for point in geometry.points]},
                include_transient=include_transient,
            )
        if isinstance(geometry, RasterMaskGeometry):
            known = {
                "type": geometry.type,
                "png_base64": geometry.png_base64,
                "width": geometry.width,
                "height": geometry.height,
            }
            if geometry.sha256 is not None:
                known["sha256"] = geometry.sha256
            return _merge_extra(geometry.extra, known, include_transient=include_transient)
        raise TypeError(f"Unsupported geometry: {type(geometry).__name__}")

    regions = []
    for region in plan.regions:
        seed = {"mode": region.seed.mode.value, "offset": region.seed.offset}
        if region.seed.value is not None:
            seed["value"] = region.seed.value
        regions.append(
            _merge_extra(
                region.extra,
                {
                    "id": str(region.id),
                    "name": region.name,
                    "enabled": region.enabled,
                    "locked": region.locked,
                    "hidden": region.hidden,
                    "positive": region.positive,
                    "negative": region.negative,
                    "inherit_global_positive": region.inherit_global_positive,
                    "inherit_global_negative": region.inherit_global_negative,
                    "geometry": geometry_to_dict(region.geometry),
                    "weight": region.weight,
                    "priority": region.priority,
                    "feather_px": region.feather_px,
                    "guidance": _merge_extra(
                        region.guidance.extra,
                        {"start": region.guidance.start, "end": region.guidance.end},
                        include_transient=include_transient,
                    ),
                    "seed": _merge_extra(region.seed.extra, seed, include_transient=include_transient),
                },
                include_transient=include_transient,
            )
        )

    return _merge_extra(
        plan.extra,
        {
            "schema": plan.schema,
            "canvas": _merge_extra(
                plan.canvas.extra,
                {"width": plan.canvas.width, "height": plan.canvas.height},
                include_transient=include_transient,
            ),
            "global": _merge_extra(
                plan.global_prompt.extra,
                {
                    "positive": plan.global_prompt.positive,
                    "negative": plan.global_prompt.negative,
                    "background_policy": plan.global_prompt.background_policy,
                },
                include_transient=include_transient,
            ),
            "regions": regions,
            "composition": _merge_extra(
                plan.composition.extra,
                {
                    "overlap_policy": plan.composition.overlap_policy.value,
                    "uncovered_policy": plan.composition.uncovered_policy.value,
                    "mask_antialias": plan.composition.mask_antialias,
                },
                include_transient=include_transient,
            ),
            "engine": _merge_extra(
                plan.engine.extra,
                {"requested": plan.engine.requested, "options": _plain(plan.engine.options, include_transient=include_transient)},
                include_transient=include_transient,
            ),
            "passes": _merge_extra(
                plan.passes.extra,
                {"base": plan.passes.base, "hires": plan.passes.hires, "refiner": plan.passes.refiner},
                include_transient=include_transient,
            ),
        },
        include_transient=include_transient,
    )


def canonical_json(plan: RegionalGenerationPlan, *, include_transient: bool = True) -> str:
    return json.dumps(
        plan_to_dict(plan, include_transient=include_transient),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def plan_hash(plan: RegionalGenerationPlan) -> str:
    canonical = canonical_json(plan, include_transient=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def save_plan(path: str | Path, plan: RegionalGenerationPlan) -> None:
    destination = Path(path)
    destination.write_text(canonical_json(plan) + "\n", encoding="utf-8", newline="\n")
