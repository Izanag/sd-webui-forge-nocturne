"""Pure editor-state operations backed by the canonical Regional plan."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, replace
from html import escape
from io import BytesIO
from typing import Any
from uuid import UUID, uuid4

import numpy as np
from PIL import Image

from modules_nocturne.regional.errors import PlanError, PlanValidationError, ValidationReport
from modules_nocturne.regional.masks import CompiledMaskSet, compile_masks
from modules_nocturne.regional.model import (
    Canvas,
    EngineSelection,
    GlobalPrompt,
    Point,
    PolygonGeometry,
    RasterMaskGeometry,
    RectGeometry,
    Region,
    RegionGeometryValue,
    RegionalGenerationPlan,
)
from modules_nocturne.regional.serialization import canonical_json, load_plan, plan_hash
from modules_nocturne.regional.validation import validate_plan

DEFAULT_GENERATION_OPTIONS = {
    "sampler": "Euler a",
    "scheduler": "Automatic",
    "steps": 32,
    "cfg_scale": 6.0,
    "batch_count": 1,
    "batch_size": 1,
    "seed": -1,
}

MAX_EDITOR_POLYGON_POINTS = 1024


@dataclass(frozen=True, slots=True)
class EditorValidation:
    accepted_json: str
    candidate_json: str
    plan_hash: str
    report: ValidationReport
    selected_region_id: UUID | None


@dataclass(frozen=True, slots=True)
class GeometryEdit:
    region_id: UUID
    before: RegionGeometryValue
    after: RegionGeometryValue


def initial_editor_plan() -> RegionalGenerationPlan:
    return RegionalGenerationPlan(
        canvas=Canvas(width=1024, height=1024),
        global_prompt=GlobalPrompt(),
        engine=EngineSelection(options=DEFAULT_GENERATION_OPTIONS),
    )


def plan_from_txt2img_settings(
    *,
    positive: str,
    negative: str,
    width: int,
    height: int,
    sampler: str,
    scheduler: str,
    steps: int,
    cfg_scale: float,
    batch_count: int,
    batch_size: int,
    seed: int,
) -> RegionalGenerationPlan:
    plan = initial_editor_plan()
    plan = update_global_prompts(plan, positive, negative)
    plan = update_canvas(plan, width, height)
    return update_generation_options(
        plan,
        sampler=sampler,
        scheduler=scheduler,
        steps=steps,
        cfg_scale=cfg_scale,
        batch_count=batch_count,
        batch_size=batch_size,
        seed=seed,
    )


def _validated(plan: RegionalGenerationPlan) -> tuple[RegionalGenerationPlan, ValidationReport]:
    report = validate_plan(plan)
    if not report.valid:
        raise PlanValidationError(report)
    return plan, report


def load_valid_editor_plan(value: str) -> RegionalGenerationPlan:
    return _validated(load_plan(value))[0]


def validate_editor_json(
    candidate_json: str,
    *,
    last_valid_json: str,
    selected_region_id: str | UUID | None = None,
) -> EditorValidation:
    fallback = load_valid_editor_plan(last_valid_json)
    try:
        candidate, report = _validated(load_plan(candidate_json))
    except PlanValidationError as error:
        candidate = fallback
        report = error.report
    except PlanError as error:
        candidate = fallback
        report = ValidationReport((error.issue,))

    selected = UUID(str(selected_region_id)) if selected_region_id else None
    region_ids = {region.id for region in candidate.regions}
    if selected not in region_ids:
        selected = candidate.regions[0].id if candidate.regions else None
    serialized = canonical_json(candidate)
    return EditorValidation(
        accepted_json=serialized,
        candidate_json=candidate_json,
        plan_hash=plan_hash(candidate),
        report=report,
        selected_region_id=selected,
    )


def _next_region_id(plan: RegionalGenerationPlan) -> UUID:
    existing = {region.id for region in plan.regions}
    while True:
        candidate = uuid4()
        if candidate not in existing:
            return candidate


def _default_geometry(plan: RegionalGenerationPlan, mode: str):
    offset = (len(plan.regions) % 5) * 0.05
    if mode == "Polygon":
        return PolygonGeometry(
            (
                Point(0.2 + offset, 0.2 + offset),
                Point(0.6 + offset, 0.2 + offset),
                Point(0.4 + offset, 0.6 + offset),
            )
        )
    if mode == "Paint Mask":
        stream = BytesIO()
        Image.new("L", (64, 64), 0).save(stream, format="PNG", optimize=True)
        payload = stream.getvalue()
        import hashlib

        return RasterMaskGeometry(
            png_base64=base64.b64encode(payload).decode("ascii"),
            width=64,
            height=64,
            sha256=hashlib.sha256(payload).hexdigest(),
        )
    return RectGeometry(x=0.1 + offset, y=0.1 + offset, width=0.4, height=0.4)


def add_region(plan: RegionalGenerationPlan, mode: str = "Rectangle") -> tuple[RegionalGenerationPlan, UUID]:
    region_id = _next_region_id(plan)
    region = Region(
        id=region_id,
        name=f"Region {len(plan.regions) + 1}",
        geometry=_default_geometry(plan, mode),
    )
    return replace(plan, regions=(*plan.regions, region)), region_id


def duplicate_region(plan: RegionalGenerationPlan, region_id: UUID) -> tuple[RegionalGenerationPlan, UUID]:
    source = next((region for region in plan.regions if region.id == region_id), None)
    if source is None:
        raise PlanError("editor.region.not_found", "$.regions", "Selected region no longer exists")
    duplicate_id = _next_region_id(plan)
    duplicate = replace(source, id=duplicate_id, name=f"{source.name} copy")
    index = plan.regions.index(source) + 1
    regions = (*plan.regions[:index], duplicate, *plan.regions[index:])
    return replace(plan, regions=regions), duplicate_id


def delete_region(plan: RegionalGenerationPlan, region_id: UUID) -> tuple[RegionalGenerationPlan, UUID | None]:
    index = next((index for index, region in enumerate(plan.regions) if region.id == region_id), None)
    if index is None:
        raise PlanError("editor.region.not_found", "$.regions", "Selected region no longer exists")
    regions = (*plan.regions[:index], *plan.regions[index + 1 :])
    selected = regions[min(index, len(regions) - 1)].id if regions else None
    return replace(plan, regions=regions), selected


def move_region(plan: RegionalGenerationPlan, region_id: UUID, offset: int) -> RegionalGenerationPlan:
    index = next((index for index, region in enumerate(plan.regions) if region.id == region_id), None)
    if index is None:
        raise PlanError("editor.region.not_found", "$.regions", "Selected region no longer exists")
    target = max(0, min(len(plan.regions) - 1, index + offset))
    if target == index:
        return plan
    regions = list(plan.regions)
    region = regions.pop(index)
    regions.insert(target, region)
    return replace(plan, regions=tuple(regions))


def update_global_prompts(plan: RegionalGenerationPlan, positive: str, negative: str) -> RegionalGenerationPlan:
    return replace(plan, global_prompt=replace(plan.global_prompt, positive=positive, negative=negative))


def update_canvas(plan: RegionalGenerationPlan, width: int, height: int) -> RegionalGenerationPlan:
    return replace(plan, canvas=replace(plan.canvas, width=int(width), height=int(height)))


def update_generation_options(plan: RegionalGenerationPlan, **changes: Any) -> RegionalGenerationPlan:
    allowed = set(DEFAULT_GENERATION_OPTIONS)
    if not set(changes).issubset(allowed):
        raise PlanError("editor.generation.field_unsupported", "$.engine.options", "Unsupported generation control")
    options = dict(plan.engine.options)
    options.update(changes)
    return replace(plan, engine=replace(plan.engine, options=options))


def update_engine_request(plan: RegionalGenerationPlan, requested: str) -> RegionalGenerationPlan:
    if not requested:
        raise PlanError("editor.engine.invalid", "$.engine.requested", "Engine selection cannot be empty")
    return replace(plan, engine=replace(plan.engine, requested=requested))


def update_region(plan: RegionalGenerationPlan, region_id: UUID, **changes: Any) -> RegionalGenerationPlan:
    index = next((index for index, region in enumerate(plan.regions) if region.id == region_id), None)
    if index is None:
        raise PlanError("editor.region.not_found", "$.regions", "Selected region no longer exists")
    allowed = {
        "name",
        "enabled",
        "locked",
        "hidden",
        "positive",
        "negative",
        "inherit_global_positive",
        "inherit_global_negative",
        "weight",
        "priority",
        "feather_px",
        "grow_shrink_px",
        "guidance",
        "seed",
    }
    if not set(changes).issubset(allowed):
        raise PlanError("editor.region.field_unsupported", "$.regions", "Editor attempted to change an unsupported region field")
    regions = list(plan.regions)
    regions[index] = replace(regions[index], **changes)
    return replace(plan, regions=tuple(regions))


def update_region_geometry(plan: RegionalGenerationPlan, region_id: UUID, geometry: Any) -> RegionalGenerationPlan:
    index = next((index for index, region in enumerate(plan.regions) if region.id == region_id), None)
    if index is None:
        raise PlanError("editor.region.not_found", "$.regions", "Selected region no longer exists")
    if plan.regions[index].locked:
        raise PlanError("editor.region.locked", f"$.regions[{index}].geometry", "Unlock the region before editing its geometry")
    if not isinstance(geometry, (RectGeometry, PolygonGeometry, RasterMaskGeometry)):
        raise PlanError("editor.geometry.unsupported", f"$.regions[{index}].geometry", "Unsupported region geometry")
    regions = list(plan.regions)
    regions[index] = replace(regions[index], geometry=geometry)
    return replace(plan, regions=tuple(regions))


def geometry_edit_between(
    before_plan: RegionalGenerationPlan,
    after_plan: RegionalGenerationPlan,
    region_id: UUID,
) -> GeometryEdit:
    before = selected_region(before_plan, region_id)
    after = selected_region(after_plan, region_id)
    if before is None or after is None:
        raise PlanError("editor.geometry.history_region_missing", "$.regions", "Geometry history region no longer exists")
    return GeometryEdit(region_id=region_id, before=before.geometry, after=after.geometry)


def apply_geometry_edit(
    plan: RegionalGenerationPlan,
    edit: GeometryEdit,
    *,
    undo: bool,
) -> RegionalGenerationPlan:
    region = selected_region(plan, edit.region_id)
    if region is None:
        raise PlanError("editor.geometry.history_region_missing", "$.regions", "Geometry history region no longer exists")
    expected = edit.after if undo else edit.before
    replacement = edit.before if undo else edit.after
    if region.geometry != expected:
        raise PlanError(
            "editor.geometry.history_stale",
            "$.regions",
            "Geometry changed outside this history; start a new geometry edit before using undo or redo",
        )
    return update_region_geometry(plan, edit.region_id, replacement)


def update_rectangle_geometry(
    plan: RegionalGenerationPlan,
    region_id: UUID,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
) -> RegionalGenerationPlan:
    current = selected_region(plan, region_id)
    extra = current.geometry.extra if current and isinstance(current.geometry, RectGeometry) else {}
    return update_region_geometry(
        plan,
        region_id,
        RectGeometry(x=float(x), y=float(y), width=float(width), height=float(height), extra=extra),
    )


def update_polygon_geometry(
    plan: RegionalGenerationPlan,
    region_id: UUID,
    points_value: str | list[Any] | tuple[Any, ...],
) -> RegionalGenerationPlan:
    try:
        raw_points = json.loads(points_value) if isinstance(points_value, str) else points_value
    except json.JSONDecodeError as error:
        raise PlanError("editor.geometry.polygon.invalid_json", "$.geometry.points", "Polygon points must be valid JSON") from error
    if not isinstance(raw_points, (list, tuple)) or not 3 <= len(raw_points) <= MAX_EDITOR_POLYGON_POINTS:
        raise PlanError(
            "editor.geometry.polygon.point_count",
            "$.geometry.points",
            f"Polygon points must contain between 3 and {MAX_EDITOR_POLYGON_POINTS} entries",
        )
    points = []
    for index, raw_point in enumerate(raw_points):
        path = f"$.geometry.points[{index}]"
        if isinstance(raw_point, dict):
            raw_x, raw_y = raw_point.get("x"), raw_point.get("y")
        elif isinstance(raw_point, (list, tuple)) and len(raw_point) == 2:
            raw_x, raw_y = raw_point
        else:
            raise PlanError("editor.geometry.polygon.point_invalid", path, "Each point must contain x and y values")
        if isinstance(raw_x, bool) or isinstance(raw_y, bool):
            raise PlanError("editor.geometry.polygon.point_invalid", path, "Point coordinates must be numbers")
        try:
            points.append(Point(float(raw_x), float(raw_y)))
        except (TypeError, ValueError) as error:
            raise PlanError("editor.geometry.polygon.point_invalid", path, "Point coordinates must be numbers") from error
    current = selected_region(plan, region_id)
    extra = current.geometry.extra if current and isinstance(current.geometry, PolygonGeometry) else {}
    return update_region_geometry(plan, region_id, PolygonGeometry(points=tuple(points), extra=extra))


def update_raster_geometry(
    plan: RegionalGenerationPlan,
    region_id: UUID,
    image: Image.Image,
) -> RegionalGenerationPlan:
    if not isinstance(image, Image.Image):
        raise PlanError("editor.geometry.raster.missing", "$.geometry", "Paint or upload a mask before applying it")
    mask = image.convert("L")
    stream = BytesIO()
    mask.save(stream, format="PNG", optimize=True)
    payload = stream.getvalue()
    geometry = RasterMaskGeometry(
        png_base64=base64.b64encode(payload).decode("ascii"),
        width=mask.width,
        height=mask.height,
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    return update_region_geometry(plan, region_id, geometry)


def raster_geometry_image(region: Region | None) -> Image.Image | None:
    if region is None or not isinstance(region.geometry, RasterMaskGeometry):
        return None
    try:
        payload = base64.b64decode(region.geometry.png_base64, validate=True)
        with Image.open(BytesIO(payload)) as source:
            source.load()
            return source.convert("L")
    except (ValueError, OSError) as error:
        raise PlanError("editor.geometry.raster.decode_failed", "$.geometry", "Stored raster mask could not be decoded") from error


def region_choices(plan: RegionalGenerationPlan) -> list[tuple[str, str]]:
    return [
        (f"{'●' if region.enabled else '○'} {region.name}", str(region.id))
        for region in plan.regions
    ]


def selected_region(plan: RegionalGenerationPlan, region_id: str | UUID | None) -> Region | None:
    if not region_id:
        return None
    selected_id = UUID(str(region_id))
    return next((region for region in plan.regions if region.id == selected_id), None)


def _region_color(region_id: UUID) -> str:
    raw = region_id.int
    return f"hsl({raw % 360} 72% 55%)"


def _region_colors(plan: RegionalGenerationPlan) -> dict[UUID, str]:
    hues: list[int] = []
    colors = {}
    for region in plan.regions:
        hue = region.id.int % 360
        while any(min(abs(hue - used), 360 - abs(hue - used)) < 36 for used in hues):
            hue = (hue + 137) % 360
        hues.append(hue)
        colors[region.id] = f"hsl({hue} 72% 55%)"
    return colors


def render_region_table(
    plan: RegionalGenerationPlan,
    selected_region_id: str | UUID | None = None,
) -> str:
    selected = UUID(str(selected_region_id)) if selected_region_id else None
    if not plan.regions:
        return '<div class="nocturne-region-table-empty">No regions yet.</div>'

    colors = _region_colors(plan)
    rows = []
    for index, region in enumerate(plan.regions, start=1):
        region_id = str(region.id)
        selected_class = " is-selected" if region.id == selected else ""
        state = "Enabled" if region.enabled else "Disabled"
        rows.append(
            f'<tr class="nocturne-region-row{selected_class}">'
            '<td>'
            f'<button type="button" data-nocturne-select-region="{region_id}" '
            f'style="--region-color:{colors[region.id]}" '
            f'aria-pressed="{"true" if region.id == selected else "false"}">'
            '<span class="nocturne-region-swatch"></span>'
            f'<span class="nocturne-region-index">{index}</span>'
            f'<span class="nocturne-region-name">{escape(region.name)}</span>'
            f'<span class="nocturne-region-state">{state}</span>'
            "</button>"
            "</td>"
            "</tr>"
        )
    return (
        '<div class="nocturne-region-table-wrap">'
        '<table class="nocturne-region-table" aria-label="Regions"><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
    )


def render_layout_svg(plan: RegionalGenerationPlan, selected_region_id: str | UUID | None = None) -> str:
    selected = UUID(str(selected_region_id)) if selected_region_id else None
    canvas_width = float(plan.canvas.width)
    canvas_height = float(plan.canvas.height)
    short_side = min(canvas_width, canvas_height)
    handle_radius = max(5.0, short_side * 0.009)
    label_size = max(14.0, short_side * 0.024)
    label_offset_x = max(8.0, canvas_width * 0.012)
    label_offset_y = max(18.0, canvas_height * 0.028)
    shapes = []
    handles = []
    labels = []
    colors = _region_colors(plan)
    for region in plan.regions:
        if region.hidden:
            continue
        color = colors[region.id]
        opacity = "0.38" if region.enabled else "0.12"
        stroke_width = "4" if region.id == selected else "2"
        geometry = region.geometry
        common = (
            f'fill="{color}" fill-opacity="{opacity}" stroke="{color}" '
            f'stroke-width="{stroke_width}" vector-effect="non-scaling-stroke" '
            f'class="nocturne-region-shape" data-region-id="{region.id}"'
        )
        if isinstance(geometry, RectGeometry):
            x = geometry.x * canvas_width
            y = geometry.y * canvas_height
            width = geometry.width * canvas_width
            height = geometry.height * canvas_height
            shapes.append(
                f'<rect x="{x:.4f}" y="{y:.4f}" '
                f'width="{width:.4f}" height="{height:.4f}" {common}/>'
            )
            if region.id == selected and not region.locked:
                corners = (
                    ("nw", geometry.x, geometry.y),
                    ("ne", geometry.x + geometry.width, geometry.y),
                    ("sw", geometry.x, geometry.y + geometry.height),
                    ("se", geometry.x + geometry.width, geometry.y + geometry.height),
                )
                handles.extend(
                    f'<circle cx="{x * canvas_width:.4f}" cy="{y * canvas_height:.4f}" r="{handle_radius:.4f}" '
                    f'fill="{color}" stroke="white" stroke-width="3" vector-effect="non-scaling-stroke" '
                    f'class="nocturne-geometry-handle" data-region-id="{region.id}" data-rect-corner="{corner}"/>'
                    for corner, x, y in corners
                )
            label_x = x + label_offset_x
            label_y = y + label_offset_y
        elif isinstance(geometry, PolygonGeometry):
            points = " ".join(
                f"{point.x * canvas_width:.4f},{point.y * canvas_height:.4f}"
                for point in geometry.points
            )
            shapes.append(f'<polygon points="{points}" {common}/>')
            if region.id == selected and not region.locked:
                handles.extend(
                    f'<circle cx="{point.x * canvas_width:.4f}" cy="{point.y * canvas_height:.4f}" r="{handle_radius:.4f}" '
                    f'fill="{color}" stroke="white" stroke-width="3" vector-effect="non-scaling-stroke" '
                    f'class="nocturne-geometry-handle" data-region-id="{region.id}" data-point-index="{index}"/>'
                    for index, point in enumerate(geometry.points)
                )
            label_x = geometry.points[0].x * canvas_width + label_offset_x
            label_y = geometry.points[0].y * canvas_height + label_offset_y
        else:
            shapes.append(
                f'<rect x="0" y="0" width="{canvas_width:.4f}" height="{canvas_height:.4f}" '
                f'{common} stroke-dasharray="12 8"/>'
            )
            label_x = label_offset_x
            label_y = label_offset_y + len(labels) * label_size * 1.35
        labels.append(
            f'<text x="{label_x:.4f}" y="{label_y:.4f}" fill="currentColor" '
            f'font-size="{label_size:.4f}" font-weight="600">{escape(region.name)}</text>'
        )

    grid_step = max(24.0, short_side / 16)
    empty_state = (
        f'<text x="{canvas_width / 2:.4f}" y="{canvas_height / 2:.4f}" '
        f'fill="currentColor" fill-opacity=".58" font-size="{max(18.0, label_size * 1.2):.4f}" '
        'font-weight="600" text-anchor="middle" dominant-baseline="middle">'
        'Add a region to begin</text>'
        if not plan.regions
        else ""
    )
    return (
        '<div class="nocturne-layout-preview" role="img" '
        'aria-label="Regional layout preview with normalised canvas bounds" '
        f'style="aspect-ratio:{plan.canvas.width}/{plan.canvas.height}">'
        f'<svg viewBox="0 0 {canvas_width:.4f} {canvas_height:.4f}" preserveAspectRatio="none" '
        f'data-selected-region="{selected or ""}" data-grid-step="{grid_step:.4f}">'
        f'<defs><pattern id="nocturne-grid" width="{grid_step:.4f}" height="{grid_step:.4f}" '
        f'patternUnits="userSpaceOnUse"><path d="M {grid_step:.4f} 0 L 0 0 0 {grid_step:.4f}" '
        'fill="none" stroke="currentColor" stroke-opacity=".2" stroke-width="1"/></pattern></defs>'
        f'<rect x="0" y="0" width="{canvas_width:.4f}" height="{canvas_height:.4f}" '
        'fill="color-mix(in srgb, var(--block-background-fill) 91%, var(--body-text-color) 9%)"/>'
        f'<rect x="0" y="0" width="{canvas_width:.4f}" height="{canvas_height:.4f}" '
        'fill="url(#nocturne-grid)"/>'
        + empty_state
        + '<g>' + "".join(shapes) + "</g><g>" + "".join(handles) + "</g><g class=\"nocturne-region-labels\">" + "".join(labels) + "</g>"
        "</svg></div>"
    )


def render_mask_preview(plan: RegionalGenerationPlan, *, maximum_dimension: int = 384) -> tuple[Image.Image, CompiledMaskSet]:
    scale = maximum_dimension / max(plan.canvas.width, plan.canvas.height)
    width = max(1, round(plan.canvas.width * scale))
    height = max(1, round(plan.canvas.height * scale))
    compiled = compile_masks(plan, width=width, height=height)
    preview = np.repeat(compiled.global_mask[:, :, None] * np.float32(0.18), 3, axis=2)
    for region_id, mask in compiled.region_masks.items():
        raw = region_id.int
        color = np.asarray(
            (
                0.35 + ((raw >> 0) & 255) / 510,
                0.35 + ((raw >> 8) & 255) / 510,
                0.35 + ((raw >> 16) & 255) / 510,
            ),
            dtype=np.float32,
        )
        preview += mask[:, :, None] * color[None, None, :]
    return Image.fromarray(np.uint8(np.clip(preview, 0.0, 1.0) * 255), mode="RGB"), compiled


def validation_markdown(report: ValidationReport) -> str:
    if not report.issues:
        return "✓ Plan is structurally valid."
    lines = []
    for issue in report.issues:
        marker = "✕" if issue.severity.value == "error" else "⚠"
        lines.append(f"- {marker} `{issue.code}` at `{issue.path}` — {issue.message}")
    return "\n".join(lines)
