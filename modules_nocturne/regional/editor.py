"""Pure editor-state operations backed by the canonical Regional plan."""

from __future__ import annotations

import base64
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


@dataclass(frozen=True, slots=True)
class EditorValidation:
    accepted_json: str
    candidate_json: str
    plan_hash: str
    report: ValidationReport
    selected_region_id: UUID | None


def initial_editor_plan() -> RegionalGenerationPlan:
    return RegionalGenerationPlan(
        canvas=Canvas(width=1024, height=1024),
        global_prompt=GlobalPrompt(),
        engine=EngineSelection(options=DEFAULT_GENERATION_OPTIONS),
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


def render_layout_svg(plan: RegionalGenerationPlan, selected_region_id: str | UUID | None = None) -> str:
    selected = UUID(str(selected_region_id)) if selected_region_id else None
    shapes = []
    labels = []
    for region in plan.regions:
        if region.hidden:
            continue
        color = _region_color(region.id)
        opacity = "0.38" if region.enabled else "0.12"
        stroke_width = "4" if region.id == selected else "2"
        geometry = region.geometry
        common = (
            f'fill="{color}" fill-opacity="{opacity}" stroke="{color}" '
            f'stroke-width="{stroke_width}" vector-effect="non-scaling-stroke"'
        )
        if isinstance(geometry, RectGeometry):
            shapes.append(
                f'<rect x="{geometry.x * 1000:.4f}" y="{geometry.y * 1000:.4f}" '
                f'width="{geometry.width * 1000:.4f}" height="{geometry.height * 1000:.4f}" {common}/>'
            )
            label_x = geometry.x * 1000 + 12
            label_y = geometry.y * 1000 + 28
        elif isinstance(geometry, PolygonGeometry):
            points = " ".join(f"{point.x * 1000:.4f},{point.y * 1000:.4f}" for point in geometry.points)
            shapes.append(f'<polygon points="{points}" {common}/>')
            label_x = geometry.points[0].x * 1000 + 12
            label_y = geometry.points[0].y * 1000 + 28
        else:
            shapes.append(f'<rect x="0" y="0" width="1000" height="1000" {common} stroke-dasharray="12 8"/>')
            label_x = 12
            label_y = 28 + len(labels) * 32
        labels.append(
            f'<text x="{label_x:.4f}" y="{label_y:.4f}" fill="currentColor" '
            f'font-size="24" font-weight="600">{escape(region.name)}</text>'
        )

    return (
        '<div class="nocturne-layout-preview" role="img" '
        'aria-label="Regional layout preview with normalised canvas bounds">'
        '<svg viewBox="0 0 1000 1000" preserveAspectRatio="xMidYMid meet">'
        '<rect x="1" y="1" width="998" height="998" rx="8" fill="var(--block-background-fill)" '
        'stroke="var(--border-color-primary)" stroke-width="2"/>'
        '<g>' + "".join(shapes) + "</g><g>" + "".join(labels) + "</g>"
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
