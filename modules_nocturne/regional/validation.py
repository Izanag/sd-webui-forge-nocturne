"""CPU-only validation for canonical Regional plans."""

import base64
import binascii
import hashlib
import math
import struct
from dataclasses import dataclass

from modules_nocturne.regional.errors import IssueSeverity, ValidationIssue, ValidationReport
from modules_nocturne.regional.model import (
    CURRENT_SCHEMA,
    OverlapPolicy,
    PolygonGeometry,
    RasterMaskGeometry,
    RectGeometry,
    RegionalGenerationPlan,
    SeedMode,
    UncoveredPolicy,
)

MIN_CANVAS_DIMENSION = 64
MAX_CANVAS_DIMENSION = 8192
MAX_REGION_NAME_LENGTH = 128
MAX_REGION_WEIGHT = 1000
MAX_RASTER_ENCODED_BYTES = 16 * 1024 * 1024
MAX_RASTER_PIXELS = 16_777_216
MAX_FEATHER_PX = 2048
MAX_GROW_SHRINK_PX = 512
MAX_SEED = (1 << 32) - 1
MIN_SEED_OFFSET = -(1 << 63)
MAX_SEED_OFFSET = (1 << 63) - 1
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True, slots=True)
class ValidationCapabilities:
    overlap_policies: frozenset[OverlapPolicy] = frozenset(OverlapPolicy)
    uncovered_policies: frozenset[UncoveredPolicy] = frozenset(
        {UncoveredPolicy.GLOBAL, UncoveredPolicy.NEAREST, UncoveredPolicy.ERROR}
    )


def _issue(
    issues: list[ValidationIssue],
    code: str,
    path: str,
    message: str,
    severity: IssueSeverity = IssueSeverity.ERROR,
    **details,
) -> None:
    issues.append(ValidationIssue(code=code, path=path, message=message, severity=severity, details=details))


def _segments_intersect(a, b, c, d) -> bool:
    def orientation(p, q, r) -> float:
        return (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)

    def on_segment(p, q, r) -> bool:
        return min(p.x, r.x) <= q.x <= max(p.x, r.x) and min(p.y, r.y) <= q.y <= max(p.y, r.y)

    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    epsilon = 1e-12

    if o1 * o2 < -epsilon and o3 * o4 < -epsilon:
        return True
    if abs(o1) <= epsilon and on_segment(a, c, b):
        return True
    if abs(o2) <= epsilon and on_segment(a, d, b):
        return True
    if abs(o3) <= epsilon and on_segment(c, a, d):
        return True
    if abs(o4) <= epsilon and on_segment(c, b, d):
        return True
    return False


def _polygon_area(points) -> float:
    return abs(
        sum(
            point.x * points[(index + 1) % len(points)].y - points[(index + 1) % len(points)].x * point.y
            for index, point in enumerate(points)
        )
    ) / 2.0


def _validate_raster(
    geometry: RasterMaskGeometry,
    path: str,
    issues: list[ValidationIssue],
) -> None:
    if len(geometry.png_base64) > MAX_RASTER_ENCODED_BYTES:
        _issue(issues, "geometry.raster.encoded_too_large", f"{path}.png_base64", "Raster mask payload is too large")
        return

    try:
        payload = base64.b64decode(geometry.png_base64, validate=True)
    except (ValueError, binascii.Error):
        _issue(issues, "geometry.raster.invalid_base64", f"{path}.png_base64", "Raster mask is not valid base64")
        return

    if len(payload) < 24 or not payload.startswith(PNG_SIGNATURE) or payload[12:16] != b"IHDR":
        _issue(issues, "geometry.raster.invalid_png", f"{path}.png_base64", "Raster mask must be a lossless PNG")
        return

    png_width, png_height = struct.unpack(">II", payload[16:24])
    if png_width != geometry.width or png_height != geometry.height:
        _issue(
            issues,
            "geometry.raster.dimension_mismatch",
            path,
            "Declared raster dimensions do not match the PNG header",
            declared_width=geometry.width,
            declared_height=geometry.height,
            png_width=png_width,
            png_height=png_height,
        )
    if png_width == 0 or png_height == 0 or png_width * png_height > MAX_RASTER_PIXELS:
        _issue(issues, "geometry.raster.unsafe_dimensions", path, "Raster mask dimensions exceed safety limits")
    if geometry.sha256 is not None and hashlib.sha256(payload).hexdigest() != geometry.sha256.lower():
        _issue(issues, "geometry.raster.hash_mismatch", f"{path}.sha256", "Raster mask checksum does not match")


def validate_plan(
    plan: RegionalGenerationPlan,
    *,
    max_enabled_regions: int = 16,
    capabilities: ValidationCapabilities | None = None,
) -> ValidationReport:
    issues: list[ValidationIssue] = []

    if plan.schema != CURRENT_SCHEMA:
        _issue(issues, "schema.unsupported_version", "$.schema", f"Expected schema {CURRENT_SCHEMA!r}")

    for field_name, value in (("width", plan.canvas.width), ("height", plan.canvas.height)):
        if value < MIN_CANVAS_DIMENSION or value > MAX_CANVAS_DIMENSION:
            _issue(
                issues,
                f"canvas.{field_name}.out_of_range",
                f"$.canvas.{field_name}",
                f"Canvas {field_name} must be between {MIN_CANVAS_DIMENSION} and {MAX_CANVAS_DIMENSION}",
            )

    enabled_count = sum(region.enabled for region in plan.regions)
    if enabled_count > max_enabled_regions:
        _issue(
            issues,
            "regions.enabled_limit_exceeded",
            "$.regions",
            f"Plan enables {enabled_count} regions; the configured limit is {max_enabled_regions}",
            enabled_count=enabled_count,
            limit=max_enabled_regions,
        )

    seen_ids = set()
    for index, region in enumerate(plan.regions):
        path = f"$.regions[{index}]"
        if region.id in seen_ids:
            _issue(issues, "region.id.duplicate", f"{path}.id", f"Region UUID {region.id} is duplicated")
        seen_ids.add(region.id)

        if not region.name.strip():
            _issue(issues, "region.name.empty", f"{path}.name", "Region name cannot be empty")
        elif len(region.name) > MAX_REGION_NAME_LENGTH:
            _issue(issues, "region.name.too_long", f"{path}.name", f"Region name exceeds {MAX_REGION_NAME_LENGTH} characters")

        if not math.isfinite(region.weight) or region.weight <= 0:
            _issue(issues, "region.weight.invalid", f"{path}.weight", "Region weight must be finite and greater than zero")
        elif region.weight > MAX_REGION_WEIGHT:
            _issue(
                issues,
                "region.weight.limit_exceeded",
                f"{path}.weight",
                f"Region weight cannot exceed {MAX_REGION_WEIGHT}",
            )
        if not math.isfinite(region.feather_px) or region.feather_px < 0:
            _issue(issues, "region.feather.invalid", f"{path}.feather_px", "Feather amount must be finite and non-negative")
        elif region.feather_px > MAX_FEATHER_PX:
            _issue(
                issues,
                "region.feather.limit_exceeded",
                f"{path}.feather_px",
                f"Feather amount cannot exceed {MAX_FEATHER_PX} canvas pixels",
            )
        if not math.isfinite(region.grow_shrink_px):
            _issue(
                issues,
                "region.grow_shrink.invalid",
                f"{path}.grow_shrink_px",
                "Grow/shrink amount must be finite",
            )
        elif abs(region.grow_shrink_px) > MAX_GROW_SHRINK_PX:
            _issue(
                issues,
                "region.grow_shrink.limit_exceeded",
                f"{path}.grow_shrink_px",
                f"Grow/shrink amount cannot exceed {MAX_GROW_SHRINK_PX} canvas pixels in either direction",
            )
        if not (0.0 <= region.guidance.start <= region.guidance.end <= 1.0):
            _issue(
                issues,
                "region.guidance.invalid",
                f"{path}.guidance",
                "Guidance start and end must satisfy 0 <= start <= end <= 1",
            )
        if region.seed.mode == SeedMode.FIXED and region.seed.value is None:
            _issue(issues, "region.seed.fixed_value_missing", f"{path}.seed.value", "Fixed seed mode requires a value")
        elif region.seed.value is not None and not 0 <= region.seed.value <= MAX_SEED:
            _issue(issues, "region.seed.value_out_of_range", f"{path}.seed.value", f"Seed must be between 0 and {MAX_SEED}")
        if not MIN_SEED_OFFSET <= region.seed.offset <= MAX_SEED_OFFSET:
            _issue(
                issues,
                "region.seed.offset_out_of_range",
                f"{path}.seed.offset",
                "Seed offset must fit a signed 64-bit integer",
            )

        geometry = region.geometry
        geometry_path = f"{path}.geometry"
        if isinstance(geometry, RectGeometry):
            values = (geometry.x, geometry.y, geometry.width, geometry.height)
            if not all(math.isfinite(value) for value in values):
                _issue(issues, "geometry.rect.non_finite", geometry_path, "Rectangle values must be finite")
            elif geometry.width <= 0 or geometry.height <= 0:
                _issue(issues, "geometry.rect.degenerate", geometry_path, "Rectangle width and height must be greater than zero")
            elif geometry.x < 0 or geometry.y < 0 or geometry.x + geometry.width > 1 or geometry.y + geometry.height > 1:
                _issue(issues, "geometry.rect.out_of_bounds", geometry_path, "Rectangle must remain inside normalised canvas bounds")
            elif geometry.width * plan.canvas.width < 1 or geometry.height * plan.canvas.height < 1:
                _issue(
                    issues,
                    "geometry.sub_pixel",
                    geometry_path,
                    "Rectangle is smaller than one output pixel",
                    IssueSeverity.WARNING,
                )

        elif isinstance(geometry, PolygonGeometry):
            if len(geometry.points) < 3:
                _issue(issues, "geometry.polygon.too_few_points", f"{geometry_path}.points", "Polygon requires at least three points")
                continue
            if any(not math.isfinite(point.x) or not math.isfinite(point.y) for point in geometry.points):
                _issue(issues, "geometry.polygon.non_finite", f"{geometry_path}.points", "Polygon points must be finite")
                continue
            if any(point.x < 0 or point.x > 1 or point.y < 0 or point.y > 1 for point in geometry.points):
                _issue(issues, "geometry.polygon.out_of_bounds", f"{geometry_path}.points", "Polygon must remain inside normalised canvas bounds")
            area = _polygon_area(geometry.points)
            if area <= 1e-12:
                _issue(issues, "geometry.polygon.degenerate", geometry_path, "Polygon area must be greater than zero")
            elif area * plan.canvas.width * plan.canvas.height < 1:
                _issue(issues, "geometry.sub_pixel", geometry_path, "Polygon covers less than one output pixel", IssueSeverity.WARNING)

            segment_count = len(geometry.points)
            intersects = False
            for first in range(segment_count):
                a = geometry.points[first]
                b = geometry.points[(first + 1) % segment_count]
                for second in range(first + 1, segment_count):
                    if second in {first, (first + 1) % segment_count}:
                        continue
                    if first == 0 and second == segment_count - 1:
                        continue
                    c = geometry.points[second]
                    d = geometry.points[(second + 1) % segment_count]
                    if _segments_intersect(a, b, c, d):
                        intersects = True
                        break
                if intersects:
                    break
            if intersects:
                _issue(issues, "geometry.polygon.self_intersection", geometry_path, "Polygon edges cannot self-intersect")

        elif isinstance(geometry, RasterMaskGeometry):
            _validate_raster(geometry, geometry_path, issues)

    if capabilities is None:
        if plan.composition.uncovered_policy == UncoveredPolicy.TRANSPARENT:
            _issue(
                issues,
                "composition.transparent.requires_capability",
                "$.composition.uncovered_policy",
                "Transparent uncovered areas require confirmation from the selected engine",
                IssueSeverity.APPROVAL_REQUIRED,
            )
    else:
        if plan.composition.overlap_policy not in capabilities.overlap_policies:
            _issue(
                issues,
                "composition.overlap_policy.unsupported",
                "$.composition.overlap_policy",
                f"The selected engine does not support {plan.composition.overlap_policy.value!r}",
            )
        if plan.composition.uncovered_policy not in capabilities.uncovered_policies:
            _issue(
                issues,
                "composition.uncovered_policy.unsupported",
                "$.composition.uncovered_policy",
                f"The selected engine does not support {plan.composition.uncovered_policy.value!r}",
            )

    return ValidationReport(tuple(issues))
