"""Bounded CPU mask compilation for canonical Regional plans."""

from __future__ import annotations

import base64
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
import math
from threading import RLock
from types import MappingProxyType
from typing import Iterable, Mapping
from uuid import UUID

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, UnidentifiedImageError

from modules_nocturne.regional.errors import PlanError, PlanValidationError
from modules_nocturne.regional.model import (
    OverlapPolicy,
    PolygonGeometry,
    RasterMaskGeometry,
    RectGeometry,
    RegionalGenerationPlan,
    UncoveredPolicy,
)
from modules_nocturne.regional.serialization import plan_hash
from modules_nocturne.regional.validation import (
    MAX_CANVAS_DIMENSION,
    MAX_RASTER_PIXELS,
    ValidationCapabilities,
    validate_plan,
)

DEFAULT_SUPERSAMPLE = 4
MAX_TARGET_PIXELS = MAX_RASTER_PIXELS
MAX_WORKING_PIXELS = 64 * 1024 * 1024
MAX_COMPILER_WORKING_BYTES = 512 * 1024 * 1024
MAX_EFFECTIVE_FILTER_RADIUS = 2048
MASK_EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class RegionMaskDiagnostics:
    region_id: UUID
    source_coverage: float
    effective_coverage: float
    empty: bool
    subpixel: bool
    fully_occluded: bool


@dataclass(frozen=True, slots=True)
class MaskDiagnostics:
    regions: tuple[RegionMaskDiagnostics, ...]
    covered_fraction: float
    uncovered_fraction: float
    overlap_fraction: float
    maximum_overlap: int


@dataclass(frozen=True, slots=True)
class MaskCacheKey:
    plan_hash: str
    width: int
    height: int
    adapter_id: str
    device: str
    dtype: str


@dataclass(frozen=True, slots=True)
class CompiledMaskSet:
    key: MaskCacheKey
    region_masks: Mapping[UUID, np.ndarray]
    global_mask: np.ndarray
    diagnostics: MaskDiagnostics

    def __post_init__(self) -> None:
        if self.global_mask.flags.writeable or any(mask.flags.writeable for mask in self.region_masks.values()):
            raise ValueError("Compiled masks must use immutable array storage")
        object.__setattr__(self, "region_masks", MappingProxyType(dict(self.region_masks)))

    @property
    def width(self) -> int:
        return self.key.width

    @property
    def height(self) -> int:
        return self.key.height

    @property
    def nbytes(self) -> int:
        return self.global_mask.nbytes + sum(mask.nbytes for mask in self.region_masks.values())


@dataclass(frozen=True, slots=True)
class MaskCacheStats:
    entries: int
    bytes: int
    max_entries: int
    max_bytes: int


class MaskCompilerCache:
    """Thread-safe LRU for immutable CPU arrays."""

    def __init__(self, *, max_entries: int = 8, max_bytes: int = 256 * 1024 * 1024):
        if max_entries < 1 or max_bytes < 1:
            raise ValueError("Mask cache bounds must be positive")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries: OrderedDict[MaskCacheKey, CompiledMaskSet] = OrderedDict()
        self._bytes = 0
        self._lock = RLock()

    def get(self, key: MaskCacheKey) -> CompiledMaskSet | None:
        with self._lock:
            value = self._entries.get(key)
            if value is not None:
                self._entries.move_to_end(key)
            return value

    def put(self, value: CompiledMaskSet) -> CompiledMaskSet:
        if value.nbytes > self.max_bytes:
            return value
        with self._lock:
            replaced = self._entries.pop(value.key, None)
            if replaced is not None:
                self._bytes -= replaced.nbytes
            self._entries[value.key] = value
            self._bytes += value.nbytes
            while len(self._entries) > self.max_entries or self._bytes > self.max_bytes:
                _, evicted = self._entries.popitem(last=False)
                self._bytes -= evicted.nbytes
        return value

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def stats(self) -> MaskCacheStats:
        with self._lock:
            return MaskCacheStats(
                entries=len(self._entries),
                bytes=self._bytes,
                max_entries=self.max_entries,
                max_bytes=self.max_bytes,
            )


mask_cache = MaskCompilerCache()


def _target_size(width: int, height: int) -> tuple[int, int]:
    if isinstance(width, bool) or isinstance(height, bool) or not isinstance(width, int) or not isinstance(height, int):
        raise PlanError("mask.target.invalid", "$.target", "Mask target dimensions must be integers")
    if width < 1 or height < 1 or width > MAX_CANVAS_DIMENSION or height > MAX_CANVAS_DIMENSION:
        raise PlanError(
            "mask.target.out_of_range",
            "$.target",
            f"Mask target dimensions must be between 1 and {MAX_CANVAS_DIMENSION}",
        )
    if width * height > MAX_TARGET_PIXELS:
        raise PlanError("mask.target.too_large", "$.target", "Mask target exceeds the safe pixel limit")
    return width, height


def _working_size(width: int, height: int, antialias: bool) -> tuple[int, int, int]:
    if not antialias:
        return width, height, 1
    maximum = max(1, int(math.sqrt(MAX_WORKING_PIXELS / (width * height))))
    factor = min(DEFAULT_SUPERSAMPLE, maximum)
    return width * factor, height * factor, factor


def _decode_raster(geometry: RasterMaskGeometry, width: int, height: int) -> Image.Image:
    try:
        payload = base64.b64decode(geometry.png_base64, validate=True)
        with Image.open(BytesIO(payload)) as source:
            if source.width * source.height > MAX_RASTER_PIXELS:
                raise PlanError("geometry.raster.decoded_too_large", "$.regions", "Decoded raster mask is too large")
            source.load()
            if "A" in source.getbands():
                alpha = source.getchannel("A")
            else:
                alpha = source.convert("L")
            return alpha.resize((width, height), Image.Resampling.LANCZOS)
    except PlanError:
        raise
    except (OSError, ValueError, UnidentifiedImageError) as error:
        raise PlanError("geometry.raster.decode_failed", "$.regions", "Raster mask could not be decoded safely") from error


def _rasterise_region(region, width: int, height: int) -> Image.Image:
    geometry = region.geometry
    if isinstance(geometry, RasterMaskGeometry):
        return _decode_raster(geometry, width, height)

    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    if isinstance(geometry, RectGeometry):
        left = geometry.x * width
        top = geometry.y * height
        right = (geometry.x + geometry.width) * width
        bottom = (geometry.y + geometry.height) * height
        draw.rectangle((left, top, max(left, right - 1), max(top, bottom - 1)), fill=255)
    elif isinstance(geometry, PolygonGeometry):
        draw.polygon(tuple((point.x * width, point.y * height) for point in geometry.points), fill=255)
    else:
        raise PlanError("geometry.unsupported_type", "$.regions", f"Unsupported geometry type {type(geometry).__name__}")
    return image


def _apply_morphology(image: Image.Image, radius: int, grow: bool) -> Image.Image:
    result = image
    remaining = radius
    while remaining:
        step = min(remaining, 15)
        filter_type = ImageFilter.MaxFilter if grow else ImageFilter.MinFilter
        result = result.filter(filter_type(step * 2 + 1))
        remaining -= step
    return result


def _prepare_region_mask(plan: RegionalGenerationPlan, region, width: int, height: int, factor: int) -> np.ndarray:
    working_width = width * factor
    working_height = height * factor
    image = _rasterise_region(region, working_width, working_height)
    output_scale = math.sqrt((width / plan.canvas.width) * (height / plan.canvas.height))

    morphology_radius = round(abs(region.grow_shrink_px) * output_scale * factor)
    feather_radius = region.feather_px * output_scale * factor
    if morphology_radius > MAX_EFFECTIVE_FILTER_RADIUS or feather_radius > MAX_EFFECTIVE_FILTER_RADIUS:
        raise PlanError(
            "mask.filter_radius.too_large",
            "$.regions",
            f"Effective mask filter radius cannot exceed {MAX_EFFECTIVE_FILTER_RADIUS} working pixels",
        )
    if morphology_radius:
        image = _apply_morphology(image, morphology_radius, region.grow_shrink_px > 0)
    if feather_radius:
        image = image.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    if factor > 1:
        image = image.resize((width, height), Image.Resampling.BOX)
    return np.asarray(image, dtype=np.float32) / np.float32(255.0)


def _apply_overlap_policy(
    masks: np.ndarray,
    regions: tuple,
    policy: OverlapPolicy,
) -> np.ndarray:
    if masks.shape[0] == 0:
        return masks

    weights = np.asarray([region.weight for region in regions], dtype=np.float32)[:, None, None]
    weighted = masks * weights
    total = weighted.sum(axis=0, dtype=np.float32)

    if policy == OverlapPolicy.NORMALIZED:
        divisor = np.maximum(total, np.float32(1.0))
        return weighted / divisor

    if policy == OverlapPolicy.ADDITIVE_CLAMPED:
        return np.clip(weighted, 0.0, 1.0)

    if policy == OverlapPolicy.MAXIMUM_WEIGHT:
        maximum = np.clip(weighted.max(axis=0), 0.0, 1.0)
        tied = np.isclose(weighted, maximum[None, :, :], rtol=0.0, atol=MASK_EPSILON)
        tied &= maximum[None, :, :] > MASK_EPSILON
        tie_count = np.maximum(tied.sum(axis=0), 1)
        return np.where(tied, maximum[None, :, :] / tie_count[None, :, :], 0.0).astype(np.float32)

    if policy == OverlapPolicy.PRIORITY:
        alphas = np.clip(weighted, 0.0, 1.0)
        result = np.zeros_like(alphas)
        transmittance = np.ones(alphas.shape[1:], dtype=np.float32)
        ordered = sorted(range(len(regions)), key=lambda index: (regions[index].priority, str(regions[index].id)))
        for index in reversed(ordered):
            result[index] = alphas[index] * transmittance
            transmittance *= 1.0 - alphas[index]
        return result

    raise PlanError("composition.overlap_policy.unsupported", "$.composition.overlap_policy", f"Unsupported overlap policy {policy}")


def _extend_nearest(region_masks: np.ndarray, deficit: np.ndarray) -> np.ndarray:
    regional_sum = region_masks.sum(axis=0)
    covered = regional_sum > MASK_EPSILON
    if not covered.any():
        raise PlanError(
            "composition.nearest.no_regions",
            "$.composition.uncovered_policy",
            "Nearest uncovered policy requires at least one non-empty enabled region",
        )

    owners = region_masks.argmax(axis=0)
    missing = ~covered
    if missing.any():
        try:
            from scipy import ndimage
        except ImportError as error:
            raise PlanError(
                "composition.nearest.unavailable",
                "$.composition.uncovered_policy",
                "Nearest uncovered policy requires the installed SciPy runtime",
            ) from error
        nearest = ndimage.distance_transform_edt(missing, return_distances=False, return_indices=True)
        owners = owners[tuple(nearest)]

    result = region_masks.copy()
    for index in range(result.shape[0]):
        selected = owners == index
        result[index, selected] += deficit[selected]
    return result


def _freeze_array(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(contiguous.tobytes(), dtype=dtype).reshape(contiguous.shape)


def _compile_uncached(
    plan: RegionalGenerationPlan,
    key: MaskCacheKey,
    dtype: np.dtype,
) -> CompiledMaskSet:
    _, _, factor = _working_size(key.width, key.height, plan.composition.mask_antialias)
    target_width = key.width
    target_height = key.height
    enabled = tuple(region for region in plan.regions if region.enabled)
    target_pixels = target_width * target_height
    estimated_bytes = target_pixels * (max(1, len(enabled)) * 16 + 32)
    if plan.composition.uncovered_policy == UncoveredPolicy.NEAREST:
        estimated_bytes += target_pixels * 20
    if estimated_bytes > MAX_COMPILER_WORKING_BYTES:
        raise PlanError(
            "mask.compile.memory_limit",
            "$.target",
            f"Estimated mask compilation memory exceeds {MAX_COMPILER_WORKING_BYTES} bytes",
        )
    source_masks = np.stack(
        tuple(_prepare_region_mask(plan, region, target_width, target_height, factor) for region in enabled),
        axis=0,
    ) if enabled else np.empty((0, target_height, target_width), dtype=np.float32)

    active_count = (source_masks > MASK_EPSILON).sum(axis=0)
    overlap_fraction = float(np.count_nonzero(active_count > 1) / active_count.size)
    maximum_overlap = int(active_count.max(initial=0))
    region_masks = _apply_overlap_policy(source_masks, enabled, plan.composition.overlap_policy)
    regional_sum = np.clip(region_masks.sum(axis=0, dtype=np.float32), 0.0, 1.0)
    deficit = np.clip(1.0 - regional_sum, 0.0, 1.0)

    if plan.composition.uncovered_policy == UncoveredPolicy.NEAREST:
        region_masks = _extend_nearest(region_masks, deficit)
        regional_sum = np.clip(region_masks.sum(axis=0, dtype=np.float32), 0.0, 1.0)
        deficit = np.clip(1.0 - regional_sum, 0.0, 1.0)
        global_mask = np.zeros((target_height, target_width), dtype=np.float32)
    elif plan.composition.uncovered_policy == UncoveredPolicy.ERROR:
        uncovered_fraction = float(np.count_nonzero(deficit > MASK_EPSILON) / deficit.size)
        if uncovered_fraction:
            raise PlanError(
                "composition.uncovered_area",
                "$.composition.uncovered_policy",
                f"Regional masks leave {uncovered_fraction:.6%} of the target uncovered",
            )
        global_mask = np.zeros((target_height, target_width), dtype=np.float32)
    elif plan.composition.uncovered_policy == UncoveredPolicy.TRANSPARENT:
        global_mask = np.zeros((target_height, target_width), dtype=np.float32)
    else:
        global_mask = deficit

    final_sum = np.clip(region_masks.sum(axis=0, dtype=np.float32), 0.0, 1.0)
    diagnostics = []
    frozen_masks: dict[UUID, np.ndarray] = {}
    for index, region in enumerate(enabled):
        source_pixels = float(source_masks[index].sum(dtype=np.float64))
        effective_pixels = float(region_masks[index].sum(dtype=np.float64))
        diagnostics.append(
            RegionMaskDiagnostics(
                region_id=region.id,
                source_coverage=source_pixels / source_masks[index].size,
                effective_coverage=effective_pixels / region_masks[index].size,
                empty=source_pixels <= MASK_EPSILON,
                subpixel=MASK_EPSILON < source_pixels < 1.0,
                fully_occluded=source_pixels > MASK_EPSILON and effective_pixels <= MASK_EPSILON,
            )
        )
        frozen_masks[region.id] = _freeze_array(np.clip(region_masks[index], 0.0, 1.0), dtype)

    frozen_global = _freeze_array(np.clip(global_mask, 0.0, 1.0), dtype)
    return CompiledMaskSet(
        key=key,
        region_masks=frozen_masks,
        global_mask=frozen_global,
        diagnostics=MaskDiagnostics(
            regions=tuple(diagnostics),
            covered_fraction=float(np.mean(final_sum, dtype=np.float64)),
            uncovered_fraction=float(np.mean(np.clip(1.0 - final_sum, 0.0, 1.0), dtype=np.float64)),
            overlap_fraction=overlap_fraction,
            maximum_overlap=maximum_overlap,
        ),
    )


def compile_masks(
    plan: RegionalGenerationPlan,
    *,
    width: int,
    height: int,
    adapter_id: str = "unbound",
    device: str = "cpu",
    dtype: str | np.dtype = "float32",
    capabilities: ValidationCapabilities | None = None,
    cache: MaskCompilerCache | None = mask_cache,
) -> CompiledMaskSet:
    """Validate and compile immutable masks for one engine resolution."""

    target_width, target_height = _target_size(width, height)
    if device != "cpu":
        raise PlanError("mask.device.unsupported", "$.device", "The shared mask compiler only owns CPU arrays")
    resolved_dtype = np.dtype(dtype)
    if resolved_dtype not in (np.dtype("float16"), np.dtype("float32"), np.dtype("float64")):
        raise PlanError("mask.dtype.unsupported", "$.dtype", "Mask dtype must be float16, float32 or float64")

    report = validate_plan(plan, capabilities=capabilities)
    if not report.valid:
        raise PlanValidationError(report)

    key = MaskCacheKey(
        plan_hash=plan_hash(plan),
        width=target_width,
        height=target_height,
        adapter_id=str(adapter_id),
        device=device,
        dtype=resolved_dtype.name,
    )
    cached = cache.get(key) if cache is not None else None
    if cached is not None:
        return cached
    compiled = _compile_uncached(plan, key, resolved_dtype)
    return cache.put(compiled) if cache is not None else compiled


def compile_mask_pyramid(
    plan: RegionalGenerationPlan,
    resolutions: Iterable[tuple[int, int]],
    **kwargs,
) -> tuple[CompiledMaskSet, ...]:
    """Compile a deterministic set of latent or attention-grid resolutions."""

    unique = tuple(dict.fromkeys(resolutions))
    return tuple(compile_masks(plan, width=width, height=height, **kwargs) for width, height in unique)
