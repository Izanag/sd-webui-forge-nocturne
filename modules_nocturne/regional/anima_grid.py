"""Exact still-image patch-grid mapping for Forge's verified Anima topology."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping
from uuid import UUID

import numpy as np

from modules_nocturne.regional.errors import PlanError
from modules_nocturne.regional.masks import (
    CompiledMaskSet,
    MaskCompilerCache,
    compile_masks,
    mask_cache,
)
from modules_nocturne.regional.model import RegionalGenerationPlan
from modules_nocturne.regional.validation import ValidationCapabilities


ANIMA_VAE_SPATIAL_SCALE = 8
ANIMA_SPATIAL_PATCH = 2
ANIMA_TEMPORAL_PATCH = 1
ANIMA_NATIVE_FRAMES = 1
ANIMA_RESOLUTION_STEP = 64
ANIMA_MIN_RESOLUTION = 64
ANIMA_MAX_RESOLUTION = 2048


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PlanError(
            f"model.anima.{name}.invalid",
            f"$.{name}",
            f"Anima {name.replace('_', ' ')} must be a positive integer",
        )
    return value


@dataclass(frozen=True, slots=True)
class AnimaPatchGrid:
    canvas_width: int
    canvas_height: int
    latent_width: int
    latent_height: int
    padded_latent_width: int
    padded_latent_height: int
    patch_width: int
    patch_height: int
    frames: int = ANIMA_NATIVE_FRAMES
    flatten_order: str = "t-h-w"
    padding_mode: str = "circular-right-bottom"

    @classmethod
    def from_canvas(cls, width: int, height: int) -> "AnimaPatchGrid":
        width = _positive_integer(width, name="canvas_width")
        height = _positive_integer(height, name="canvas_height")
        for name, value in (("width", width), ("height", height)):
            if (
                value < ANIMA_MIN_RESOLUTION
                or value > ANIMA_MAX_RESOLUTION
                or value % ANIMA_RESOLUTION_STEP
            ):
                raise PlanError(
                    f"model.anima.canvas_{name}.unsupported",
                    f"$.canvas.{name}",
                    "Anima canvas dimensions must be between 64 and 2048 pixels "
                    "in 64-pixel steps",
                )
        return cls.from_latent_shape(
            width // ANIMA_VAE_SPATIAL_SCALE,
            height // ANIMA_VAE_SPATIAL_SCALE,
            canvas_width=width,
            canvas_height=height,
        )

    @classmethod
    def from_latent_shape(
        cls,
        latent_width: int,
        latent_height: int,
        *,
        canvas_width: int | None = None,
        canvas_height: int | None = None,
    ) -> "AnimaPatchGrid":
        latent_width = _positive_integer(latent_width, name="latent_width")
        latent_height = _positive_integer(latent_height, name="latent_height")
        resolved_canvas_width = (
            latent_width * ANIMA_VAE_SPATIAL_SCALE
            if canvas_width is None
            else _positive_integer(canvas_width, name="canvas_width")
        )
        resolved_canvas_height = (
            latent_height * ANIMA_VAE_SPATIAL_SCALE
            if canvas_height is None
            else _positive_integer(canvas_height, name="canvas_height")
        )
        padded_width = (
            latent_width
            + (ANIMA_SPATIAL_PATCH - latent_width % ANIMA_SPATIAL_PATCH)
            % ANIMA_SPATIAL_PATCH
        )
        padded_height = (
            latent_height
            + (ANIMA_SPATIAL_PATCH - latent_height % ANIMA_SPATIAL_PATCH)
            % ANIMA_SPATIAL_PATCH
        )
        return cls(
            canvas_width=resolved_canvas_width,
            canvas_height=resolved_canvas_height,
            latent_width=latent_width,
            latent_height=latent_height,
            padded_latent_width=padded_width,
            padded_latent_height=padded_height,
            patch_width=padded_width // ANIMA_SPATIAL_PATCH,
            patch_height=padded_height // ANIMA_SPATIAL_PATCH,
        )

    @property
    def pad_right(self) -> int:
        return self.padded_latent_width - self.latent_width

    @property
    def pad_bottom(self) -> int:
        return self.padded_latent_height - self.latent_height

    @property
    def query_count(self) -> int:
        return self.frames * self.patch_height * self.patch_width

    def query_index(self, *, x: int, y: int, t: int = 0) -> int:
        for name, value, upper in (
            ("x", x, self.patch_width),
            ("y", y, self.patch_height),
            ("t", t, self.frames),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value >= upper
            ):
                raise IndexError(f"Anima patch {name} coordinate is out of range")
        return (t * self.patch_height + y) * self.patch_width + x

    def query_coordinates(self, index: int) -> tuple[int, int, int]:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= self.query_count
        ):
            raise IndexError("Anima patch query index is out of range")
        t, spatial = divmod(index, self.patch_height * self.patch_width)
        y, x = divmod(spatial, self.patch_width)
        return t, y, x

    def flatten_mask(self, mask: np.ndarray) -> np.ndarray:
        value = np.asarray(mask)
        if value.shape != (self.patch_height, self.patch_width):
            raise ValueError(
                "Anima patch mask shape must match "
                f"{self.patch_height}x{self.patch_width}"
            )
        flattened = np.ascontiguousarray(value).reshape(self.query_count)
        flattened.flags.writeable = False
        return flattened


@dataclass(frozen=True, slots=True)
class AnimaPatchMaskSet:
    grid: AnimaPatchGrid
    compiled: CompiledMaskSet
    global_mask: np.ndarray
    region_masks: Mapping[UUID, np.ndarray]

    def __post_init__(self) -> None:
        if self.global_mask.flags.writeable or any(
            mask.flags.writeable for mask in self.region_masks.values()
        ):
            raise ValueError("Anima flattened masks must be immutable")
        object.__setattr__(
            self,
            "region_masks",
            MappingProxyType(dict(self.region_masks)),
        )


def compile_anima_patch_masks(
    plan: RegionalGenerationPlan,
    *,
    width: int,
    height: int,
    adapter_id: str = "forge-anima-still-v1",
    dtype: str = "float32",
    capabilities: ValidationCapabilities | None = None,
    cache: MaskCompilerCache | None = mask_cache,
) -> AnimaPatchMaskSet:
    grid = AnimaPatchGrid.from_canvas(width, height)
    if (plan.canvas.width, plan.canvas.height) != (width, height):
        raise PlanError(
            "model.anima.canvas_mismatch",
            "$.canvas",
            "The Anima mask target must match the canonical plan canvas",
        )
    compiled = compile_masks(
        plan,
        width=grid.patch_width,
        height=grid.patch_height,
        adapter_id=adapter_id,
        dtype=dtype,
        capabilities=capabilities,
        cache=cache,
    )
    return AnimaPatchMaskSet(
        grid=grid,
        compiled=compiled,
        global_mask=grid.flatten_mask(compiled.global_mask),
        region_masks={
            region_id: grid.flatten_mask(mask)
            for region_id, mask in compiled.region_masks.items()
        },
    )
