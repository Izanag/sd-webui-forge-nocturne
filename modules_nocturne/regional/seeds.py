"""Deterministic image and stable-region seed derivation."""

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
from uuid import UUID

from modules_nocturne.regional.errors import PlanError
from modules_nocturne.regional.model import RegionalGenerationPlan, SeedMode

MAX_SEED = (1 << 32) - 1
_SEED_DOMAIN = b"nocturne.regional.seed/v1\0"
MIN_OFFSET = -(1 << 63)
MAX_OFFSET = (1 << 63) - 1


def _normalise_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise PlanError("seed.invalid", "$.seed", "Seed must be an integer")
    if seed < 0:
        raise PlanError("seed.unresolved", "$.seed", "Random seed values must be resolved before Regional seed derivation")
    return seed & MAX_SEED


def _normalise_offset(offset: int) -> int:
    if isinstance(offset, bool) or not isinstance(offset, int) or not MIN_OFFSET <= offset <= MAX_OFFSET:
        raise PlanError("seed.offset.invalid", "$.seed.offset", "Seed offset must be a signed 64-bit integer")
    return offset


def derive_region_seed(base_seed: int, region_id: UUID, offset: int = 0) -> int:
    image_seed = _normalise_seed(base_seed)
    seed_offset = _normalise_offset(offset)
    digest = hashlib.sha256(
        _SEED_DOMAIN
        + image_seed.to_bytes(4, "big")
        + region_id.bytes
        + seed_offset.to_bytes(8, "big", signed=True)
    ).digest()
    return int.from_bytes(digest[:8], "big") & MAX_SEED


@dataclass(frozen=True, slots=True)
class ResolvedSeedPlan:
    requested_base_seed: int
    image_seed: int
    batch_index: int
    region_seeds: Mapping[UUID, int]

    def __post_init__(self) -> None:
        if (
            isinstance(self.requested_base_seed, bool)
            or isinstance(self.image_seed, bool)
            or not isinstance(self.requested_base_seed, int)
            or not isinstance(self.image_seed, int)
            or not 0 <= self.requested_base_seed <= MAX_SEED
            or not 0 <= self.image_seed <= MAX_SEED
        ):
            raise ValueError("Resolved image seeds must be 32-bit unsigned integers")
        if isinstance(self.batch_index, bool) or not isinstance(self.batch_index, int) or self.batch_index < 0:
            raise ValueError("Resolved batch index must be a non-negative integer")
        if any(
            not isinstance(region_id, UUID)
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed <= MAX_SEED
            for region_id, seed in self.region_seeds.items()
        ):
            raise ValueError("Resolved region seeds must map UUIDs to 32-bit unsigned integers")
        object.__setattr__(self, "region_seeds", MappingProxyType(dict(self.region_seeds)))


@dataclass(frozen=True, slots=True)
class ResolvedSeedBatch:
    batch_size: int
    batch_count: int
    images: tuple[ResolvedSeedPlan, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "images", tuple(self.images))
        if (
            isinstance(self.batch_size, bool)
            or isinstance(self.batch_count, bool)
            or not isinstance(self.batch_size, int)
            or not isinstance(self.batch_count, int)
            or self.batch_size < 1
            or self.batch_count < 1
            or len(self.images) != self.batch_size * self.batch_count
        ):
            raise ValueError("Resolved seed batch dimensions must match its image records")


def resolve_seed_plan(plan: RegionalGenerationPlan, base_seed: int, batch_index: int = 0) -> ResolvedSeedPlan:
    requested_base_seed = _normalise_seed(base_seed)
    if isinstance(batch_index, bool) or not isinstance(batch_index, int) or batch_index < 0:
        raise PlanError("seed.batch_index.invalid", "$.batch_index", "Batch index must be a non-negative integer")

    image_seed = (requested_base_seed + batch_index) & MAX_SEED
    region_seeds: dict[UUID, int] = {}

    for region in plan.regions:
        if region.seed.mode == SeedMode.DERIVED:
            resolved = derive_region_seed(image_seed, region.id, region.seed.offset)
        elif region.seed.mode == SeedMode.BASE:
            resolved = (image_seed + _normalise_offset(region.seed.offset)) & MAX_SEED
        elif region.seed.mode == SeedMode.FIXED:
            if region.seed.value is None:
                raise PlanError("seed.fixed_value.missing", "$.regions", f"Region {region.id} requires a fixed seed value")
            resolved = (_normalise_seed(region.seed.value) + _normalise_offset(region.seed.offset)) & MAX_SEED
        else:
            raise PlanError("seed.mode.unsupported", "$.regions", f"Unsupported seed mode {region.seed.mode!r}")
        region_seeds[region.id] = resolved

    return ResolvedSeedPlan(
        requested_base_seed=requested_base_seed,
        image_seed=image_seed,
        batch_index=batch_index,
        region_seeds=region_seeds,
    )


def resolve_seed_batch(
    plan: RegionalGenerationPlan,
    base_seed: int,
    *,
    batch_size: int,
    batch_count: int,
    increment_seed: bool = True,
) -> ResolvedSeedBatch:
    """Match Forge's flattened image order after its random seed is resolved."""

    for name, value in (("batch_size", batch_size), ("batch_count", batch_count)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise PlanError(f"seed.{name}.invalid", f"$.{name}", f"{name} must be a positive integer")
    image_count = batch_size * batch_count
    if image_count > 10_000:
        raise PlanError("seed.batch.too_large", "$", "Seed batch cannot exceed 10000 images")
    resolved_base = _normalise_seed(base_seed)
    images = tuple(
        resolve_seed_plan(plan, resolved_base, image_index if increment_seed else 0)
        for image_index in range(image_count)
    )
    return ResolvedSeedBatch(
        batch_size=batch_size,
        batch_count=batch_count,
        images=images,
    )


def resolve_forge_seed_sequence(
    plan: RegionalGenerationPlan,
    image_seeds: tuple[int, ...] | list[int],
) -> tuple[ResolvedSeedPlan, ...]:
    """Use Forge's finalized ``all_seeds`` without recreating its random state."""

    if not image_seeds:
        raise PlanError("seed.sequence.empty", "$.all_seeds", "Forge seed sequence cannot be empty")
    return tuple(
        _resolve_exact_image_seed(plan, _normalise_seed(seed), index)
        for index, seed in enumerate(image_seeds)
    )


def _resolve_exact_image_seed(plan: RegionalGenerationPlan, image_seed: int, batch_index: int) -> ResolvedSeedPlan:
    resolved = resolve_seed_plan(plan, image_seed, batch_index=0)
    return ResolvedSeedPlan(
        requested_base_seed=image_seed,
        image_seed=resolved.image_seed,
        batch_index=batch_index,
        region_seeds=resolved.region_seeds,
    )
