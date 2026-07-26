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
        object.__setattr__(self, "region_seeds", MappingProxyType(dict(self.region_seeds)))


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
