"""Immutable domain objects for ``nocturne.regional/v1``."""

from dataclasses import dataclass, field
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias
from uuid import UUID, uuid4

CURRENT_SCHEMA = "nocturne.regional/v1"

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]


def normalise_float(value: int | float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("JSON numbers must be finite")
    return 0.0 if result == 0.0 else result


def freeze_json(value: Any) -> JsonValue:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return normalise_float(value)
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, JsonValue]:
    return MappingProxyType({str(key): freeze_json(item) for key, item in value.items()})


class OverlapPolicy(str, Enum):
    NORMALIZED = "normalized"
    PRIORITY = "priority"
    ADDITIVE_CLAMPED = "additive_clamped"
    MAXIMUM_WEIGHT = "maximum_weight"


class UncoveredPolicy(str, Enum):
    GLOBAL = "global"
    NEAREST = "nearest"
    ERROR = "error"
    TRANSPARENT = "transparent"


class SeedMode(str, Enum):
    DERIVED = "derived"
    BASE = "base"
    FIXED = "fixed"


@dataclass(frozen=True, slots=True)
class Canvas:
    width: int = 1024
    height: int = 1024
    extra: Mapping[str, JsonValue] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra", freeze_mapping(self.extra))


@dataclass(frozen=True, slots=True)
class GlobalPrompt:
    positive: str = ""
    negative: str = ""
    background_policy: str = "global"
    extra: Mapping[str, JsonValue] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra", freeze_mapping(self.extra))


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", normalise_float(self.x))
        object.__setattr__(self, "y", normalise_float(self.y))


@dataclass(frozen=True, slots=True)
class RectGeometry:
    x: float
    y: float
    width: float
    height: float
    type: str = field(default="rect", init=False)
    extra: Mapping[str, JsonValue] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", normalise_float(self.x))
        object.__setattr__(self, "y", normalise_float(self.y))
        object.__setattr__(self, "width", normalise_float(self.width))
        object.__setattr__(self, "height", normalise_float(self.height))
        object.__setattr__(self, "extra", freeze_mapping(self.extra))


@dataclass(frozen=True, slots=True)
class PolygonGeometry:
    points: tuple[Point, ...]
    type: str = field(default="polygon", init=False)
    extra: Mapping[str, JsonValue] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", tuple(self.points))
        object.__setattr__(self, "extra", freeze_mapping(self.extra))


@dataclass(frozen=True, slots=True)
class RasterMaskGeometry:
    png_base64: str
    width: int
    height: int
    sha256: str | None = None
    type: str = field(default="raster_mask", init=False)
    extra: Mapping[str, JsonValue] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))
        object.__setattr__(self, "extra", freeze_mapping(self.extra))


RegionGeometry: TypeAlias = RectGeometry | PolygonGeometry | RasterMaskGeometry


@dataclass(frozen=True, slots=True)
class GuidanceSchedule:
    start: float = 0.0
    end: float = 1.0
    extra: Mapping[str, JsonValue] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", normalise_float(self.start))
        object.__setattr__(self, "end", normalise_float(self.end))
        object.__setattr__(self, "extra", freeze_mapping(self.extra))


@dataclass(frozen=True, slots=True)
class SeedPolicy:
    mode: SeedMode = SeedMode.DERIVED
    offset: int = 0
    value: int | None = None
    extra: Mapping[str, JsonValue] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", SeedMode(self.mode))
        object.__setattr__(self, "offset", int(self.offset))
        if self.value is not None:
            object.__setattr__(self, "value", int(self.value))
        object.__setattr__(self, "extra", freeze_mapping(self.extra))


@dataclass(frozen=True, slots=True)
class Region:
    id: UUID
    name: str
    geometry: RegionGeometry
    enabled: bool = True
    locked: bool = False
    hidden: bool = False
    positive: str = ""
    negative: str = ""
    inherit_global_positive: bool = True
    inherit_global_negative: bool = True
    weight: float = 1.0
    priority: int = 0
    feather_px: float = 0.0
    guidance: GuidanceSchedule = field(default_factory=GuidanceSchedule)
    seed: SeedPolicy = field(default_factory=SeedPolicy)
    extra: Mapping[str, JsonValue] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.id, UUID):
            object.__setattr__(self, "id", UUID(str(self.id)))
        object.__setattr__(self, "weight", normalise_float(self.weight))
        object.__setattr__(self, "priority", int(self.priority))
        object.__setattr__(self, "feather_px", normalise_float(self.feather_px))
        object.__setattr__(self, "extra", freeze_mapping(self.extra))


@dataclass(frozen=True, slots=True)
class Composition:
    overlap_policy: OverlapPolicy = OverlapPolicy.NORMALIZED
    uncovered_policy: UncoveredPolicy = UncoveredPolicy.GLOBAL
    mask_antialias: bool = True
    extra: Mapping[str, JsonValue] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "overlap_policy", OverlapPolicy(self.overlap_policy))
        object.__setattr__(self, "uncovered_policy", UncoveredPolicy(self.uncovered_policy))
        object.__setattr__(self, "extra", freeze_mapping(self.extra))


@dataclass(frozen=True, slots=True)
class EngineSelection:
    requested: str = "auto"
    options: Mapping[str, JsonValue] = field(default_factory=dict)
    extra: Mapping[str, JsonValue] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", freeze_mapping(self.options))
        object.__setattr__(self, "extra", freeze_mapping(self.extra))


@dataclass(frozen=True, slots=True)
class PassPolicy:
    base: str = "regional"
    hires: str = "recompile"
    refiner: str = "preserve_when_supported"
    extra: Mapping[str, JsonValue] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra", freeze_mapping(self.extra))


@dataclass(frozen=True, slots=True)
class RegionalGenerationPlan:
    canvas: Canvas = field(default_factory=Canvas)
    global_prompt: GlobalPrompt = field(default_factory=GlobalPrompt)
    regions: tuple[Region, ...] = ()
    composition: Composition = field(default_factory=Composition)
    engine: EngineSelection = field(default_factory=EngineSelection)
    passes: PassPolicy = field(default_factory=PassPolicy)
    schema: str = CURRENT_SCHEMA
    extra: Mapping[str, JsonValue] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "regions", tuple(self.regions))
        object.__setattr__(self, "extra", freeze_mapping(self.extra))


def new_region(name: str, geometry: RegionGeometry, **kwargs: Any) -> Region:
    """Create a region with a stable random UUID."""

    return Region(id=uuid4(), name=name, geometry=geometry, **kwargs)
