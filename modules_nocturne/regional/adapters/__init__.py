"""Model-family adapters with deliberately narrow topology matching."""

from modules_nocturne.regional.adapters.sd15 import (
    AttentionBlockSpec,
    AttentionGrid,
    SD15AdapterMatch,
    StrictSD15Adapter,
    sd15_adapter,
)
from modules_nocturne.regional.adapters.sdxl import (
    CONDITIONING_SPEC as SDXL_CONDITIONING_SPEC,
    SDXLAdapterMatch,
    SDXLConditioningSpec,
    StrictSDXLAdapter,
    sdxl_adapter,
)

__all__ = [
    "AttentionBlockSpec",
    "AttentionGrid",
    "SD15AdapterMatch",
    "StrictSD15Adapter",
    "sd15_adapter",
    "SDXL_CONDITIONING_SPEC",
    "SDXLAdapterMatch",
    "SDXLConditioningSpec",
    "StrictSDXLAdapter",
    "sdxl_adapter",
]
