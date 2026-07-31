"""Strict adapter fingerprint for Forge's conventional SD 1.5 U-Net."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from modules_nocturne.regional.errors import PlanError

ENGINE_TYPE = "backend.diffusion_engine.sd15.StableDiffusion"
MODEL_CONFIG_TYPE = "huggingface_guess.model_list.SD15"
PATCHER_TYPE = "backend.patcher.unet.UnetPatcher"
K_MODEL_TYPE = "backend.modules.k_model.KModel"
UNET_TYPE = "backend.nn.unet.IntegratedUNet2DConditionModel"
SPATIAL_TRANSFORMER_TYPE = "backend.nn.unet.SpatialTransformer"
DOWNSAMPLE_TYPE = "backend.nn.unet.Downsample"
UPSAMPLE_TYPE = "backend.nn.unet.Upsample"


def _type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


@dataclass(frozen=True, slots=True)
class AttentionBlockSpec:
    block_kind: str
    block_index: int
    transformer_index: int
    downsample_factor: int
    channels: int
    heads: int
    head_dim: int
    context_dim: int

    @property
    def identity(self) -> tuple[str, int, int]:
        return self.block_kind, self.block_index, self.transformer_index


@dataclass(frozen=True, slots=True)
class AttentionGrid:
    block: AttentionBlockSpec
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class SD15AdapterMatch:
    matched: bool
    reason_code: str | None
    reason: str | None
    attention_blocks: tuple[AttentionBlockSpec, ...] = ()


EXPECTED_ATTENTION_BLOCKS = (
    AttentionBlockSpec("input", 1, 0, 1, 320, 8, 40, 768),
    AttentionBlockSpec("input", 2, 0, 1, 320, 8, 40, 768),
    AttentionBlockSpec("input", 4, 0, 2, 640, 8, 80, 768),
    AttentionBlockSpec("input", 5, 0, 2, 640, 8, 80, 768),
    AttentionBlockSpec("input", 7, 0, 4, 1280, 8, 160, 768),
    AttentionBlockSpec("input", 8, 0, 4, 1280, 8, 160, 768),
    AttentionBlockSpec("middle", 0, 0, 8, 1280, 8, 160, 768),
    AttentionBlockSpec("output", 3, 0, 4, 1280, 8, 160, 768),
    AttentionBlockSpec("output", 4, 0, 4, 1280, 8, 160, 768),
    AttentionBlockSpec("output", 5, 0, 4, 1280, 8, 160, 768),
    AttentionBlockSpec("output", 6, 0, 2, 640, 8, 80, 768),
    AttentionBlockSpec("output", 7, 0, 2, 640, 8, 80, 768),
    AttentionBlockSpec("output", 8, 0, 2, 640, 8, 80, 768),
    AttentionBlockSpec("output", 9, 0, 1, 320, 8, 40, 768),
    AttentionBlockSpec("output", 10, 0, 1, 320, 8, 40, 768),
    AttentionBlockSpec("output", 11, 0, 1, 320, 8, 40, 768),
)


class _TopologyMismatch(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _require_attribute(value: Any, name: str, code: str) -> Any:
    try:
        return getattr(value, name)
    except (AttributeError, TypeError) as error:
        raise _TopologyMismatch(code, f"Missing required SD 1.5 topology field {name!r}") from error


def _spatial_transformers(sequence) -> tuple[Any, ...]:
    try:
        return tuple(layer for layer in sequence if _type_name(layer) == SPATIAL_TRANSFORMER_TYPE)
    except TypeError as error:
        raise _TopologyMismatch("model.sd15.blocks.invalid", "A U-Net block is not iterable") from error


def _has_layer(sequence, type_name: str) -> bool:
    return any(_type_name(layer) == type_name for layer in sequence)


def _attention_specs(unet: Any) -> tuple[AttentionBlockSpec, ...]:
    input_blocks = _require_attribute(unet, "input_blocks", "model.sd15.input_blocks.missing")
    middle_block = _require_attribute(unet, "middle_block", "model.sd15.middle_block.missing")
    output_blocks = _require_attribute(unet, "output_blocks", "model.sd15.output_blocks.missing")
    if len(input_blocks) != 12 or len(middle_block) != 3 or len(output_blocks) != 12:
        raise _TopologyMismatch(
            "model.sd15.block_count_mismatch",
            "The SD 1.5 U-Net block counts differ from the verified Forge topology",
        )

    specs: list[AttentionBlockSpec] = []

    def append_spatial(block_kind: str, block_index: int, factor: int, sequence) -> None:
        for spatial in _spatial_transformers(sequence):
            transformer_blocks = _require_attribute(
                spatial,
                "transformer_blocks",
                "model.sd15.transformer_blocks.missing",
            )
            for transformer_index, transformer in enumerate(transformer_blocks):
                attn2 = _require_attribute(transformer, "attn2", "model.sd15.cross_attention.missing")
                to_k = _require_attribute(attn2, "to_k", "model.sd15.cross_attention_key.missing")
                specs.append(
                    AttentionBlockSpec(
                        block_kind=block_kind,
                        block_index=block_index,
                        transformer_index=transformer_index,
                        downsample_factor=factor,
                        channels=int(_require_attribute(spatial, "in_channels", "model.sd15.channels.missing")),
                        heads=int(_require_attribute(transformer, "n_heads", "model.sd15.heads.missing")),
                        head_dim=int(_require_attribute(transformer, "d_head", "model.sd15.head_dim.missing")),
                        context_dim=int(_require_attribute(to_k, "in_features", "model.sd15.context_dim.missing")),
                    )
                )

    factor = 1
    for index, block in enumerate(input_blocks):
        append_spatial("input", index, factor, block)
        if _has_layer(block, DOWNSAMPLE_TYPE):
            factor *= 2
    append_spatial("middle", 0, factor, middle_block)
    for index, block in enumerate(output_blocks):
        append_spatial("output", index, factor, block)
        if _has_layer(block, UPSAMPLE_TYPE):
            factor //= 2
    return tuple(specs)


class StrictSD15Adapter:
    """Match only the exact Forge SD 1.5 topology measured by this build."""

    adapter_id = "forge-sd15-v1"
    adapter_version = "1.0.0"

    def diagnose(self, model_context: Any) -> SD15AdapterMatch:
        try:
            if model_context is None:
                raise _TopologyMismatch("model.not_loaded", "No model is loaded")
            if _type_name(model_context) != ENGINE_TYPE:
                raise _TopologyMismatch(
                    "model.sd15.engine_type_mismatch",
                    "The loaded diffusion engine is not Forge's verified SD 1.5 engine",
                )
            if (
                _require_attribute(model_context, "is_sd1", "model.sd15.flag_missing") is not True
                or _require_attribute(model_context, "is_sdxl", "model.sd15.flag_missing") is not False
            ):
                raise _TopologyMismatch("model.sd15.family_flags_mismatch", "The loaded model family flags are not strict SD 1.5")
            if _require_attribute(model_context, "is_inpaint", "model.sd15.inpaint_flag_missing") is not False:
                raise _TopologyMismatch(
                    "model.sd15.inpaint_unproven",
                    "SD 1.5 inpainting U-Nets are not enabled by the strict baseline adapter",
                )

            model_config = _require_attribute(model_context, "model_config", "model.sd15.config_missing")
            if _type_name(model_config) != MODEL_CONFIG_TYPE:
                raise _TopologyMismatch(
                    "model.sd15.config_type_mismatch",
                    "The model configuration is not the verified SD 1.5 configuration",
                )
            model_type = _require_attribute(model_config, "model_type", "model.sd15.prediction_type_missing")
            if _require_attribute(model_type, "name", "model.sd15.prediction_type_missing") != "EPS":
                raise _TopologyMismatch(
                    "model.sd15.prediction_type_mismatch",
                    "Only the verified EPS prediction type is enabled",
                )

            forge_objects = _require_attribute(model_context, "forge_objects", "model.sd15.forge_objects_missing")
            patcher = _require_attribute(forge_objects, "unet", "model.sd15.unet_patcher_missing")
            if _type_name(patcher) != PATCHER_TYPE:
                raise _TopologyMismatch(
                    "model.sd15.patcher_type_mismatch",
                    "The active U-Net patcher is not the verified Forge patcher",
                )
            k_model = _require_attribute(patcher, "model", "model.sd15.k_model_missing")
            if _type_name(k_model) != K_MODEL_TYPE:
                raise _TopologyMismatch("model.sd15.k_model_type_mismatch", "The active denoiser wrapper is not verified")
            unet = _require_attribute(k_model, "diffusion_model", "model.sd15.unet_missing")
            if _type_name(unet) != UNET_TYPE:
                raise _TopologyMismatch(
                    "model.sd15.unet_type_mismatch",
                    "The active U-Net implementation is not the verified Forge SD 1.5 U-Net",
                )
            if (
                _require_attribute(unet, "model_channels", "model.sd15.model_channels_missing") != 320
                or _require_attribute(unet, "in_channels", "model.sd15.in_channels_missing") != 4
                or _require_attribute(unet, "out_channels", "model.sd15.out_channels_missing") != 4
            ):
                raise _TopologyMismatch(
                    "model.sd15.channel_layout_mismatch",
                    "The U-Net channel layout differs from the verified SD 1.5 baseline",
                )

            vae = _require_attribute(forge_objects, "vae", "model.sd15.vae_missing")
            if (
                _require_attribute(vae, "latent_channels", "model.sd15.vae_latent_channels_missing") != 4
                or _require_attribute(vae, "upscale_ratio", "model.sd15.vae_scale_missing") != 8
            ):
                raise _TopologyMismatch(
                    "model.sd15.vae_layout_mismatch",
                    "The VAE latent layout differs from the verified SD 1.5 baseline",
                )

            attention_blocks = _attention_specs(unet)
            if attention_blocks != EXPECTED_ATTENTION_BLOCKS:
                raise _TopologyMismatch(
                    "model.sd15.attention_topology_mismatch",
                    "Cross-attention blocks differ from the verified SD 1.5 topology",
                )
            return SD15AdapterMatch(
                matched=True,
                reason_code=None,
                reason=None,
                attention_blocks=attention_blocks,
            )
        except _TopologyMismatch as mismatch:
            return SD15AdapterMatch(
                matched=False,
                reason_code=mismatch.code,
                reason=str(mismatch),
            )
        except Exception:
            return SD15AdapterMatch(
                matched=False,
                reason_code="model.sd15.inspection_failed",
                reason="The loaded model topology could not be inspected safely",
            )

    def matches(self, model_context: Any) -> bool:
        return self.diagnose(model_context).matched

    def supported_engine_ids(self) -> frozenset[str]:
        return frozenset({"attention-decomposition", "denoising-fusion"})

    def unsupported_fields(self) -> frozenset[str]:
        return frozenset(
            {
                "passes.refiner",
                "composition.uncovered_policy.transparent",
            }
        )

    def expected_fallbacks(self) -> tuple[str, ...]:
        return ()

    def attention_blocks(self, model_context: Any) -> tuple[AttentionBlockSpec, ...]:
        match = self.diagnose(model_context)
        if not match.matched:
            raise PlanError(
                match.reason_code or "model.sd15.unsupported",
                "$.engine",
                match.reason or "The loaded model is not supported by the strict SD 1.5 adapter",
            )
        return match.attention_blocks

    def cross_attention_modules(self, model_context: Any) -> Mapping[tuple[str, int, int], Any]:
        blocks = self.attention_blocks(model_context)
        unet = model_context.forge_objects.unet.model.diffusion_model
        result = {}
        for block in blocks:
            sequence = (
                unet.input_blocks[block.block_index]
                if block.block_kind == "input"
                else unet.output_blocks[block.block_index]
                if block.block_kind == "output"
                else unet.middle_block
            )
            spatial = _spatial_transformers(sequence)
            if len(spatial) != 1 or block.transformer_index >= len(spatial[0].transformer_blocks):
                raise PlanError(
                    "model.sd15.attention_module_mismatch",
                    "$.engine",
                    "The verified SD 1.5 cross-attention module could not be resolved",
                )
            result[block.identity] = spatial[0].transformer_blocks[block.transformer_index].attn2
        return MappingProxyType(result)

    def attention_grids(self, model_context: Any, *, width: int, height: int) -> tuple[AttentionGrid, ...]:
        for name, value in (("width", width), ("height", height)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 64 or value % 8:
                raise PlanError(
                    f"model.sd15.{name}.invalid",
                    f"$.canvas.{name}",
                    f"SD 1.5 canvas {name} must be an integer of at least 64 pixels and divisible by 8",
                )
        latent_width = width // 8
        latent_height = height // 8
        return tuple(
            AttentionGrid(
                block=block,
                width=(latent_width + block.downsample_factor - 1) // block.downsample_factor,
                height=(latent_height + block.downsample_factor - 1) // block.downsample_factor,
            )
            for block in self.attention_blocks(model_context)
        )

    def supported_conditioning_branches(self) -> frozenset[str]:
        return frozenset({"positive", "negative"})

    def validate_conditioning_value(self, value: Any) -> None:
        import torch

        if (
            not torch.is_tensor(value)
            or value.ndim != 2
            or value.shape[-1] != 768
        ):
            raise RuntimeError(
                "SD 1.5 Regional conditioning must be a 768-wide two-dimensional tensor"
            )

    def cross_attention_context(self, value: Any) -> Any:
        self.validate_conditioning_value(value)
        return value

    def sampler_hooks(self) -> frozenset[str]:
        return frozenset({"attn2_replace"})


sd15_adapter = StrictSD15Adapter()
