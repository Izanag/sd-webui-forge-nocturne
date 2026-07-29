"""Strict adapter fingerprint for Forge's conventional SDXL base U-Net."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from modules_nocturne.regional.adapters.sd15 import (
    AttentionBlockSpec,
    AttentionGrid,
)
from modules_nocturne.regional.errors import PlanError

ENGINE_TYPE = "backend.diffusion_engine.sdxl.StableDiffusionXL"
MODEL_CONFIG_TYPE = "huggingface_guess.model_list.SDXL"
PATCHER_TYPE = "backend.patcher.unet.UnetPatcher"
K_MODEL_TYPE = "backend.modules.k_model.KModel"
UNET_TYPE = "backend.nn.unet.IntegratedUNet2DConditionModel"
SPATIAL_TRANSFORMER_TYPE = "backend.nn.unet.SpatialTransformer"
DOWNSAMPLE_TYPE = "backend.nn.unet.Downsample"
UPSAMPLE_TYPE = "backend.nn.unet.Upsample"
TEXT_PROCESSING_ENGINE_TYPE = "backend.text_processing.classic_engine.ClassicTextProcessingEngine"
TIMESTEP_TYPE = "backend.nn.unet.Timestep"


def _type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


@dataclass(frozen=True, slots=True)
class SDXLConditioningSpec:
    text_encoder_keys: tuple[str, str]
    cross_attention_dim: int
    pooled_dim: int
    vector_dim: int


@dataclass(frozen=True, slots=True)
class SDXLAdapterMatch:
    matched: bool
    reason_code: str | None
    reason: str | None
    attention_blocks: tuple[AttentionBlockSpec, ...] = ()
    conditioning: SDXLConditioningSpec | None = None


CONDITIONING_SPEC = SDXLConditioningSpec(
    text_encoder_keys=("clip_l", "clip_g"),
    cross_attention_dim=2048,
    pooled_dim=1280,
    vector_dim=2816,
)


def _expected_specs(
    block_kind: str,
    block_index: int,
    transformer_count: int,
    downsample_factor: int,
    channels: int,
) -> tuple[AttentionBlockSpec, ...]:
    heads = channels // 64
    return tuple(
        AttentionBlockSpec(
            block_kind=block_kind,
            block_index=block_index,
            transformer_index=transformer_index,
            downsample_factor=downsample_factor,
            channels=channels,
            heads=heads,
            head_dim=64,
            context_dim=CONDITIONING_SPEC.cross_attention_dim,
        )
        for transformer_index in range(transformer_count)
    )


EXPECTED_ATTENTION_BLOCKS = (
    *_expected_specs("input", 4, 2, 2, 640),
    *_expected_specs("input", 5, 2, 2, 640),
    *_expected_specs("input", 7, 10, 4, 1280),
    *_expected_specs("input", 8, 10, 4, 1280),
    *_expected_specs("middle", 0, 10, 4, 1280),
    *_expected_specs("output", 0, 10, 4, 1280),
    *_expected_specs("output", 1, 10, 4, 1280),
    *_expected_specs("output", 2, 10, 4, 1280),
    *_expected_specs("output", 3, 2, 2, 640),
    *_expected_specs("output", 4, 2, 2, 640),
    *_expected_specs("output", 5, 2, 2, 640),
)


class _TopologyMismatch(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _require_attribute(value: Any, name: str, code: str) -> Any:
    try:
        return getattr(value, name)
    except (AttributeError, TypeError) as error:
        raise _TopologyMismatch(code, f"Missing required SDXL topology field {name!r}") from error


def _spatial_transformers(sequence) -> tuple[Any, ...]:
    try:
        return tuple(layer for layer in sequence if _type_name(layer) == SPATIAL_TRANSFORMER_TYPE)
    except TypeError as error:
        raise _TopologyMismatch("model.sdxl.blocks.invalid", "A U-Net block is not iterable") from error


def _has_layer(sequence, type_name: str) -> bool:
    return any(_type_name(layer) == type_name for layer in sequence)


def _attention_specs(unet: Any) -> tuple[AttentionBlockSpec, ...]:
    input_blocks = _require_attribute(unet, "input_blocks", "model.sdxl.input_blocks.missing")
    middle_block = _require_attribute(unet, "middle_block", "model.sdxl.middle_block.missing")
    output_blocks = _require_attribute(unet, "output_blocks", "model.sdxl.output_blocks.missing")
    if len(input_blocks) != 9 or len(middle_block) != 3 or len(output_blocks) != 9:
        raise _TopologyMismatch(
            "model.sdxl.block_count_mismatch",
            "The SDXL U-Net block counts differ from the verified Forge topology",
        )

    specs: list[AttentionBlockSpec] = []

    def append_spatial(block_kind: str, block_index: int, factor: int, sequence) -> None:
        for spatial in _spatial_transformers(sequence):
            transformer_blocks = _require_attribute(
                spatial,
                "transformer_blocks",
                "model.sdxl.transformer_blocks.missing",
            )
            for transformer_index, transformer in enumerate(transformer_blocks):
                attn2 = _require_attribute(transformer, "attn2", "model.sdxl.cross_attention.missing")
                to_k = _require_attribute(attn2, "to_k", "model.sdxl.cross_attention_key.missing")
                specs.append(
                    AttentionBlockSpec(
                        block_kind=block_kind,
                        block_index=block_index,
                        transformer_index=transformer_index,
                        downsample_factor=factor,
                        channels=int(_require_attribute(spatial, "in_channels", "model.sdxl.channels.missing")),
                        heads=int(_require_attribute(transformer, "n_heads", "model.sdxl.heads.missing")),
                        head_dim=int(_require_attribute(transformer, "d_head", "model.sdxl.head_dim.missing")),
                        context_dim=int(_require_attribute(to_k, "in_features", "model.sdxl.context_dim.missing")),
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


def _validate_text_conditioning(model_context: Any) -> SDXLConditioningSpec:
    local = _require_attribute(
        model_context,
        "text_processing_engine_l",
        "model.sdxl.text_encoder_l.missing",
    )
    global_encoder = _require_attribute(
        model_context,
        "text_processing_engine_g",
        "model.sdxl.text_encoder_g.missing",
    )
    if (
        _type_name(local) != TEXT_PROCESSING_ENGINE_TYPE
        or _type_name(global_encoder) != TEXT_PROCESSING_ENGINE_TYPE
    ):
        raise _TopologyMismatch(
            "model.sdxl.text_engine_type_mismatch",
            "SDXL must use Forge's verified dual Classic text-processing engines",
        )
    if (
        _require_attribute(local, "embedding_key", "model.sdxl.text_encoder_l.invalid") != "clip_l"
        or _require_attribute(local, "return_pooled", "model.sdxl.text_encoder_l.invalid") is not False
        or _require_attribute(local, "text_projection", "model.sdxl.text_encoder_l.invalid") is not False
    ):
        raise _TopologyMismatch(
            "model.sdxl.text_encoder_l.invalid",
            "The SDXL local text encoder contract differs from Forge's verified CLIP-L path",
        )
    if (
        _require_attribute(global_encoder, "embedding_key", "model.sdxl.text_encoder_g.invalid") != "clip_g"
        or _require_attribute(global_encoder, "return_pooled", "model.sdxl.text_encoder_g.invalid") is not True
        or _require_attribute(global_encoder, "text_projection", "model.sdxl.text_encoder_g.invalid") is not True
    ):
        raise _TopologyMismatch(
            "model.sdxl.text_encoder_g.invalid",
            "The SDXL global text encoder contract differs from Forge's verified pooled CLIP-G path",
        )
    if _require_attribute(local, "text_encoder", "model.sdxl.text_encoder_l.invalid") is _require_attribute(
        global_encoder,
        "text_encoder",
        "model.sdxl.text_encoder_g.invalid",
    ):
        raise _TopologyMismatch(
            "model.sdxl.text_encoders_aliased",
            "SDXL requires distinct CLIP-L and CLIP-G encoder objects",
        )
    embedder = _require_attribute(model_context, "embedder", "model.sdxl.embedder.missing")
    if _type_name(embedder) != TIMESTEP_TYPE:
        raise _TopologyMismatch(
            "model.sdxl.embedder_type_mismatch",
            "The SDXL pooled-conditioning size embedder is not Forge's verified implementation",
        )
    return CONDITIONING_SPEC


class StrictSDXLAdapter:
    """Match only the exact Forge SDXL base topology measured by this build."""

    adapter_id = "forge-sdxl-base-v1"
    adapter_version = "1.0.0"

    def diagnose(self, model_context: Any) -> SDXLAdapterMatch:
        try:
            if model_context is None:
                raise _TopologyMismatch("model.not_loaded", "No model is loaded")
            if _type_name(model_context) != ENGINE_TYPE:
                raise _TopologyMismatch(
                    "model.sdxl.engine_type_mismatch",
                    "The loaded diffusion engine is not Forge's verified SDXL base engine",
                )
            if (
                _require_attribute(model_context, "is_sd1", "model.sdxl.flag_missing") is not False
                or _require_attribute(model_context, "is_sdxl", "model.sdxl.flag_missing") is not True
            ):
                raise _TopologyMismatch("model.sdxl.family_flags_mismatch", "The loaded model family flags are not strict SDXL")
            if _require_attribute(model_context, "is_inpaint", "model.sdxl.inpaint_flag_missing") is not False:
                raise _TopologyMismatch(
                    "model.sdxl.inpaint_unproven",
                    "SDXL inpainting U-Nets are not enabled by the strict base adapter",
                )

            model_config = _require_attribute(model_context, "model_config", "model.sdxl.config_missing")
            if _type_name(model_config) != MODEL_CONFIG_TYPE:
                raise _TopologyMismatch(
                    "model.sdxl.config_type_mismatch",
                    "The model configuration is not Forge's verified SDXL base configuration",
                )
            model_type = _require_attribute(model_config, "model_type", "model.sdxl.prediction_type_missing")
            if _require_attribute(model_type, "name", "model.sdxl.prediction_type_missing") != "EPS":
                raise _TopologyMismatch(
                    "model.sdxl.prediction_type_mismatch",
                    "Only the verified SDXL EPS prediction type is enabled",
                )

            conditioning = _validate_text_conditioning(model_context)
            forge_objects = _require_attribute(model_context, "forge_objects", "model.sdxl.forge_objects_missing")
            patcher = _require_attribute(forge_objects, "unet", "model.sdxl.unet_patcher_missing")
            if _type_name(patcher) != PATCHER_TYPE:
                raise _TopologyMismatch(
                    "model.sdxl.patcher_type_mismatch",
                    "The active U-Net patcher is not Forge's verified patcher",
                )
            k_model = _require_attribute(patcher, "model", "model.sdxl.k_model_missing")
            if _type_name(k_model) != K_MODEL_TYPE:
                raise _TopologyMismatch("model.sdxl.k_model_type_mismatch", "The active denoiser wrapper is not verified")
            unet = _require_attribute(k_model, "diffusion_model", "model.sdxl.unet_missing")
            if _type_name(unet) != UNET_TYPE:
                raise _TopologyMismatch(
                    "model.sdxl.unet_type_mismatch",
                    "The active U-Net implementation is not Forge's verified SDXL U-Net",
                )
            if (
                _require_attribute(unet, "model_channels", "model.sdxl.model_channels_missing") != 320
                or _require_attribute(unet, "in_channels", "model.sdxl.in_channels_missing") != 4
                or _require_attribute(unet, "out_channels", "model.sdxl.out_channels_missing") != 4
                or _require_attribute(unet, "num_classes", "model.sdxl.num_classes_missing") != "sequential"
            ):
                raise _TopologyMismatch(
                    "model.sdxl.channel_layout_mismatch",
                    "The U-Net channel and pooled-conditioning layout differs from the verified SDXL base",
                )

            vae = _require_attribute(forge_objects, "vae", "model.sdxl.vae_missing")
            if (
                _require_attribute(vae, "latent_channels", "model.sdxl.vae_latent_channels_missing") != 4
                or _require_attribute(vae, "upscale_ratio", "model.sdxl.vae_scale_missing") != 8
            ):
                raise _TopologyMismatch(
                    "model.sdxl.vae_layout_mismatch",
                    "The VAE latent layout differs from the verified SDXL base",
                )

            attention_blocks = _attention_specs(unet)
            if attention_blocks != EXPECTED_ATTENTION_BLOCKS:
                raise _TopologyMismatch(
                    "model.sdxl.attention_topology_mismatch",
                    "Cross-attention blocks differ from the verified SDXL base topology",
                )
            return SDXLAdapterMatch(
                matched=True,
                reason_code=None,
                reason=None,
                attention_blocks=attention_blocks,
                conditioning=conditioning,
            )
        except _TopologyMismatch as mismatch:
            return SDXLAdapterMatch(
                matched=False,
                reason_code=mismatch.code,
                reason=str(mismatch),
            )
        except Exception:
            return SDXLAdapterMatch(
                matched=False,
                reason_code="model.sdxl.inspection_failed",
                reason="The loaded SDXL topology could not be inspected safely",
            )

    def matches(self, model_context: Any) -> bool:
        return self.diagnose(model_context).matched

    def supported_engine_ids(self) -> frozenset[str]:
        return frozenset({"attention-decomposition"})

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
                match.reason_code or "model.sdxl.unsupported",
                "$.engine",
                match.reason or "The loaded model is not supported by the strict SDXL adapter",
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
                    "model.sdxl.attention_module_mismatch",
                    "$.engine",
                    "The verified SDXL cross-attention module could not be resolved",
                )
            result[block.identity] = spatial[0].transformer_blocks[block.transformer_index].attn2
        return MappingProxyType(result)

    def attention_grids(self, model_context: Any, *, width: int, height: int) -> tuple[AttentionGrid, ...]:
        for name, value in (("width", width), ("height", height)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 64 or value % 8:
                raise PlanError(
                    f"model.sdxl.{name}.invalid",
                    f"$.canvas.{name}",
                    f"SDXL canvas {name} must be an integer of at least 64 pixels and divisible by 8",
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

    def validate_conditioning_value(self, value: Any) -> None:
        try:
            import torch

            cross_attention = value["crossattn"]
            vector = value["vector"]
            valid = (
                set(value) == {"crossattn", "vector"}
                and torch.is_tensor(cross_attention)
                and cross_attention.ndim == 2
                and cross_attention.shape[-1] == CONDITIONING_SPEC.cross_attention_dim
                and torch.is_tensor(vector)
                and vector.ndim == 1
                and vector.shape[-1] == CONDITIONING_SPEC.vector_dim
            )
        except (KeyError, TypeError):
            valid = False
        if not valid:
            raise RuntimeError(
                "SDXL Regional conditioning must preserve 2048-wide dual-encoder "
                "cross-attention and the 2816-wide pooled/vector conditioning"
            )

    def cross_attention_context(self, value: Any) -> Any:
        self.validate_conditioning_value(value)
        return value["crossattn"]

    def supported_conditioning_branches(self) -> frozenset[str]:
        return frozenset({"positive", "negative"})

    def sampler_hooks(self) -> frozenset[str]:
        return frozenset({"attn2_replace"})


sdxl_adapter = StrictSDXLAdapter()
