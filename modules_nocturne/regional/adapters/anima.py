"""Strict topology fingerprint for Forge's still-image Anima implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping


ENGINE_TYPE = "backend.diffusion_engine.anima.Anima"
MODEL_CONFIG_TYPE = "huggingface_guess.model_list.Anima"
PATCHER_TYPE = "backend.patcher.unet.UnetPatcher"
K_MODEL_TYPE = "backend.modules.k_model.KModel"
DIFFUSION_TYPE = "backend.nn.anima.Anima"
BLOCK_TYPE = "backend.nn.anima.Block"
CROSS_ATTENTION_TYPE = "backend.nn.anima.SelfCrossAttention"
CLIP_TYPE = "backend.patcher.clip.CLIP"
JOINT_TEXT_ENCODER_TYPE = "backend.patcher.clip.JointTextEncoder"
TEXT_ENGINE_TYPE = "backend.text_processing.anima_engine.AnimaTextProcessingEngine"
QWEN_TYPE = "backend.nn.llm.llama.Qwen3_06B"
LLM_ADAPTER_TYPE = "backend.nn.anima.LLMAdapter"
VAE_TYPE = "backend.patcher.vae.VAE"
VAE_MODEL_TYPE = "backend.nn.wan_vae.WanVAE"


def _type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


@dataclass(frozen=True, slots=True)
class AnimaArchitectureFacts:
    family: str = "anima"
    temporal_frames: int = 1
    latent_channels: int = 16
    patch_spatial: int = 2
    patch_temporal: int = 1
    model_channels: int = 2048
    transformer_blocks: int = 28
    cross_attention_dim: int = 1024
    attention_heads: int = 16
    attention_head_dim: int = 128
    llm_adapter_blocks: int = 6
    native_width: int = 1024
    native_height: int = 1024
    resolution_step: int = 64

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "adapter_version": "1.0.0",
                "conditioning": "qwen3-0.6b-plus-t5-token-ids",
                "prediction": "discrete-flow",
                "vae": "wan-vae",
                "supported_inference_dtypes": ["bfloat16", "float16", "float32"],
                "verified_attention_backend": "pytorch",
                "sampling_defaults": {
                    "sampler": "ER SDE",
                    "scheduler": "Beta",
                    "steps": 32,
                    "cfg_scale": 4.0,
                    "shift": 3.0,
                },
                "capabilities": {
                    "regional_generation": "engine-unavailable",
                    "cfg": "unverified",
                    "negative_prompt": "unverified",
                    "hires": "blocked-unverified",
                    "edit": "blocked-unverified",
                    "controlnet": "blocked-unverified",
                    "refiner": "unsupported",
                },
                "resolution": {
                    "native": [self.native_width, self.native_height],
                    "step": self.resolution_step,
                    "minimum": 64,
                    "maximum": 2048,
                },
            }
        )
        return result


@dataclass(frozen=True, slots=True)
class AnimaAdapterMatch:
    matched: bool
    reason_code: str | None
    reason: str | None
    facts: AnimaArchitectureFacts | None = None


EXPECTED_FACTS = AnimaArchitectureFacts()


class _TopologyMismatch(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _require_attribute(value: Any, name: str, code: str) -> Any:
    try:
        return getattr(value, name)
    except (AttributeError, TypeError) as error:
        raise _TopologyMismatch(
            code,
            f"Missing required Anima topology field {name!r}",
        ) from error


def _require_type(value: Any, expected: str, code: str, label: str) -> None:
    if _type_name(value) != expected:
        raise _TopologyMismatch(code, f"The active {label} is not Forge's verified Anima implementation")


def _linear_shape(value: Any, code: str) -> tuple[int, int]:
    return (
        int(_require_attribute(value, "out_features", code)),
        int(_require_attribute(value, "in_features", code)),
    )


def _shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", value if isinstance(value, (tuple, list)) else None)
    if shape is None:
        return None
    try:
        return tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError):
        return None


_BLOCK_KEY = re.compile(r"^blocks\.(\d+)\.mlp\.layer1\.weight$")
_LLM_BLOCK_KEY = re.compile(
    r"^llm_adapter\.blocks\.(\d+)\.cross_attn\.q_proj\.weight$"
)


def diagnose_anima_state_dict(state_dict: Mapping[str, Any]) -> AnimaAdapterMatch:
    """Validate the architecture-bearing keys without trusting filenames or metadata."""

    try:
        normalized = {
            key.removeprefix("model.diffusion_model."): value
            for key, value in state_dict.items()
        }
        expected_shapes = {
            "x_embedder.proj.1.weight": (2048, 68),
            "final_layer.linear.weight": (64, 2048),
        }
        for key, expected in expected_shapes.items():
            if _shape(normalized.get(key)) != expected:
                raise _TopologyMismatch(
                    "model.anima.state_shape_mismatch",
                    f"Anima state tensor {key!r} does not have the verified shape {expected}",
                )

        block_indices = {
            int(match.group(1))
            for key in normalized
            if (match := _BLOCK_KEY.match(key)) is not None
            and _shape(normalized[key]) == (8192, 2048)
        }
        if block_indices != set(range(28)):
            raise _TopologyMismatch(
                "model.anima.state_block_topology_mismatch",
                "Anima state tensors do not describe exactly 28 verified transformer blocks",
            )

        llm_indices = {
            int(match.group(1))
            for key in normalized
            if (match := _LLM_BLOCK_KEY.match(key)) is not None
            and _shape(normalized[key]) == (1024, 1024)
        }
        if llm_indices != set(range(6)):
            raise _TopologyMismatch(
                "model.anima.state_conditioning_topology_mismatch",
                "Anima state tensors do not describe exactly 6 verified LLM-adapter blocks",
            )
        return AnimaAdapterMatch(True, None, None, EXPECTED_FACTS)
    except _TopologyMismatch as mismatch:
        return AnimaAdapterMatch(False, mismatch.code, str(mismatch))
    except Exception:
        return AnimaAdapterMatch(
            False,
            "model.anima.state_inspection_failed",
            "The Anima state topology could not be inspected safely",
        )


class StrictAnimaAdapter:
    """Match only the live Forge Anima topology verified by this build."""

    adapter_id = "forge-anima-still-v1"
    adapter_version = "1.0.0"

    def diagnose(self, model_context: Any) -> AnimaAdapterMatch:
        try:
            if model_context is None:
                raise _TopologyMismatch("model.not_loaded", "No model is loaded")
            _require_type(
                model_context,
                ENGINE_TYPE,
                "model.anima.engine_type_mismatch",
                "diffusion engine",
            )
            flags = {
                "is_sd1": False,
                "is_sdxl": False,
                "is_wan": True,
                "use_shift": True,
                "use_distilled_cfg_scale": False,
                "is_inpaint": False,
            }
            if any(
                _require_attribute(model_context, name, "model.anima.flag_missing") is not expected
                for name, expected in flags.items()
            ):
                raise _TopologyMismatch(
                    "model.anima.family_flags_mismatch",
                    "The loaded model family flags are not strict still-image Anima",
                )

            model_config = _require_attribute(
                model_context,
                "model_config",
                "model.anima.config_missing",
            )
            _require_type(
                model_config,
                MODEL_CONFIG_TYPE,
                "model.anima.config_type_mismatch",
                "model configuration",
            )
            model_type = _require_attribute(
                model_config,
                "model_type",
                "model.anima.prediction_type_missing",
            )
            if _require_attribute(
                model_type,
                "name",
                "model.anima.prediction_type_missing",
            ) != "FLOW":
                raise _TopologyMismatch(
                    "model.anima.prediction_type_mismatch",
                    "Only Forge's verified discrete-flow Anima prediction path is enabled",
                )

            forge_objects = _require_attribute(
                model_context,
                "forge_objects",
                "model.anima.forge_objects_missing",
            )
            patcher = _require_attribute(
                forge_objects,
                "unet",
                "model.anima.patcher_missing",
            )
            _require_type(
                patcher,
                PATCHER_TYPE,
                "model.anima.patcher_type_mismatch",
                "denoiser patcher",
            )
            k_model = _require_attribute(
                patcher,
                "model",
                "model.anima.k_model_missing",
            )
            _require_type(
                k_model,
                K_MODEL_TYPE,
                "model.anima.k_model_type_mismatch",
                "denoiser wrapper",
            )
            diffusion = _require_attribute(
                k_model,
                "diffusion_model",
                "model.anima.diffusion_missing",
            )
            _require_type(
                diffusion,
                DIFFUSION_TYPE,
                "model.anima.diffusion_type_mismatch",
                "diffusion transformer",
            )

            if (
                _require_attribute(diffusion, "in_channels", "model.anima.channels_missing") != 16
                or _require_attribute(diffusion, "out_channels", "model.anima.channels_missing") != 16
                or _require_attribute(diffusion, "patch_spatial", "model.anima.patch_missing") != 2
                or _require_attribute(diffusion, "patch_temporal", "model.anima.patch_missing") != 1
            ):
                raise _TopologyMismatch(
                    "model.anima.latent_patch_mismatch",
                    "The Anima latent or patch layout differs from the verified still-image topology",
                )

            x_embedder = _require_attribute(
                diffusion,
                "x_embedder",
                "model.anima.x_embedder_missing",
            )
            projection = _require_attribute(
                x_embedder,
                "proj",
                "model.anima.x_embedder_missing",
            )
            if len(projection) != 2 or _linear_shape(
                projection[1],
                "model.anima.x_embedder_invalid",
            ) != (2048, 68):
                raise _TopologyMismatch(
                    "model.anima.x_embedder_mismatch",
                    "The Anima patch embedding differs from the verified 17-channel layout",
                )

            blocks = _require_attribute(
                diffusion,
                "blocks",
                "model.anima.blocks_missing",
            )
            if len(blocks) != 28:
                raise _TopologyMismatch(
                    "model.anima.block_count_mismatch",
                    "The Anima transformer does not contain exactly 28 verified blocks",
                )
            for block in blocks:
                _require_type(
                    block,
                    BLOCK_TYPE,
                    "model.anima.block_type_mismatch",
                    "transformer block",
                )
                cross_attention = _require_attribute(
                    block,
                    "cross_attn",
                    "model.anima.cross_attention_missing",
                )
                _require_type(
                    cross_attention,
                    CROSS_ATTENTION_TYPE,
                    "model.anima.cross_attention_type_mismatch",
                    "cross-attention block",
                )
                if (
                    _require_attribute(
                        cross_attention,
                        "context_dim",
                        "model.anima.cross_attention_invalid",
                    )
                    != 1024
                    or _require_attribute(
                        cross_attention,
                        "n_heads",
                        "model.anima.cross_attention_invalid",
                    )
                    != 16
                    or _require_attribute(
                        cross_attention,
                        "head_dim",
                        "model.anima.cross_attention_invalid",
                    )
                    != 128
                ):
                    raise _TopologyMismatch(
                        "model.anima.cross_attention_layout_mismatch",
                        "Anima cross-attention dimensions differ from the verified topology",
                    )

            final_layer = _require_attribute(
                diffusion,
                "final_layer",
                "model.anima.final_layer_missing",
            )
            final_linear = _require_attribute(
                final_layer,
                "linear",
                "model.anima.final_layer_missing",
            )
            if _linear_shape(
                final_linear,
                "model.anima.final_layer_invalid",
            ) != (64, 2048):
                raise _TopologyMismatch(
                    "model.anima.final_layer_mismatch",
                    "The Anima output projection differs from the verified topology",
                )

            vae = _require_attribute(forge_objects, "vae", "model.anima.vae_missing")
            _require_type(vae, VAE_TYPE, "model.anima.vae_type_mismatch", "VAE wrapper")
            vae_model = _require_attribute(
                vae,
                "first_stage_model",
                "model.anima.vae_model_missing",
            )
            _require_type(
                vae_model,
                VAE_MODEL_TYPE,
                "model.anima.vae_model_type_mismatch",
                "VAE model",
            )
            upscale_ratio = _require_attribute(
                vae,
                "upscale_ratio",
                "model.anima.vae_scale_missing",
            )
            if (
                _require_attribute(
                    vae,
                    "latent_channels",
                    "model.anima.vae_channels_missing",
                )
                != 16
                or not isinstance(upscale_ratio, (tuple, list))
                or len(upscale_ratio) != 3
                or not callable(upscale_ratio[0])
                or tuple(upscale_ratio[1:]) != (8, 8)
            ):
                raise _TopologyMismatch(
                    "model.anima.vae_layout_mismatch",
                    "The Anima Wan VAE latent or spatial scale differs from the verified topology",
                )

            clip = _require_attribute(forge_objects, "clip", "model.anima.clip_missing")
            _require_type(clip, CLIP_TYPE, "model.anima.clip_type_mismatch", "text encoder wrapper")
            clip_model = _require_attribute(
                clip,
                "cond_stage_model",
                "model.anima.clip_model_missing",
            )
            _require_type(
                clip_model,
                JOINT_TEXT_ENCODER_TYPE,
                "model.anima.clip_model_type_mismatch",
                "joint text encoder",
            )
            qwen = _require_attribute(
                clip_model,
                "qwen3_06b",
                "model.anima.qwen_missing",
            )
            _require_type(qwen, QWEN_TYPE, "model.anima.qwen_type_mismatch", "Qwen3 encoder")
            llm_adapter = _require_attribute(
                qwen,
                "llm_adapter",
                "model.anima.llm_adapter_missing",
            )
            _require_type(
                llm_adapter,
                LLM_ADAPTER_TYPE,
                "model.anima.llm_adapter_type_mismatch",
                "LLM adapter",
            )
            if len(
                _require_attribute(
                    llm_adapter,
                    "blocks",
                    "model.anima.llm_adapter_blocks_missing",
                )
            ) != 6:
                raise _TopologyMismatch(
                    "model.anima.llm_adapter_block_count_mismatch",
                    "The Anima conditioning adapter does not contain exactly 6 verified blocks",
                )

            text_engine = _require_attribute(
                model_context,
                "text_processing_engine_anima",
                "model.anima.text_engine_missing",
            )
            _require_type(
                text_engine,
                TEXT_ENGINE_TYPE,
                "model.anima.text_engine_type_mismatch",
                "text-processing engine",
            )
            if (
                _require_attribute(text_engine, "text_encoder", "model.anima.text_engine_invalid")
                is not qwen
                or _require_attribute(text_engine, "id_pad", "model.anima.text_engine_invalid")
                != 151643
                or _require_attribute(text_engine, "id_end", "model.anima.text_engine_invalid") != 1
            ):
                raise _TopologyMismatch(
                    "model.anima.text_engine_contract_mismatch",
                    "The Anima text-processing contract differs from the verified Qwen/T5 path",
                )

            return AnimaAdapterMatch(True, None, None, EXPECTED_FACTS)
        except _TopologyMismatch as mismatch:
            return AnimaAdapterMatch(False, mismatch.code, str(mismatch))
        except Exception:
            return AnimaAdapterMatch(
                False,
                "model.anima.inspection_failed",
                "The loaded Anima topology could not be inspected safely",
            )

    def matches(self, model_context: Any) -> bool:
        return self.diagnose(model_context).matched

    def architecture_facts(self, model_context: Any) -> Mapping[str, Any] | None:
        match = self.diagnose(model_context)
        return match.facts.as_dict() if match.facts is not None else None

    def supported_engine_ids(self) -> frozenset[str]:
        # Anima cross-attention is not Forge's U-Net attn2 replacement seam.
        return frozenset({"anima-attention"})

    def unsupported_fields(self) -> frozenset[str]:
        return frozenset(
            {
                "conditioning.prompt_schedules",
                "controlnet",
                "edit",
                "passes.hires",
                "passes.refiner",
            }
        )

    def expected_fallbacks(self) -> tuple[str, ...]:
        return ()


anima_adapter = StrictAnimaAdapter()
