"""Masked Forge condition-batch fusion for verified U-Net adapters."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Any

from modules_nocturne.regional.adapters.sd15 import sd15_adapter
from modules_nocturne.regional.attention_engine import (
    RegionalAttentionAdapter,
    _denoising_position,
    _require_reliable_schedule_sampler,
    _schedule_entry,
)
from modules_nocturne.regional.capabilities import EngineCapabilities
from modules_nocturne.regional.conditioning import RegionalConditioningBatch
from modules_nocturne.regional.errors import PlanError
from modules_nocturne.regional.model import OverlapPolicy, UncoveredPolicy
from modules_nocturne.regional.prompts import PromptOwner
from modules_nocturne.regional.runtime import RegionalBatchCompilation, RegionalRuntime


MAX_FUSION_REGIONS = 8


def _lcm(left: int, right: int) -> int:
    return abs(left * right) // gcd(left, right)


class ConditionBatchFusionEngine:
    engine_id = "denoising-fusion"
    engine_version = "0.1.0"
    fusion_policy_version = "forge-condition-batch/v1"
    path_limit = MAX_FUSION_REGIONS

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            engine_id=self.engine_id,
            engine_version=self.engine_version,
            overlap_policies=frozenset(OverlapPolicy),
            uncovered_policies=frozenset(
                {
                    UncoveredPolicy.GLOBAL,
                    UncoveredPolicy.NEAREST,
                    UncoveredPolicy.ERROR,
                }
            ),
            supported_fields=frozenset(
                {
                    "global.positive",
                    "global.negative",
                    "regions.positive",
                    "regions.negative",
                    "regions.inherit_global_positive",
                    "regions.inherit_global_negative",
                    "regions.weight",
                    "regions.priority",
                    "regions.feather_px",
                    "regions.grow_shrink_px",
                    "regions.guidance.start",
                    "regions.guidance.end",
                    "composition.overlap_policy",
                    "composition.uncovered_policy",
                }
            ),
            cost_warning=(
                "Denoising fusion evaluates one complete Forge condition path per active "
                "region and can be substantially slower than attention decomposition."
            ),
        )

    def runtime_installer(
        self,
        *,
        adapter: RegionalAttentionAdapter = sd15_adapter,
    ) -> "ConditionBatchFusionRuntimeInstaller":
        return ConditionBatchFusionRuntimeInstaller(adapter=adapter)


@dataclass(slots=True)
class ConditionBatchFusionRuntimeInstaller:
    adapter: RegionalAttentionAdapter = sd15_adapter
    requires_conditioning: bool = True

    def install(
        self,
        *,
        runtime: RegionalRuntime,
        batch: RegionalBatchCompilation,
        model_context: Any,
    ) -> "InstalledConditionBatchFusion":
        if self.adapter.adapter_id != "forge-sd15-v1":
            raise PlanError(
                "engine.fusion.adapter.unsupported",
                "$.engine",
                "Denoising fusion is currently available only for the verified SD 1.5 adapter",
            )
        enabled_regions = tuple(
            region for region in runtime.plan.regions if region.enabled
        )
        if len(enabled_regions) > MAX_FUSION_REGIONS:
            raise PlanError(
                "engine.fusion.region_limit",
                "$.regions",
                f"Denoising fusion supports at most {MAX_FUSION_REGIONS} active regions",
            )
        scheduled = any(
            region.guidance.start != 0.0 or region.guidance.end != 1.0
            for region in enabled_regions
        )
        if scheduled:
            _require_reliable_schedule_sampler(
                str(runtime.plan.engine.options.get("sampler", "Euler a"))
            )

        latent_width = max(1, batch.context.width // 8)
        latent_height = max(1, batch.context.height // 8)
        compiled_masks = runtime.compile_masks(((latent_width, latent_height),))[0]
        previous_unet = model_context.forge_objects.unet
        cloned_unet = previous_unet.clone()
        installation = InstalledConditionBatchFusion(
            model_context=model_context,
            previous_unet=previous_unet,
            cloned_unet=cloned_unet,
            adapter=self.adapter,
            compiled_masks=compiled_masks,
            enabled_region_ids=tuple(region.id for region in enabled_regions),
            guidance_by_region={
                region.id: (region.guidance.start, region.guidance.end)
                for region in enabled_regions
            },
            image_indices=batch.context.image_indices,
        )
        try:
            installation.install()
            model_context.forge_objects.unet = cloned_unet
        except BaseException:
            installation.close()
            raise
        return installation


class InstalledConditionBatchFusion:
    """Own one cloned condition-batch callback and its compiled masks."""

    engine_id = ConditionBatchFusionEngine.engine_id

    def __init__(
        self,
        *,
        model_context: Any,
        previous_unet: Any,
        cloned_unet: Any,
        adapter: RegionalAttentionAdapter,
        compiled_masks: Any,
        enabled_region_ids: tuple[Any, ...],
        guidance_by_region: dict[Any, tuple[float, float]],
        image_indices: tuple[int, ...],
    ) -> None:
        self.model_context = model_context
        self.previous_unet = previous_unet
        self.cloned_unet = cloned_unet
        self.adapter = adapter
        self.compiled_masks = compiled_masks
        self.enabled_region_ids = tuple(enabled_region_ids)
        self.guidance_by_region = dict(guidance_by_region)
        self.image_indices = tuple(image_indices)
        self.conditioning: RegionalConditioningBatch | None = None
        self._has_prompt_schedules = False
        self._mask_tensors: dict[tuple[Any, ...], tuple[Any, dict[Any, Any]]] = {}
        self._closed = False

    @property
    def path_count(self) -> int:
        return 1 + len(self.enabled_region_ids)

    def install(self) -> None:
        self.cloned_unet.set_model_sampler_calc_cond_batch_function(
            self.calculate_cond_uncond_batch
        )

    def bind_conditioning(self, conditioning: RegionalConditioningBatch) -> None:
        if self._closed:
            raise RuntimeError("A closed fusion engine cannot accept conditioning")
        if tuple(image.image_index for image in conditioning.images) != self.image_indices:
            raise RuntimeError("Regional conditioning image order does not match the fusion engine")
        for image in conditioning.images:
            for prompt in image.prompts.values():
                for entry in prompt.entries:
                    if len(entry.encoded) != 1:
                        raise PlanError(
                            "engine.prompt_composition.unsupported",
                            "$",
                            "Weighted AND prompts are not yet enabled in denoising fusion",
                        )
                    self.adapter.validate_conditioning_value(entry.encoded[0].value)
        self.conditioning = conditioning
        self._has_prompt_schedules = any(
            len(prompt.entries) > 1
            for image in conditioning.images
            for prompt in image.prompts.values()
        )

    def _materialized_masks(self, latent):
        import torch

        height, width = latent.shape[-2:]
        if (width, height) != (self.compiled_masks.width, self.compiled_masks.height):
            raise RuntimeError(
                "Regional fusion mask grid does not match the active latent: "
                f"{width}x{height} != {self.compiled_masks.width}x{self.compiled_masks.height}"
            )
        key = (
            self.compiled_masks.key,
            latent.device.type,
            latent.device.index,
            latent.dtype,
        )
        cached = self._mask_tensors.get(key)
        if cached is not None:
            return cached

        def tensor(mask):
            return torch.tensor(mask, device=latent.device, dtype=latent.dtype).reshape(
                1, height, width
            )

        materialized = (
            tensor(self.compiled_masks.global_mask),
            {
                region_id: tensor(mask)
                for region_id, mask in self.compiled_masks.region_masks.items()
            },
        )
        self._mask_tensors[key] = materialized
        return materialized

    @staticmethod
    def _stack_tensors(tensors: tuple[Any, ...]):
        import torch

        values = tuple(
            tensor if torch.is_tensor(tensor) else torch.as_tensor(tensor)
            for tensor in tensors
        )
        dimensions = values[0].ndim
        if any(value.ndim != dimensions for value in values):
            raise RuntimeError("Regional conditioning ranks differ between batch rows")
        if dimensions == 2:
            maximum = values[0].shape[0]
            for value in values[1:]:
                if value.shape[-1] != values[0].shape[-1]:
                    raise RuntimeError("Regional cross-attention widths differ")
                maximum = _lcm(maximum, value.shape[0])
            if any(maximum // value.shape[0] > 4 for value in values):
                raise RuntimeError("Regional prompt lengths cannot be combined safely")
            values = tuple(
                value.repeat(maximum // value.shape[0], 1)
                if value.shape[0] != maximum
                else value
                for value in values
            )
        elif dimensions == 1:
            if any(value.shape != values[0].shape for value in values):
                raise RuntimeError("Regional pooled-conditioning widths differ")
        else:
            raise RuntimeError("Regional conditioning has an unexpected rank")
        return torch.stack(values, dim=0)

    @classmethod
    def _batched_value(cls, values: tuple[Any, ...]):
        if isinstance(values[0], dict):
            keys = set(values[0])
            if any(set(value) != keys for value in values):
                raise RuntimeError("Regional conditioning fields differ between batch rows")
            return {
                key: cls._stack_tensors(tuple(value[key] for value in values))
                for key in sorted(keys)
            }
        return cls._stack_tensors(values)

    def _compiled_region_condition(
        self,
        *,
        region_id: Any,
        polarity: str,
        step_index: int,
        source_conditions: list[dict[str, Any]],
        mask,
    ) -> list[dict[str, Any]]:
        from backend.sampling.condition import compile_conditions

        if self.conditioning is None:
            raise RuntimeError("Regional fusion ran without active conditioning")
        values = []
        for image in self.conditioning.images:
            prompt = image.get(PromptOwner(kind="region", region_id=region_id), polarity)
            entry = _schedule_entry(prompt, step_index)
            values.append(entry.encoded[0].value)
        compiled = compile_conditions(self._batched_value(tuple(values)))
        if len(compiled) != 1:
            raise RuntimeError("Regional conditioning produced multiple unexpected branches")
        regional = dict(compiled[0])
        regional["mask"] = mask
        regional["mask_strength"] = 1.0
        if source_conditions:
            source = source_conditions[0]
            for name in ("control",):
                if name in source:
                    regional[name] = source[name]
        return [regional]

    @staticmethod
    def _masked_global_conditions(
        conditions: list[dict[str, Any]] | None,
        mask,
    ) -> list[dict[str, Any]] | None:
        if conditions is None:
            return None
        result = []
        for condition in conditions:
            item = dict(condition)
            item["mask"] = mask
            item["mask_strength"] = 1.0
            result.append(item)
        return result

    def calculate_cond_uncond_batch(
        self,
        model,
        cond,
        uncond,
        latent,
        timestep,
        model_options,
    ):
        from backend.sampling.sampling_function import calc_cond_uncond_batch

        if self._closed or self.conditioning is None:
            raise RuntimeError("Regional fusion ran without active conditioning")
        if latent.shape[0] != len(self.conditioning.images):
            raise RuntimeError("Regional conditioning batch does not align with the latent batch")

        global_mask, region_masks = self._materialized_masks(latent)
        transformer_options = dict(model_options.get("transformer_options", {}))
        scheduled = self._has_prompt_schedules or any(
            start != 0.0 or end != 1.0
            for start, end in self.guidance_by_region.values()
        )
        if scheduled:
            transformer_options["sigmas"] = timestep
            step_index, progress = _denoising_position(transformer_options)
        else:
            step_index, progress = 0, 0.0

        active_region_ids = []
        inactive_mask = global_mask.clone()
        for region_id in self.enabled_region_ids:
            start, end = self.guidance_by_region[region_id]
            if progress < start or progress > end:
                inactive_mask = inactive_mask + region_masks[region_id]
            else:
                active_region_ids.append(region_id)

        regional_cond = []
        regional_uncond = [] if uncond is not None else None
        for region_id in active_region_ids:
            mask = region_masks[region_id]
            regional_cond.extend(
                self._compiled_region_condition(
                    region_id=region_id,
                    polarity="positive",
                    step_index=step_index,
                    source_conditions=cond,
                    mask=mask,
                )
            )
            if regional_uncond is not None:
                regional_uncond.extend(
                    self._compiled_region_condition(
                        region_id=region_id,
                        polarity="negative",
                        step_index=step_index,
                        source_conditions=uncond,
                        mask=mask,
                    )
                )

        fused_cond = [
            *self._masked_global_conditions(cond, inactive_mask),
            *regional_cond,
        ]
        masked_uncond = self._masked_global_conditions(uncond, inactive_mask)
        fused_uncond = (
            [*masked_uncond, *regional_uncond]
            if masked_uncond is not None and regional_uncond is not None
            else None
        )
        return calc_cond_uncond_batch(
            model,
            fused_cond,
            fused_uncond,
            latent,
            timestep,
            model_options,
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            current_unet = getattr(self.model_context.forge_objects, "unet", None)
            patcher = current_unet
            descendants_seen: set[int] = set()
            while patcher is not None and id(patcher) not in descendants_seen:
                if patcher is self.cloned_unet:
                    self.model_context.forge_objects.unet = self.previous_unet
                    break
                descendants_seen.add(id(patcher))
                patcher = getattr(patcher, "parent", None)
        finally:
            self.conditioning = None
            self._mask_tensors.clear()
            self.guidance_by_region = {}
            self.adapter = None
            self.compiled_masks = None
            self.model_context = None
            self.previous_unet = None
            self.cloned_unet = None
            self._closed = True


condition_batch_fusion_engine = ConditionBatchFusionEngine()
