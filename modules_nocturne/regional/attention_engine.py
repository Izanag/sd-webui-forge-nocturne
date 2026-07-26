"""SD 1.5 cross-attention decomposition engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from modules_nocturne.regional.adapters.sd15 import AttentionBlockSpec, StrictSD15Adapter, sd15_adapter
from modules_nocturne.regional.capabilities import EngineCapabilities
from modules_nocturne.regional.conditioning import RegionalConditioningBatch
from modules_nocturne.regional.errors import PlanError
from modules_nocturne.regional.model import OverlapPolicy, UncoveredPolicy
from modules_nocturne.regional.prompts import PromptOwner
from modules_nocturne.regional.runtime import RegionalBatchCompilation, RegionalRuntime


def _normalised_denoising_progress(extra_options: dict[str, Any]) -> float:
    """Map Forge's current sigma to the sampler's zero-to-one denoising span."""

    import torch

    sampling_sigmas = extra_options.get("sampling_sigmas")
    current_sigmas = extra_options.get("sigmas")
    if not torch.is_tensor(sampling_sigmas) or not torch.is_tensor(current_sigmas):
        raise RuntimeError(
            "Scheduled Regional guidance requires Forge sampler sigma metadata"
        )
    schedule = sampling_sigmas.detach().flatten()
    current = current_sigmas.detach().flatten()
    if schedule.numel() < 2 or current.numel() < 1:
        raise RuntimeError("Forge sampler sigma metadata is incomplete")
    closest = int(torch.argmin(torch.abs(schedule - current[0].to(schedule))).item())
    last_denoising_step = max(1, schedule.numel() - 2)
    return min(1.0, max(0.0, closest / last_denoising_step))


class AttentionDecompositionEngine:
    engine_id = "attention-decomposition"
    engine_version = "0.1.0"

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
            expected_fallbacks=(),
            cost_warning=(
                "Cross-attention is evaluated once per enabled region and can increase runtime and VRAM use. "
                "Regional seed policies do not alter attention-decomposition noise."
            ),
        )

    def runtime_installer(
        self,
        *,
        adapter: StrictSD15Adapter = sd15_adapter,
        attention_function: Callable | None = None,
    ) -> "SD15AttentionRuntimeInstaller":
        return SD15AttentionRuntimeInstaller(
            adapter=adapter,
            attention_function=attention_function,
        )


@dataclass(slots=True)
class SD15AttentionRuntimeInstaller:
    adapter: StrictSD15Adapter = sd15_adapter
    attention_function: Callable | None = None
    memory_budget_mb: int | None = None
    requires_conditioning: bool = True

    def install(
        self,
        *,
        runtime: RegionalRuntime,
        batch: RegionalBatchCompilation,
        model_context: Any,
    ) -> "InstalledAttentionDecomposition":
        plan = runtime.plan
        scheduled_guidance = any(
            region.enabled
            and (region.guidance.start != 0.0 or region.guidance.end != 1.0)
            for region in plan.regions
        )
        if scheduled_guidance:
            from modules import sd_samplers_kdiffusion

            sampler_name = str(plan.engine.options.get("sampler", "Euler a"))
            k_diffusion_samplers = {
                sampler.name
                for sampler in sd_samplers_kdiffusion.samplers_data_k_diffusion
            }
            if sampler_name not in k_diffusion_samplers:
                raise PlanError(
                    "engine.guidance_sampler.unsupported",
                    "$.engine.options.sampler",
                    "Scheduled Regional guidance currently requires a K-diffusion sampler",
                )
        grids = self.adapter.attention_grids(
            model_context,
            width=batch.context.width,
            height=batch.context.height,
        )
        compiled = runtime.compile_masks(
            tuple((grid.width, grid.height) for grid in grids),
            dtype="float32",
        )
        masks_by_size = {(mask.width, mask.height): mask for mask in compiled}
        masks_by_block = {
            grid.block.identity: masks_by_size[(grid.width, grid.height)]
            for grid in grids
        }
        modules = self.adapter.cross_attention_modules(model_context)
        memory_budget_mb = self.memory_budget_mb
        if memory_budget_mb is None:
            from modules import shared

            memory_budget_mb = int(
                getattr(shared.opts, "nocturne_regional_attention_memory_mb", 64)
            )
        if not 16 <= memory_budget_mb <= 2048:
            raise PlanError(
                "engine.attention_memory_budget.invalid",
                "$.engine",
                "Regional attention working-memory budget must be between 16 and 2048 MiB",
            )

        previous_unet = model_context.forge_objects.unet
        cloned_unet = previous_unet.clone()
        installation = InstalledAttentionDecomposition(
            model_context=model_context,
            previous_unet=previous_unet,
            cloned_unet=cloned_unet,
            blocks=self.adapter.attention_blocks(model_context),
            attention_modules=modules,
            masks_by_block=masks_by_block,
            enabled_region_ids=tuple(region.id for region in plan.regions if region.enabled),
            guidance_by_region={
                region.id: (region.guidance.start, region.guidance.end)
                for region in plan.regions
                if region.enabled
            },
            image_indices=batch.context.image_indices,
            attention_function=self.attention_function,
            attention_memory_budget_bytes=memory_budget_mb * 1024 * 1024,
        )
        try:
            installation.install()
            model_context.forge_objects.unet = cloned_unet
        except BaseException:
            installation.close()
            raise
        return installation


class InstalledAttentionDecomposition:
    """Own patches on one cloned U-Net and restore the previous patcher once."""

    def __init__(
        self,
        *,
        model_context: Any,
        previous_unet: Any,
        cloned_unet: Any,
        blocks: tuple[AttentionBlockSpec, ...],
        attention_modules,
        masks_by_block,
        enabled_region_ids,
        guidance_by_region,
        image_indices,
        attention_function: Callable | None,
        attention_memory_budget_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self.model_context = model_context
        self.previous_unet = previous_unet
        self.cloned_unet = cloned_unet
        self.blocks = blocks
        self.attention_modules = attention_modules
        self.masks_by_block = masks_by_block
        self.enabled_region_ids = tuple(enabled_region_ids)
        self.guidance_by_region = dict(guidance_by_region)
        self.image_indices = tuple(image_indices)
        self.attention_function = attention_function
        self.attention_memory_budget_bytes = int(attention_memory_budget_bytes)
        self.conditioning: RegionalConditioningBatch | None = None
        self._mask_tensors: dict[tuple[Any, ...], tuple[Any, dict[Any, Any]]] = {}
        self._closed = False

    def install(self) -> None:
        for block in self.blocks:
            module = self.attention_modules[block.identity]

            def patch(q, k, v, extra_options, *, block=block, module=module):
                return self._apply(block, module, q, k, v, extra_options)

            self.cloned_unet.set_model_attn2_replace(
                patch,
                block.block_kind,
                block.block_index,
                block.transformer_index,
            )

    def bind_conditioning(self, conditioning: RegionalConditioningBatch) -> None:
        if self._closed:
            raise RuntimeError("A closed attention engine cannot accept conditioning")
        if tuple(image.image_index for image in conditioning.images) != self.image_indices:
            raise RuntimeError("Regional conditioning image order does not match the installed engine")
        for image in conditioning.images:
            for prompt in image.prompts.values():
                if len(prompt.entries) != 1:
                    raise PlanError(
                        "engine.prompt_schedule.unsupported",
                        "$",
                        "Scheduled or alternating prompts are not yet enabled in the sampler engine",
                    )
                if len(prompt.entries[0].encoded) != 1:
                    raise PlanError(
                        "engine.prompt_composition.unsupported",
                        "$",
                        "Weighted AND prompts are not yet enabled in the sampler engine",
                    )
        self.conditioning = conditioning

    def _attention(self):
        if self.attention_function is not None:
            return self.attention_function
        from backend.nn.unet import attention_function

        return attention_function

    def _materialized_masks(self, block: AttentionBlockSpec, q):
        import torch

        compiled = self.masks_by_block[block.identity]
        cache_key = (compiled.key, q.device.type, q.device.index, q.dtype)
        cached = self._mask_tensors.get(cache_key)
        if cached is not None:
            return cached

        def tensor(value):
            return torch.tensor(value, device=q.device, dtype=q.dtype).reshape(1, -1, 1)

        materialized = (
            tensor(compiled.global_mask),
            {region_id: tensor(mask) for region_id, mask in compiled.region_masks.items()},
        )
        self._mask_tensors[cache_key] = materialized
        return materialized

    def _apply(self, block: AttentionBlockSpec, module, q, k, v, extra_options):
        import torch

        if self._closed or self.conditioning is None:
            raise RuntimeError("Regional attention ran without active conditioning")
        compiled_masks = self.masks_by_block[block.identity]
        if q.shape[1] != compiled_masks.width * compiled_masks.height:
            raise RuntimeError(
                f"Regional attention grid mismatch for {block.identity}: "
                f"{q.shape[1]} tokens != {compiled_masks.width}x{compiled_masks.height}"
            )
        if len(self.conditioning.images) < 1 or q.shape[0] % len(self.conditioning.images):
            raise RuntimeError("Regional attention batch does not align with conditioning images")

        cond_mark = extra_options.get("cond_mark")
        if cond_mark is None or len(cond_mark) != q.shape[0]:
            raise RuntimeError("Forge did not provide explicit conditional/unconditional branch markers")

        attention = self._attention()
        global_output = attention(q, k, v, block.heads, None)
        global_mask, region_masks = self._materialized_masks(block, q)
        output = torch.empty_like(global_output)
        image_count = len(self.conditioning.images)
        scheduled = any(
            start != 0.0 or end != 1.0
            for start, end in self.guidance_by_region.values()
        )
        progress = _normalised_denoising_progress(extra_options) if scheduled else 0.0

        for row in range(q.shape[0]):
            image = self.conditioning.images[row % image_count]
            polarity = "negative" if float(cond_mark[row]) >= 0.5 else "positive"
            composed = global_output[row : row + 1] * global_mask
            active_regions = []
            for region_id in self.enabled_region_ids:
                start, end = self.guidance_by_region[region_id]
                if progress < start or progress > end:
                    composed = (
                        composed
                        + global_output[row : row + 1] * region_masks[region_id]
                    )
                    continue
                prompt = image.get(PromptOwner(kind="region", region_id=region_id), polarity)
                context = prompt.entries[0].encoded[0].value
                if isinstance(context, dict):
                    context = context.get("crossattn")
                if not torch.is_tensor(context) or context.ndim != 2:
                    raise RuntimeError("SD 1.5 Regional conditioning must be a two-dimensional tensor")
                context = context.to(device=q.device, dtype=q.dtype).unsqueeze(0)
                active_regions.append((region_id, context))

            if not active_regions:
                output[row : row + 1] = composed
                continue

            context_elements = active_regions[0][1].numel()
            score_elements = (
                q.shape[1]
                * active_regions[0][1].shape[1]
                * max(1, block.heads)
            )
            estimated_elements_per_region = (
                3 * q[row : row + 1].numel()
                + 4 * context_elements
                + score_elements
            )
            estimated_bytes_per_region = max(
                q.element_size(),
                estimated_elements_per_region * q.element_size(),
            )
            chunk_size = max(
                1,
                min(
                    len(active_regions),
                    self.attention_memory_budget_bytes
                    // estimated_bytes_per_region,
                ),
            )
            for offset in range(0, len(active_regions), chunk_size):
                chunk = active_regions[offset : offset + chunk_size]
                contexts = torch.cat([context for _, context in chunk], dim=0)
                query = q[row : row + 1].expand(len(chunk), -1, -1).contiguous()
                regional_k = module.to_k(contexts)
                regional_v = module.to_v(contexts)
                regional_output = attention(
                    query,
                    regional_k,
                    regional_v,
                    block.heads,
                    None,
                )
                masks = torch.cat(
                    [region_masks[region_id] for region_id, _ in chunk],
                    dim=0,
                )
                composed = composed + (regional_output * masks).sum(
                    dim=0,
                    keepdim=True,
                )
            output[row : row + 1] = composed
        return output

    def close(self) -> None:
        if self._closed:
            return
        try:
            if getattr(self.model_context.forge_objects, "unet", None) is self.cloned_unet:
                self.model_context.forge_objects.unet = self.previous_unet
        finally:
            self.conditioning = None
            self._mask_tensors.clear()
            self.attention_modules = {}
            self.masks_by_block = {}
            self.guidance_by_region = {}
            self.model_context = None
            self.previous_unet = None
            self.cloned_unet = None
            self._closed = True


attention_decomposition_engine = AttentionDecompositionEngine()
