"""Clone-scoped Regional cross-attention for Forge's verified Anima model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from modules_nocturne.regional.adapters.anima import StrictAnimaAdapter, anima_adapter
from modules_nocturne.regional.attention_engine import (
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


class AnimaAttentionEngine:
    engine_id = "anima-attention"
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
                "Anima cross-attention is evaluated once per enabled region and "
                "can increase runtime and VRAM use. Regional seed policies do not "
                "alter Anima noise."
            ),
        )

    def runtime_installer(
        self,
        *,
        adapter: StrictAnimaAdapter = anima_adapter,
    ) -> "AnimaAttentionRuntimeInstaller":
        return AnimaAttentionRuntimeInstaller(adapter=adapter)


@dataclass(slots=True)
class AnimaAttentionRuntimeInstaller:
    adapter: StrictAnimaAdapter = anima_adapter
    memory_budget_mb: int | None = None
    requires_conditioning: bool = True

    def install(
        self,
        *,
        runtime: RegionalRuntime,
        batch: RegionalBatchCompilation,
        model_context: Any,
    ) -> "InstalledAnimaAttention":
        plan = runtime.plan
        scheduled_guidance = any(
            region.enabled
            and (region.guidance.start != 0.0 or region.guidance.end != 1.0)
            for region in plan.regions
        )
        scheduled_prompts = bool(
            batch.conditioning
            and any(
                len(prompt.entries) > 1
                for image in batch.conditioning.images
                for prompt in image.prompts.values()
            )
        )
        if scheduled_guidance or scheduled_prompts:
            sampler_name = str(plan.engine.options.get("sampler", "ER SDE"))
            _require_reliable_schedule_sampler(sampler_name)

        grid = self.adapter.patch_grid(
            width=batch.context.width,
            height=batch.context.height,
        )
        compiled = runtime.compile_masks(
            ((grid.patch_width, grid.patch_height),),
            dtype="float32",
        )[0]

        memory_budget_mb = self.memory_budget_mb
        if memory_budget_mb is None:
            from modules import shared

            memory_budget_mb = int(
                getattr(
                    shared.opts,
                    "nocturne_regional_attention_memory_mb",
                    64,
                )
            )
        if not 16 <= memory_budget_mb <= 2048:
            raise PlanError(
                "engine.attention_memory_budget.invalid",
                "$.engine",
                "Regional attention working-memory budget must be between 16 and 2048 MiB",
            )

        from backend.nn.anima import attention_function

        if not callable(attention_function):
            raise PlanError(
                "engine.attention_backend.unsupported",
                "$.engine",
                "The selected Anima attention backend is not callable",
            )
        attention_backend_id = (
            f"{getattr(attention_function, '__module__', type(attention_function).__module__)}."
            f"{getattr(attention_function, '__qualname__', type(attention_function).__qualname__)}"
        )

        previous_unet = model_context.forge_objects.unet
        cloned_unet = previous_unet.clone()
        installation = InstalledAnimaAttention(
            model_context=model_context,
            previous_unet=previous_unet,
            cloned_unet=cloned_unet,
            adapter=self.adapter,
            compiled_masks=compiled,
            grid=grid,
            enabled_region_ids=tuple(
                region.id for region in plan.regions if region.enabled
            ),
            guidance_by_region={
                region.id: (region.guidance.start, region.guidance.end)
                for region in plan.regions
                if region.enabled
            },
            image_indices=batch.context.image_indices,
            attention_backend_id=attention_backend_id,
            attention_memory_budget_bytes=memory_budget_mb * 1024 * 1024,
        )
        try:
            installation.install()
            model_context.forge_objects.unet = cloned_unet
        except BaseException:
            installation.close()
            raise
        return installation


class InstalledAnimaAttention:
    """Own one Anima cross-attention hook and restore its parent patcher once."""

    def __init__(
        self,
        *,
        model_context: Any,
        previous_unet: Any,
        cloned_unet: Any,
        adapter: StrictAnimaAdapter,
        compiled_masks,
        grid,
        enabled_region_ids,
        guidance_by_region,
        image_indices,
        attention_backend_id: str,
        attention_memory_budget_bytes: int,
    ) -> None:
        self.model_context = model_context
        self.previous_unet = previous_unet
        self.cloned_unet = cloned_unet
        self.adapter = adapter
        self.compiled_masks = compiled_masks
        self.grid = grid
        self.enabled_region_ids = tuple(enabled_region_ids)
        self.guidance_by_region = dict(guidance_by_region)
        self.image_indices = tuple(image_indices)
        self.attention_backend_id = attention_backend_id
        self.attention_memory_budget_bytes = int(attention_memory_budget_bytes)
        self.conditioning: RegionalConditioningBatch | None = None
        self._has_prompt_schedules = False
        self._mask_tensors: dict[tuple[Any, ...], tuple[Any, dict[Any, Any]]] = {}
        self._closed = False

    def install(self) -> None:
        if self._closed:
            raise RuntimeError("A closed Anima attention engine cannot be installed")
        self.cloned_unet.set_transformer_option(
            "anima_cross_attention",
            self._apply,
        )

    def bind_conditioning(self, conditioning: RegionalConditioningBatch) -> None:
        if self._closed:
            raise RuntimeError("A closed Anima attention engine cannot accept conditioning")
        if tuple(
            image.image_index for image in conditioning.images
        ) != self.image_indices:
            raise RuntimeError(
                "Regional conditioning image order does not match the Anima engine"
            )
        for image in conditioning.images:
            for prompt in image.prompts.values():
                for entry in prompt.entries:
                    if len(entry.encoded) != 1:
                        raise PlanError(
                            "engine.prompt_composition.unsupported",
                            "$",
                            "Weighted AND prompts are not yet enabled in the Anima engine",
                        )
                    self.adapter.validate_conditioning_value(
                        entry.encoded[0].value
                    )
        self.conditioning = conditioning
        self._has_prompt_schedules = any(
            len(prompt.entries) > 1
            for image in conditioning.images
            for prompt in image.prompts.values()
        )

    def _materialized_masks(self, hidden_states):
        import torch

        cache_key = (
            self.compiled_masks.key,
            hidden_states.device.type,
            hidden_states.device.index,
            hidden_states.dtype,
        )
        cached = self._mask_tensors.get(cache_key)
        if cached is not None:
            return cached

        def tensor(value):
            return torch.tensor(
                value,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            ).reshape(1, -1, 1)

        materialized = (
            tensor(self.compiled_masks.global_mask),
            {
                region_id: tensor(mask)
                for region_id, mask in self.compiled_masks.region_masks.items()
            },
        )
        self._mask_tensors[cache_key] = materialized
        return materialized

    def _apply(
        self,
        *,
        module,
        hidden_states,
        context,
        rope_emb,
        transformer_options,
    ):
        import torch

        if self._closed or self.conditioning is None:
            raise RuntimeError(
                "Anima Regional cross-attention ran without active conditioning"
            )
        if hidden_states.ndim != 3:
            raise RuntimeError("Anima Regional hidden states must be three-dimensional")
        if hidden_states.shape[1] != self.grid.query_count:
            raise RuntimeError(
                "Anima Regional patch-grid mismatch: "
                f"{hidden_states.shape[1]} tokens != {self.grid.query_count}"
            )
        image_count = len(self.conditioning.images)
        if image_count < 1 or hidden_states.shape[0] % image_count:
            raise RuntimeError(
                "Anima Regional batch does not align with conditioning images"
            )
        cond_mark = transformer_options.get("cond_mark")
        if cond_mark is None or len(cond_mark) != hidden_states.shape[0]:
            raise RuntimeError(
                "Forge did not provide explicit Anima conditional/unconditional markers"
            )
        block_index = transformer_options.get("anima_block_index")
        if (
            isinstance(block_index, bool)
            or not isinstance(block_index, int)
            or not 0 <= block_index < 28
        ):
            raise RuntimeError("Forge did not provide a valid Anima block index")

        module_options = dict(transformer_options)
        module_options.pop("anima_cross_attention", None)
        global_output = module(
            hidden_states,
            context,
            rope_emb=rope_emb,
            transformer_options=module_options,
        )
        global_mask, region_masks = self._materialized_masks(hidden_states)
        output = torch.empty_like(global_output)

        scheduled = any(
            start != 0.0 or end != 1.0
            for start, end in self.guidance_by_region.values()
        )
        if scheduled or self._has_prompt_schedules:
            step_index, progress = _denoising_position(transformer_options)
        else:
            step_index, progress = 0, 0.0

        for row in range(hidden_states.shape[0]):
            image = self.conditioning.images[row % image_count]
            polarity = (
                "negative" if float(cond_mark[row]) >= 0.5 else "positive"
            )
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
                prompt = image.get(
                    PromptOwner(kind="region", region_id=region_id),
                    polarity,
                )
                entry = _schedule_entry(prompt, step_index)
                local_context = self.adapter.cross_attention_context(
                    entry.encoded[0].value
                ).to(
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                active_regions.append((region_id, local_context))

            if not active_regions:
                output[row : row + 1] = composed
                continue

            context_elements = active_regions[0][1].numel()
            score_elements = (
                self.grid.query_count
                * active_regions[0][1].shape[1]
                * 16
            )
            estimated_elements_per_region = (
                3 * hidden_states[row : row + 1].numel()
                + 4 * context_elements
                + score_elements
            )
            estimated_bytes_per_region = max(
                hidden_states.element_size(),
                estimated_elements_per_region * hidden_states.element_size(),
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
                local_contexts = torch.cat(
                    [local_context for _, local_context in chunk],
                    dim=0,
                )
                local_hidden = hidden_states[row : row + 1].expand(
                    len(chunk),
                    -1,
                    -1,
                ).contiguous()
                local_output = module(
                    local_hidden,
                    local_contexts,
                    rope_emb=rope_emb,
                    transformer_options=module_options,
                )
                masks = torch.cat(
                    [region_masks[region_id] for region_id, _ in chunk],
                    dim=0,
                )
                composed = composed + (local_output * masks).sum(
                    dim=0,
                    keepdim=True,
                )
            output[row : row + 1] = composed
        return output

    def close(self) -> None:
        if self._closed:
            return
        try:
            current_unet = getattr(
                self.model_context.forge_objects,
                "unet",
                None,
            )
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
            self.compiled_masks = None
            self.grid = None
            self.guidance_by_region = {}
            self.adapter = None
            self.model_context = None
            self.previous_unet = None
            self.cloned_unet = None
            self._closed = True


anima_attention_engine = AnimaAttentionEngine()
