"""Job-scoped Regional compilation and resource ownership."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Protocol

from modules_nocturne.regional.generation import AuthorizedRegionalPlan
from modules_nocturne.regional.masks import CompiledMaskSet, MaskCompilerCache, compile_mask_pyramid
from modules_nocturne.regional.prompts import CompiledPromptPlan, PromptExpansionService, PromptTokenCounter, compile_prompt_plan
from modules_nocturne.regional.seeds import ResolvedSeedPlan, resolve_forge_seed_sequence
from modules_nocturne.regional.validation import ValidationCapabilities


@dataclass(frozen=True, slots=True)
class RegionalBatchContext:
    """Explicit Forge batch inputs captured at the lifecycle boundary."""

    batch_number: int
    image_start: int
    width: int
    height: int
    prompts: tuple[str, ...]
    negative_prompts: tuple[str, ...]
    seeds: tuple[int, ...]
    subseeds: tuple[int, ...]
    pass_name: str = "base"

    def __post_init__(self) -> None:
        values = (self.prompts, self.negative_prompts, self.seeds, self.subseeds)
        if isinstance(self.batch_number, bool) or not isinstance(self.batch_number, int) or self.batch_number < 0:
            raise ValueError("Batch number must be a non-negative integer")
        if isinstance(self.image_start, bool) or not isinstance(self.image_start, int) or self.image_start < 0:
            raise ValueError("Image start must be a non-negative integer")
        if (
            isinstance(self.width, bool)
            or isinstance(self.height, bool)
            or not isinstance(self.width, int)
            or not isinstance(self.height, int)
            or self.width < 1
            or self.height < 1
        ):
            raise ValueError("Batch dimensions must be positive integers")
        if not self.pass_name:
            raise ValueError("Pass name cannot be empty")
        if not self.seeds or any(len(value) != len(self.seeds) for value in values):
            raise ValueError("Batch prompt and seed sequences must be non-empty and have equal lengths")
        object.__setattr__(self, "prompts", tuple(self.prompts))
        object.__setattr__(self, "negative_prompts", tuple(self.negative_prompts))
        object.__setattr__(self, "seeds", tuple(int(seed) for seed in self.seeds))
        object.__setattr__(self, "subseeds", tuple(int(seed) for seed in self.subseeds))

    @property
    def image_indices(self) -> tuple[int, ...]:
        return tuple(range(self.image_start, self.image_start + len(self.seeds)))


@dataclass(frozen=True, slots=True)
class RegionalBatchCompilation:
    context: RegionalBatchContext
    prompts: tuple[CompiledPromptPlan, ...]
    seeds: tuple[ResolvedSeedPlan, ...]
    masks: tuple[CompiledMaskSet, ...] = ()


class RegionalRuntimeInstaller(Protocol):
    """Adapter-owned installation called after Forge finalises the active model."""

    def install(
        self,
        *,
        runtime: "RegionalRuntime",
        batch: RegionalBatchCompilation,
        model_context: Any,
    ) -> Any | None: ...


@dataclass(slots=True)
class _OwnedResource:
    value: Any
    releaser: Callable[[Any], None] | None

    def close(self) -> None:
        if self.releaser is not None:
            self.releaser(self.value)
            return
        for method_name in ("close", "remove", "release"):
            method = getattr(self.value, method_name, None)
            if callable(method):
                method()
                return


class RegionalRuntime:
    """Own one Regional job and every temporary resource installed by it."""

    def __init__(
        self,
        authorized_plan: AuthorizedRegionalPlan,
        *,
        prompt_expansion: PromptExpansionService | None = None,
        token_counter: PromptTokenCounter | None = None,
        mask_cache_entries: int = 8,
        mask_cache_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        if not isinstance(authorized_plan, AuthorizedRegionalPlan):
            raise TypeError("RegionalRuntime requires an authorized canonical plan")
        self.authorized_plan = authorized_plan
        self.prompt_expansion = prompt_expansion
        self.token_counter = token_counter
        self._mask_cache = MaskCompilerCache(max_entries=mask_cache_entries, max_bytes=mask_cache_bytes)
        self._active_batch: RegionalBatchCompilation | None = None
        self._resources: list[_OwnedResource] = []
        self._closed = False

    @property
    def plan(self):
        return self.authorized_plan.plan

    @property
    def active_batch(self) -> RegionalBatchCompilation | None:
        return self._active_batch

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def owned_resource_count(self) -> int:
        return len(self._resources)

    def __enter__(self) -> "RegionalRuntime":
        if self._closed:
            raise RuntimeError("A closed Regional runtime cannot be reused")
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc is None:
                raise
            add_note = getattr(exc, "add_note", None)
            if callable(add_note):
                add_note(f"Regional cleanup also failed: {cleanup_error}")
        return False

    def begin_batch(
        self,
        context: RegionalBatchContext,
        *,
        model_context: Any,
        installer: RegionalRuntimeInstaller | None = None,
    ) -> RegionalBatchCompilation:
        if self._closed:
            raise RuntimeError("A closed Regional runtime cannot begin a batch")
        if self._active_batch is not None:
            raise RuntimeError("The previous Regional batch has not been closed")

        resolved_seeds = resolve_forge_seed_sequence(
            self.plan,
            context.seeds,
            start_index=context.image_start,
        )
        compiled_prompts = tuple(
            compile_prompt_plan(
                self.plan,
                base_seed=seed.image_seed,
                batch_index=image_index,
                expansion_service=self.prompt_expansion,
                token_counter=self.token_counter,
            )
            for seed, image_index in zip(resolved_seeds, context.image_indices)
        )
        batch = RegionalBatchCompilation(
            context=context,
            prompts=compiled_prompts,
            seeds=resolved_seeds,
        )
        self._active_batch = batch

        if installer is not None:
            try:
                installed = installer.install(runtime=self, batch=batch, model_context=model_context)
                if installed is not None:
                    self.own_resource(installed)
            except BaseException as install_error:
                try:
                    self.end_batch()
                except BaseException as cleanup_error:
                    add_note = getattr(install_error, "add_note", None)
                    if callable(add_note):
                        add_note(f"Regional cleanup also failed: {cleanup_error}")
                raise
        return self._active_batch

    def compile_masks(
        self,
        resolutions: Iterable[tuple[int, int]],
        *,
        dtype: str = "float32",
    ) -> tuple[CompiledMaskSet, ...]:
        if self._closed:
            raise RuntimeError("A closed Regional runtime cannot compile masks")
        if self._active_batch is None:
            raise RuntimeError("Regional masks can only be compiled for an active batch")
        compiled = compile_mask_pyramid(
            self.plan,
            resolutions,
            adapter_id=self.authorized_plan.adapter_id,
            device="cpu",
            dtype=dtype,
            capabilities=ValidationCapabilities(
                overlap_policies=self.authorized_plan.engine.overlap_policies,
                uncovered_policies=self.authorized_plan.engine.uncovered_policies,
            ),
            cache=self._mask_cache,
        )
        known = {item.key for item in self._active_batch.masks}
        merged = (*self._active_batch.masks, *(item for item in compiled if item.key not in known))
        self._active_batch = replace(self._active_batch, masks=merged)
        return compiled

    def own_resource(
        self,
        value: Any,
        releaser: Callable[[Any], None] | None = None,
    ) -> Any:
        """Retain a hook, patch, or tensor owner until the active batch closes."""

        if self._closed:
            raise RuntimeError("A closed Regional runtime cannot own resources")
        if self._active_batch is None:
            raise RuntimeError("Regional resources require an active batch")
        self._resources.append(_OwnedResource(value=value, releaser=releaser))
        return value

    def materialize(self, factory: Callable[[RegionalBatchCompilation], Any], *, releaser=None) -> Any:
        """Lazily allocate adapter tensors after model device and dtype are final."""

        if self._active_batch is None:
            raise RuntimeError("Regional tensors require an active batch")
        return self.own_resource(factory(self._active_batch), releaser=releaser)

    def end_batch(self) -> None:
        if self._active_batch is None and not self._resources:
            return
        resources = tuple(reversed(self._resources))
        self._resources.clear()
        self._active_batch = None
        first_error: BaseException | None = None
        for resource in resources:
            try:
                resource.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def close(self) -> None:
        if self._closed:
            return
        error: BaseException | None = None
        try:
            self.end_batch()
        except BaseException as cleanup_error:
            error = cleanup_error
        finally:
            self._mask_cache.clear()
            self._closed = True
        if error is not None:
            raise error
