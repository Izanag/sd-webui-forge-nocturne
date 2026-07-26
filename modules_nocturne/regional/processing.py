"""Forge txt2img-compatible processing boundary for Regional jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from modules import processing, scripts
from modules_nocturne.regional.errors import PlanError
from modules_nocturne.regional.generation import AuthorizedRegionalPlan
from modules_nocturne.regional.runtime import RegionalBatchContext, RegionalRuntime, RegionalRuntimeInstaller


_CANONICAL_PROCESSING_FIELDS = frozenset(
    {
        "prompt",
        "negative_prompt",
        "width",
        "height",
        "sampler_name",
        "scheduler",
        "steps",
        "cfg_scale",
        "batch_size",
        "n_iter",
        "seed",
        "enable_hr",
    }
)


def forge_fields_from_plan(authorized_plan: AuthorizedRegionalPlan) -> Mapping[str, Any]:
    """Translate canonical data once at the Forge processing boundary."""

    if not isinstance(authorized_plan, AuthorizedRegionalPlan):
        raise TypeError("Regional processing requires an authorized canonical plan")
    plan = authorized_plan.plan
    options = plan.engine.options
    return MappingProxyType(
        {
            "prompt": plan.global_prompt.positive,
            "negative_prompt": plan.global_prompt.negative,
            "width": plan.canvas.width,
            "height": plan.canvas.height,
            "sampler_name": options.get("sampler", "Euler a"),
            "scheduler": options.get("scheduler", "Automatic"),
            "steps": int(options.get("steps", 32)),
            "cfg_scale": float(options.get("cfg_scale", 6.0)),
            "batch_size": int(options.get("batch_size", 1)),
            "n_iter": int(options.get("batch_count", 1)),
            "seed": int(options.get("seed", -1)),
            "enable_hr": False,
        }
    )


@dataclass(repr=False)
class StableDiffusionProcessingRegional(processing.StableDiffusionProcessingTxt2Img):
    """A contained Regional identity that retains Forge's normal job lifecycle."""

    authorized_plan: AuthorizedRegionalPlan | None = field(default=None, repr=False)
    runtime_installer: RegionalRuntimeInstaller | None = field(default=None, repr=False)
    regional_runtime: RegionalRuntime = field(init=False, repr=False)

    generation_context = scripts.GenerationContext.REGIONAL
    is_regional = True
    api_route = "/sdapi/v1/regional"

    @classmethod
    def from_authorized_plan(
        cls,
        authorized_plan: AuthorizedRegionalPlan,
        *,
        scripts_runner=None,
        script_args=(),
        is_api: bool = False,
        runtime_installer: RegionalRuntimeInstaller | None = None,
        **forge_fields,
    ) -> "StableDiffusionProcessingRegional":
        conflicts = _CANONICAL_PROCESSING_FIELDS.intersection(forge_fields)
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise TypeError(f"Canonical Regional processing fields cannot be overridden: {names}")
        instance = cls(
            **forge_fields_from_plan(authorized_plan),
            **forge_fields,
            authorized_plan=authorized_plan,
            runtime_installer=runtime_installer,
        )
        instance.is_api = bool(is_api)
        if scripts_runner is not None:
            instance.scripts = scripts_runner
            instance.script_args = tuple(script_args)
        return instance

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.authorized_plan, AuthorizedRegionalPlan):
            raise TypeError("Regional processing requires an authorized canonical plan")
        expected = forge_fields_from_plan(self.authorized_plan)
        mismatches = tuple(
            name
            for name in _CANONICAL_PROCESSING_FIELDS
            if name != "enable_hr" and getattr(self, name) != expected[name]
        )
        if mismatches:
            raise PlanError(
                "processing.canonical_mismatch",
                "$",
                f"Forge processing fields differ from the canonical plan: {', '.join(sorted(mismatches))}",
            )
        if self.enable_hr:
            raise PlanError(
                "passes.hires.unsupported",
                "$.passes.hires",
                "Hires processing is unavailable until the active Regional adapter proves pass support",
            )
        self.regional_runtime = RegionalRuntime(self.authorized_plan)
        self.extra_generation_params.update(
            {
                "Nocturne Regional Hash": self.authorized_plan.plan_hash,
                "Nocturne Regional Adapter": self.authorized_plan.adapter_id,
                "Nocturne Regional Engine": self.authorized_plan.engine.engine_id,
                "Nocturne Regional Engine Version": self.authorized_plan.engine.engine_version,
            }
        )

    def _batch_context(self) -> RegionalBatchContext:
        batch_number = int(self.iteration)
        image_start = batch_number * int(self.batch_size)
        return RegionalBatchContext(
            batch_number=batch_number,
            image_start=image_start,
            width=int(self.width),
            height=int(self.height),
            prompts=tuple(self.prompts),
            negative_prompts=tuple(self.negative_prompts),
            seeds=tuple(self.seeds),
            subseeds=tuple(self.subseeds),
        )

    def setup_conds(self):
        self.regional_runtime.begin_batch(
            self._batch_context(),
            model_context=self.sd_model,
            installer=self.runtime_installer,
        )
        try:
            result = super().setup_conds()
            if self.runtime_installer is not None and getattr(self.runtime_installer, "requires_conditioning", False):
                self.regional_runtime.build_conditioning(
                    model_context=self.sd_model,
                    steps=int(self.firstpass_steps),
                    width=int(self.width),
                    height=int(self.height),
                    distilled_cfg_scale=float(self.distilled_cfg_scale),
                )
            return result
        except BaseException as conditioning_error:
            try:
                self.regional_runtime.end_batch()
            except BaseException as cleanup_error:
                add_note = getattr(conditioning_error, "add_note", None)
                if callable(add_note):
                    add_note(f"Regional cleanup also failed: {cleanup_error}")
            raise

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
        if self.regional_runtime.active_batch is None:
            raise RuntimeError("Regional sampling requires a prepared batch")
        try:
            result = super().sample(
                conditioning=conditioning,
                unconditional_conditioning=unconditional_conditioning,
                seeds=seeds,
                subseeds=subseeds,
                subseed_strength=subseed_strength,
                prompts=prompts,
            )
        except BaseException as sampling_error:
            try:
                self.regional_runtime.end_batch()
            except BaseException as cleanup_error:
                add_note = getattr(sampling_error, "add_note", None)
                if callable(add_note):
                    add_note(f"Regional cleanup also failed: {cleanup_error}")
            raise
        else:
            self.regional_runtime.end_batch()
            return result

    def close(self):
        runtime_error: BaseException | None = None
        try:
            self.regional_runtime.close()
        except BaseException as error:
            runtime_error = error
        try:
            super().close()
        except BaseException as base_error:
            if runtime_error is not None:
                add_note = getattr(base_error, "add_note", None)
                if callable(add_note):
                    add_note(f"Regional runtime cleanup also failed: {runtime_error}")
            raise
        if runtime_error is not None:
            raise runtime_error
