"""Forge txt2img-compatible processing boundary for Regional jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from modules import processing, scripts
from modules_nocturne.regional.errors import PlanError
from modules_nocturne.regional.capabilities import capability_service
from modules_nocturne.regional.generation import AuthorizedRegionalPlan, authorize_generation
from modules_nocturne.regional.model import RegionalGenerationPlan
from modules_nocturne.regional.project import MetadataBundle, build_metadata
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
        "denoising_strength",
        "hr_scale",
        "hr_upscaler",
        "hr_second_pass_steps",
        "hr_resize_x",
        "hr_resize_y",
        "hr_cfg",
        "hr_distilled_cfg",
    }
)


def forge_fields_from_plan(
    plan_or_authorized: RegionalGenerationPlan | AuthorizedRegionalPlan,
) -> Mapping[str, Any]:
    """Translate canonical data once at the Forge processing boundary."""

    if isinstance(plan_or_authorized, AuthorizedRegionalPlan):
        plan = plan_or_authorized.plan
    elif isinstance(plan_or_authorized, RegionalGenerationPlan):
        plan = plan_or_authorized
    else:
        raise TypeError("Regional processing requires a canonical plan")
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
            "enable_hr": bool(options.get("hires_enabled", False)),
            "denoising_strength": float(options.get("hires_denoising_strength", 0.6)),
            "hr_scale": float(options.get("hires_scale", 2.0)),
            "hr_upscaler": str(options.get("hires_upscaler", "Latent")),
            "hr_second_pass_steps": int(options.get("hires_steps", 0)),
            "hr_resize_x": int(options.get("hires_width", 0)),
            "hr_resize_y": int(options.get("hires_height", 0)),
            "hr_cfg": float(options.get("hires_cfg_scale", options.get("cfg_scale", 6.0))),
            "hr_distilled_cfg": float(options.get("hires_distilled_cfg_scale", 3.0)),
        }
    )


@dataclass(repr=False)
class StableDiffusionProcessingRegional(processing.StableDiffusionProcessingTxt2Img):
    """A contained Regional identity that retains Forge's normal job lifecycle."""

    authorized_plan: AuthorizedRegionalPlan | None = field(default=None, repr=False)
    regional_plan: RegionalGenerationPlan | None = field(default=None, repr=False)
    accepted_issue_codes: frozenset[str] = field(default_factory=frozenset, repr=False)
    accepted_fallbacks: tuple[str, ...] = field(default=(), repr=False)
    accept_required_fallbacks: bool = field(default=False, repr=False)
    runtime_installer: RegionalRuntimeInstaller | None = field(default=None, repr=False)
    regional_runtime: RegionalRuntime | None = field(default=None, init=False, repr=False)
    regional_metadata_bundle: MetadataBundle | None = field(default=None, init=False, repr=False)
    regional_engine_runtime_options: Mapping[str, Any] = field(init=False, repr=False)
    regional_final_prompts: list[Any] = field(init=False, repr=False)
    regional_resolved_seeds: list[Any] = field(init=False, repr=False)

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
        forge_fields.setdefault("hr_additional_modules", ["Use same choices"])
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

    @classmethod
    def from_plan(
        cls,
        plan: RegionalGenerationPlan,
        *,
        accepted_issue_codes: frozenset[str] = frozenset(),
        accepted_fallbacks: tuple[str, ...] = (),
        accept_required_fallbacks: bool = False,
        scripts_runner=None,
        script_args=(),
        is_api: bool = False,
        **forge_fields,
    ) -> "StableDiffusionProcessingRegional":
        """Create a job that Forge authorizes after its normal model reload."""

        conflicts = _CANONICAL_PROCESSING_FIELDS.intersection(forge_fields)
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise TypeError(f"Canonical Regional processing fields cannot be overridden: {names}")
        forge_fields.setdefault("hr_additional_modules", ["Use same choices"])
        instance = cls(
            **forge_fields_from_plan(plan),
            **forge_fields,
            regional_plan=plan,
            accepted_issue_codes=frozenset(accepted_issue_codes),
            accepted_fallbacks=tuple(accepted_fallbacks),
            accept_required_fallbacks=bool(accept_required_fallbacks),
        )
        instance.is_api = bool(is_api)
        if scripts_runner is not None:
            instance.scripts = scripts_runner
            instance.script_args = tuple(script_args)
        return instance

    def __post_init__(self):
        super().__post_init__()
        plan = (
            self.authorized_plan.plan
            if isinstance(self.authorized_plan, AuthorizedRegionalPlan)
            else self.regional_plan
        )
        if not isinstance(plan, RegionalGenerationPlan):
            raise TypeError("Regional processing requires a canonical plan")
        self.regional_plan = plan
        expected = forge_fields_from_plan(plan)
        mismatches = tuple(
            name for name in _CANONICAL_PROCESSING_FIELDS if getattr(self, name) != expected[name]
        )
        if mismatches:
            raise PlanError(
                "processing.canonical_mismatch",
                "$",
                f"Forge processing fields differ from the canonical plan: {', '.join(sorted(mismatches))}",
            )
        self.regional_final_prompts = []
        self.regional_resolved_seeds = []
        from modules import shared

        self.regional_engine_runtime_options = MappingProxyType(
            {
                "attention_memory_budget_mb": int(
                    getattr(
                        shared.opts,
                        "nocturne_regional_attention_memory_mb",
                        64,
                    )
                )
            }
        )
        if self.authorized_plan is not None:
            self._initialize_authorized_runtime()

    def _initialize_authorized_runtime(self) -> None:
        if self.regional_runtime is not None:
            return
        authorized = self.authorized_plan
        if authorized is None:
            report = capability_service.report(self.sd_model)
            accepted_fallbacks = self.accepted_fallbacks
            if self.accept_required_fallbacks and report.status == "supported":
                requested = self.regional_plan.engine.requested
                engines = (
                    report.eligible_engines
                    if requested == "auto"
                    else tuple(
                        engine
                        for engine in report.eligible_engines
                        if engine.engine_id == requested
                    )
                )
                accepted_fallbacks = tuple(
                    dict.fromkeys(
                        (
                            *report.expected_fallbacks,
                            *(
                                fallback
                                for engine in engines
                                for fallback in engine.expected_fallbacks
                            ),
                        )
                    )
                )
            authorized = authorize_generation(
                self.regional_plan,
                report,
                accepted_issue_codes=self.accepted_issue_codes,
                accepted_fallbacks=accepted_fallbacks,
            )
            engine = capability_service.engines.get(authorized.engine.engine_id)
            installer_factory = getattr(engine, "runtime_installer", None)
            if engine is None or not callable(installer_factory):
                raise RuntimeError("The selected Regional engine has no runtime installer")
            adapter = capability_service.adapters.get(authorized.adapter_id)
            if adapter is None:
                raise RuntimeError("The authorized Regional adapter is no longer registered")
            self.authorized_plan = authorized
            self.runtime_installer = installer_factory(adapter=adapter)

        self.regional_runtime = RegionalRuntime(authorized)
        self.regional_metadata_bundle = build_metadata(
            authorized.plan,
            selected_engine=authorized.engine.engine_id,
            engine_version=authorized.engine.engine_version,
            engine_runtime_options=self.regional_engine_runtime_options,
            adapter_id=authorized.adapter_id,
            accepted_fallbacks=authorized.accepted_fallbacks,
        )
        self.extra_generation_params.update(self.regional_metadata_bundle.fields)
        self.extra_generation_params.update(
            {
                "Nocturne Regional Engine Version": authorized.engine.engine_version,
                "Nocturne Regional Attention Memory Budget": (
                    f"{self.regional_engine_runtime_options['attention_memory_budget_mb']} MiB"
                ),
            }
        )
        if authorized.engine.cost_warning:
            self.extra_generation_params["Nocturne Regional Warnings"] = (
                authorized.engine.cost_warning
            )
        if authorized.accepted_fallbacks:
            self.extra_generation_params["Nocturne Regional Accepted Fallbacks"] = "; ".join(
                authorized.accepted_fallbacks
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
        self._initialize_authorized_runtime()
        if self.regional_runtime is None:
            raise RuntimeError("Regional runtime was not initialized")
        self.regional_runtime.begin_batch(
            self._batch_context(),
            model_context=self.sd_model,
        )
        try:
            result = super().setup_conds()
            if self.runtime_installer is not None and getattr(self.runtime_installer, "requires_conditioning", False):
                conditioning = self.regional_runtime.build_conditioning(
                    model_context=self.sd_model,
                    steps=int(self.firstpass_steps),
                    width=int(self.width),
                    height=int(self.height),
                    distilled_cfg_scale=float(self.distilled_cfg_scale),
                )
                self.regional_final_prompts.extend(conditioning.final_prompts)
            active_batch = self.regional_runtime.active_batch
            if active_batch is not None:
                self.regional_resolved_seeds.extend(active_batch.seeds)
            return result
        except BaseException as conditioning_error:
            try:
                self.regional_runtime.end_batch()
            except BaseException as cleanup_error:
                add_note = getattr(conditioning_error, "add_note", None)
                if callable(add_note):
                    add_note(f"Regional cleanup also failed: {cleanup_error}")
            raise

    def prepare_sampling_context(
        self,
        *,
        x,
        noise,
        conditioning,
        unconditional_conditioning,
        pass_name,
    ):
        if pass_name == "hires":
            if self.regional_plan.passes.hires != "recompile":
                raise PlanError(
                    "passes.hires.policy_unsupported",
                    "$.passes.hires",
                    f"Regional hires does not support policy {self.regional_plan.passes.hires!r}",
                )
            if self.regional_runtime is None:
                raise RuntimeError("Regional runtime was not initialized")
            self.regional_runtime.transition_pass(
                width=int(self.hr_upscale_to_x),
                height=int(self.hr_upscale_to_y),
                pass_name="hires",
            )
            if self.runtime_installer is not None and getattr(
                self.runtime_installer, "requires_conditioning", False
            ):
                hires_steps = int(self.hr_second_pass_steps or self.steps)
                hires_conditioning = self.regional_runtime.build_conditioning(
                    model_context=self.sd_model,
                    steps=int(self.steps),
                    hires_steps=hires_steps,
                    width=int(self.hr_upscale_to_x),
                    height=int(self.hr_upscale_to_y),
                    distilled_cfg_scale=float(self.hr_distilled_cfg),
                )
                self.regional_final_prompts.extend(hires_conditioning.final_prompts)
            self.extra_generation_params["Nocturne Regional Base Engine"] = (
                self.authorized_plan.engine.engine_id
            )
            self.extra_generation_params["Nocturne Regional Hires Engine"] = (
                self.authorized_plan.engine.engine_id
            )
            self.extra_generation_params["Nocturne Regional Hires Policy"] = "recompile"
        elif pass_name != "base":
            raise PlanError(
                "passes.runtime.unsupported",
                "$.passes",
                f"Regional processing does not support the {pass_name!r} sampling pass",
            )
        if self.runtime_installer is None:
            raise RuntimeError("Regional sampling requires a runtime engine installer")
        if self.regional_runtime is None:
            raise RuntimeError("Regional runtime was not initialized")
        self.regional_runtime.install_engine(
            installer=self.runtime_installer,
            model_context=self.sd_model,
        )

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
        if self.regional_runtime is None or self.regional_runtime.active_batch is None:
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
        if self.regional_runtime is not None:
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
