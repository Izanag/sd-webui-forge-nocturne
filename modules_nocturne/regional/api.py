"""FastAPI transport models and routes for Regional validation."""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from modules_nocturne.regional.capabilities import CapabilityReport, CapabilityService, capability_service
from modules_nocturne.regional.errors import PlanError, ValidationIssue, ValidationReport
from modules_nocturne.regional.serialization import load_plan, plan_hash, plan_to_dict
from modules_nocturne.regional.validation import ValidationCapabilities, validate_plan


class RegionalValidationIssueResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    path: str
    message: str
    severity: str
    details: dict[str, Any] = Field(default_factory=dict)


class RegionalEngineCapabilityResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine_id: str
    engine_version: str
    overlap_policies: list[str]
    uncovered_policies: list[str]
    supported_fields: list[str]
    expected_fallbacks: list[str]
    cost_warning: str | None = None


class RegionalCapabilityResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    adapter_id: str | None = None
    eligible_engines: list[RegionalEngineCapabilityResponse] = Field(default_factory=list)
    unsupported_fields: list[str] = Field(default_factory=list)
    expected_fallbacks: list[str] = Field(default_factory=list)
    reason_code: str | None = None
    reason: str | None = None


class RegionalValidateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan: dict[str, Any]
    max_enabled_regions: int | None = Field(default=None, ge=1, le=64)


class RegionalValidateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    requires_approval: bool
    plan_hash: str | None = None
    normalized_plan: dict[str, Any] | None = None
    capabilities: RegionalCapabilityResponse
    issues: list[RegionalValidationIssueResponse] = Field(default_factory=list)


class RegionalGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan: dict[str, Any]
    accepted_issue_codes: list[str] = Field(default_factory=list)
    accepted_fallbacks: list[str] = Field(default_factory=list)
    send_images: bool = True
    save_images: bool = False
    force_task_id: str | None = None


class RegionalResolvedPromptResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_index: int
    owner: str
    region_id: str | None = None
    polarity: str
    entries: list[dict[str, Any]] = Field(default_factory=list)


class RegionalGenerateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    images: list[str] = Field(default_factory=list)
    parameters: dict[str, Any]
    info: str
    task_id: str
    plan_hash: str
    normalized_plan: dict[str, Any]
    adapter_id: str
    selected_engine: str
    engine_version: str
    accepted_fallbacks: list[str] = Field(default_factory=list)
    warnings: list[RegionalValidationIssueResponse] = Field(default_factory=list)
    final_prompts: list[RegionalResolvedPromptResponse] = Field(default_factory=list)
    metadata: dict[str, Any]


def _capability_response(report: CapabilityReport) -> RegionalCapabilityResponse:
    return RegionalCapabilityResponse(
        status=report.status,
        adapter_id=report.adapter_id,
        eligible_engines=[
            RegionalEngineCapabilityResponse(
                engine_id=engine.engine_id,
                engine_version=engine.engine_version,
                overlap_policies=sorted(policy.value for policy in engine.overlap_policies),
                uncovered_policies=sorted(policy.value for policy in engine.uncovered_policies),
                supported_fields=sorted(engine.supported_fields),
                expected_fallbacks=list(engine.expected_fallbacks),
                cost_warning=engine.cost_warning,
            )
            for engine in report.eligible_engines
        ],
        unsupported_fields=list(report.unsupported_fields),
        expected_fallbacks=list(report.expected_fallbacks),
        reason_code=report.reason_code,
        reason=report.reason,
    )


def _issue_response(issue: ValidationIssue) -> RegionalValidationIssueResponse:
    return RegionalValidationIssueResponse(**issue.as_dict())


def _request_dict(request: BaseModel) -> dict[str, Any]:
    if hasattr(request, "model_dump"):
        return request.model_dump()
    return request.dict()


def _resolved_prompt_response(record) -> RegionalResolvedPromptResponse:
    return RegionalResolvedPromptResponse(
        image_index=record.image_index,
        owner=record.owner.kind,
        region_id=str(record.owner.region_id) if record.owner.region_id is not None else None,
        polarity=record.polarity,
        entries=[
            {
                "end_at_step": entry.end_at_step,
                "text": entry.text,
            }
            for entry in record.entries
        ],
    )


class RegionalApi:
    def __init__(
        self,
        *,
        service: CapabilityService,
        model_provider: Callable[[], Any],
        max_regions_provider: Callable[[], int],
        queue_lock=None,
    ) -> None:
        self.service = service
        self.model_provider = model_provider
        self.max_regions_provider = max_regions_provider
        self.queue_lock = queue_lock

    def capabilities(self) -> RegionalCapabilityResponse:
        return _capability_response(self.service.report(self.model_provider()))

    def validate(self, request: RegionalValidateRequest) -> RegionalValidateResponse:
        capability_report = self.service.report(self.model_provider())
        capability_response = _capability_response(capability_report)

        try:
            plan = load_plan(request.plan)
        except PlanError as error:
            return RegionalValidateResponse(
                valid=False,
                requires_approval=False,
                capabilities=capability_response,
                issues=[_issue_response(error.issue)],
            )

        capabilities = None
        if capability_report.status == "supported":
            requested = plan.engine.requested.lower()
            eligible = capability_report.eligible_engines
            selected = next((engine for engine in eligible if engine.engine_id.lower() == requested), None)
            if requested == "auto" and eligible:
                selected = eligible[0]
            if selected is not None:
                capabilities = ValidationCapabilities(
                    overlap_policies=selected.overlap_policies,
                    uncovered_policies=selected.uncovered_policies,
                )

        report = validate_plan(
            plan,
            max_enabled_regions=request.max_enabled_regions or self.max_regions_provider(),
            capabilities=capabilities,
        )

        issues = list(report.issues)
        if capability_report.status != "supported":
            issues.append(
                ValidationIssue(
                    code=capability_report.reason_code or "model.unsupported",
                    path="$.engine.requested",
                    message=capability_report.reason or "Regional generation is unsupported",
                )
            )
        elif capabilities is None:
            issues.append(
                ValidationIssue(
                    code="engine.requested.unsupported",
                    path="$.engine.requested",
                    message=f"Requested Regional engine {plan.engine.requested!r} is not eligible",
                )
            )

        combined = ValidationReport(tuple(issues))
        return RegionalValidateResponse(
            valid=combined.valid,
            requires_approval=combined.requires_approval,
            plan_hash=plan_hash(plan),
            normalized_plan=plan_to_dict(plan),
            capabilities=capability_response,
            issues=[_issue_response(issue) for issue in combined.issues],
        )

    def generate(self, request: RegionalGenerateRequest) -> RegionalGenerateResponse:
        from contextlib import closing

        from fastapi import HTTPException

        from modules import processing, shared
        from modules.api.api import encode_pil_to_base64
        from modules.progress import add_task_to_queue, create_task_id, finish_task, start_task
        from modules_nocturne.regional.processing import StableDiffusionProcessingRegional
        from modules_nocturne.regional.project import build_metadata, save_sidecar

        try:
            plan = load_plan(request.plan)
        except PlanError as error:
            raise HTTPException(status_code=422, detail=error.issue.as_dict()) from error

        task_id = request.force_task_id or create_task_id("regional")
        add_task_to_queue(task_id)
        lock = self.queue_lock
        if lock is None:
            raise HTTPException(status_code=500, detail="Regional generation queue is unavailable")

        try:
            with lock:
                start_task(task_id)
                shared.state.begin(job="regional")
                try:
                    with closing(
                        StableDiffusionProcessingRegional.from_plan(
                            plan,
                            sd_model=self.model_provider(),
                            outpath_samples=shared.opts.outdir_samples
                            or shared.opts.outdir_txt2img_samples,
                            outpath_grids=shared.opts.outdir_grids
                            or shared.opts.outdir_txt2img_grids,
                            do_not_save_samples=not request.save_images,
                            do_not_save_grid=not request.save_images,
                            is_api=True,
                            accepted_issue_codes=frozenset(request.accepted_issue_codes),
                            accepted_fallbacks=tuple(request.accepted_fallbacks),
                        )
                    ) as regional:
                        processed = processing.process_images(regional)
                        processing.process_extra_images(processed)
                        authorized = regional.authorized_plan
                        if authorized is None:
                            raise RuntimeError("Regional generation completed without authorization")
                        metadata = build_metadata(
                            authorized.plan,
                            selected_engine=authorized.engine.engine_id,
                            engine_version=authorized.engine.engine_version,
                            engine_runtime_options=regional.regional_engine_runtime_options,
                            adapter_id=authorized.adapter_id,
                            accepted_fallbacks=authorized.accepted_fallbacks,
                            resolved_seeds=tuple(regional.regional_resolved_seeds),
                        )
                        output_images = processed.images + processed.extra_images
                        for image in output_images:
                            image.info.update(metadata.fields)
                            saved_path = getattr(image, "already_saved_as", None)
                            if metadata.sidecar_required and saved_path:
                                save_sidecar(saved_path, metadata)
                        final_prompts = tuple(regional.regional_final_prompts)
                finally:
                    shared.state.end()
                    shared.total_tqdm.clear()
        except HTTPException:
            raise
        except PlanError as error:
            raise HTTPException(status_code=422, detail=error.issue.as_dict()) from error
        finally:
            finish_task(task_id)

        images = (
            [
                encode_pil_to_base64(image).decode("ascii")
                for image in processed.images + processed.extra_images
            ]
            if request.send_images
            else []
        )
        warnings = [
            _issue_response(issue)
            for issue in authorized.validation.issues
            if issue.severity.value != "error"
        ]
        return RegionalGenerateResponse(
            images=images,
            parameters=_request_dict(request),
            info=processed.js(),
            task_id=task_id,
            plan_hash=authorized.plan_hash,
            normalized_plan=plan_to_dict(authorized.plan),
            adapter_id=authorized.adapter_id,
            selected_engine=authorized.engine.engine_id,
            engine_version=authorized.engine.engine_version,
            accepted_fallbacks=list(authorized.accepted_fallbacks),
            warnings=warnings,
            final_prompts=[
                _resolved_prompt_response(record)
                for record in final_prompts
            ],
            metadata=dict(metadata.sidecar_document),
        )


def register_routes(api_host) -> RegionalApi:
    from modules import shared

    regional_api = RegionalApi(
        service=capability_service,
        model_provider=lambda: shared.sd_model,
        max_regions_provider=lambda: shared.opts.nocturne_regional_max_regions,
        queue_lock=api_host.queue_lock,
    )
    api_host.add_api_route(
        "/sdapi/v1/regional/capabilities",
        regional_api.capabilities,
        methods=["GET"],
        response_model=RegionalCapabilityResponse,
    )
    api_host.add_api_route(
        "/sdapi/v1/regional/validate",
        regional_api.validate,
        methods=["POST"],
        response_model=RegionalValidateResponse,
    )
    api_host.add_api_route(
        "/sdapi/v1/regional",
        regional_api.generate,
        methods=["POST"],
        response_model=RegionalGenerateResponse,
    )
    return regional_api
