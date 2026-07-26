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


class RegionalApi:
    def __init__(
        self,
        *,
        service: CapabilityService,
        model_provider: Callable[[], Any],
        max_regions_provider: Callable[[], int],
    ) -> None:
        self.service = service
        self.model_provider = model_provider
        self.max_regions_provider = max_regions_provider

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


def register_routes(api_host) -> RegionalApi:
    from modules import shared

    regional_api = RegionalApi(
        service=capability_service,
        model_provider=lambda: shared.sd_model,
        max_regions_provider=lambda: shared.opts.nocturne_regional_max_regions,
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
    return regional_api
