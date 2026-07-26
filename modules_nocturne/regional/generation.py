"""Authorization token required by model-specific Regional generation code."""

from dataclasses import dataclass

from modules_nocturne.regional.capabilities import CapabilityReport, EngineCapabilities
from modules_nocturne.regional.errors import IssueSeverity, PlanError, PlanValidationError, ValidationReport
from modules_nocturne.regional.model import RegionalGenerationPlan
from modules_nocturne.regional.serialization import plan_hash
from modules_nocturne.regional.validation import ValidationCapabilities, validate_plan


@dataclass(frozen=True, slots=True)
class AuthorizedRegionalPlan:
    plan: RegionalGenerationPlan
    plan_hash: str
    adapter_id: str
    engine: EngineCapabilities
    validation: ValidationReport
    accepted_issue_codes: frozenset[str]
    accepted_fallbacks: tuple[str, ...]


def authorize_generation(
    plan: RegionalGenerationPlan,
    capability_report: CapabilityReport,
    *,
    selected_engine_id: str | None = None,
    accepted_issue_codes: frozenset[str] = frozenset(),
    accepted_fallbacks: tuple[str, ...] = (),
) -> AuthorizedRegionalPlan:
    """Reject unsupported or unapproved plans before an adapter can sample."""

    if capability_report.status != "supported" or capability_report.adapter_id is None:
        raise PlanError(
            capability_report.reason_code or "model.unsupported",
            "$.engine",
            capability_report.reason or "The loaded model does not support Regional generation",
        )

    requested = selected_engine_id or plan.engine.requested
    if requested == "auto":
        engine = capability_report.eligible_engines[0] if capability_report.eligible_engines else None
    else:
        engine = next((item for item in capability_report.eligible_engines if item.engine_id == requested), None)
    if engine is None:
        raise PlanError(
            "engine.requested.unavailable",
            "$.engine.requested",
            f"Regional engine {requested!r} is not available for the loaded model",
        )

    required_fallbacks = tuple(dict.fromkeys((*capability_report.expected_fallbacks, *engine.expected_fallbacks)))
    missing_fallbacks = tuple(item for item in required_fallbacks if item not in accepted_fallbacks)
    if missing_fallbacks:
        raise PlanError(
            "engine.fallback.approval_required",
            "$.engine",
            f"Regional generation requires approval for: {', '.join(missing_fallbacks)}",
        )

    report = validate_plan(
        plan,
        capabilities=ValidationCapabilities(
            overlap_policies=engine.overlap_policies,
            uncovered_policies=engine.uncovered_policies,
        ),
    )
    if not report.valid:
        raise PlanValidationError(report)
    missing_approvals = tuple(
        issue.code
        for issue in report.issues
        if issue.severity == IssueSeverity.APPROVAL_REQUIRED and issue.code not in accepted_issue_codes
    )
    if missing_approvals:
        raise PlanError(
            "plan.approval_required",
            "$",
            f"Regional generation requires approval for: {', '.join(missing_approvals)}",
        )

    return AuthorizedRegionalPlan(
        plan=plan,
        plan_hash=plan_hash(plan),
        adapter_id=capability_report.adapter_id,
        engine=engine,
        validation=report,
        accepted_issue_codes=frozenset(accepted_issue_codes),
        accepted_fallbacks=tuple(accepted_fallbacks),
    )
