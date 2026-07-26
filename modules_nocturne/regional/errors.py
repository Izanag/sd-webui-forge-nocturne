"""Stable, serialisable Regional error contracts."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class IssueSeverity(str, Enum):
    ERROR = "error"
    APPROVAL_REQUIRED = "approval_required"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    path: str
    message: str
    severity: IssueSeverity = IssueSeverity.ERROR
    details: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "path": self.path,
            "message": self.message,
            "severity": self.severity.value,
        }
        if self.details:
            result["details"] = dict(self.details)
        return result


@dataclass(frozen=True, slots=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def valid(self) -> bool:
        return not any(issue.severity == IssueSeverity.ERROR for issue in self.issues)

    @property
    def requires_approval(self) -> bool:
        return any(issue.severity == IssueSeverity.APPROVAL_REQUIRED for issue in self.issues)

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == IssueSeverity.ERROR)

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "requires_approval": self.requires_approval,
            "issues": [issue.as_dict() for issue in self.issues],
        }


class PlanError(ValueError):
    """Raised when input cannot be represented as a Regional plan."""

    def __init__(self, code: str, path: str, message: str):
        super().__init__(message)
        self.issue = ValidationIssue(code=code, path=path, message=message)

    @property
    def code(self) -> str:
        return self.issue.code

    @property
    def path(self) -> str:
        return self.issue.path

    def as_dict(self) -> dict[str, Any]:
        return self.issue.as_dict()


class PlanValidationError(PlanError):
    """Raised when a complete plan fails pre-compilation validation."""

    def __init__(self, report: ValidationReport):
        first = report.errors[0] if report.errors else ValidationIssue(
            code="plan.validation_failed",
            path="$",
            message="Regional plan validation failed",
        )
        super().__init__(first.code, first.path, first.message)
        self.report = report

    def as_dict(self) -> dict[str, Any]:
        return self.report.as_dict()
