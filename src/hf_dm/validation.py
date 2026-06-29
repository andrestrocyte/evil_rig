"""Typed validation results shared by rig diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence

ValidationStatus = Literal["PASS", "WARN", "FAIL"]


@dataclass(frozen=True)
class ValidationResult:
    name: str
    status: ValidationStatus
    detail: str
    hint: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


def validation_status(results: Sequence[ValidationResult]) -> ValidationStatus:
    statuses = {result.status for result in results}
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"


def validation_report(results: Sequence[ValidationResult]) -> dict[str, object]:
    return {
        "status": validation_status(results),
        "passed": all(result.status != "FAIL" for result in results),
        "results": [result.to_dict() for result in results],
    }
