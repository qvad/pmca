"""Task state machine and status definitions."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime


class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    DESIGNING = "designing"
    DECOMPOSED = "decomposed"
    CODING = "coding"
    REVIEWING = "reviewing"
    INTEGRATING = "integrating"
    VERIFIED = "verified"
    FAILED = "failed"


class TaskType(str, enum.Enum):
    ARCHITECTURE = "architecture"
    MODULE = "module"
    FUNCTION = "function"
    METHOD = "method"


# Valid state transitions
TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.PENDING: {TaskStatus.DESIGNING},
    TaskStatus.DESIGNING: {TaskStatus.DECOMPOSED, TaskStatus.CODING, TaskStatus.FAILED},
    TaskStatus.DECOMPOSED: {TaskStatus.INTEGRATING, TaskStatus.FAILED},
    TaskStatus.CODING: {TaskStatus.REVIEWING, TaskStatus.FAILED},
    TaskStatus.REVIEWING: {TaskStatus.VERIFIED, TaskStatus.CODING, TaskStatus.FAILED},
    TaskStatus.INTEGRATING: {TaskStatus.VERIFIED, TaskStatus.DECOMPOSED, TaskStatus.FAILED},
    TaskStatus.VERIFIED: set(),
    TaskStatus.FAILED: {TaskStatus.PENDING},
}


@dataclass
class ReviewResult:
    passed: bool
    issues: list[str]
    suggestions: list[str]
    timestamp: datetime
    model_used: str

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "timestamp": self.timestamp.isoformat(),
            "model_used": self.model_used,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ReviewResult:
        return cls(
            passed=data["passed"],
            issues=data.get("issues", []),
            suggestions=data.get("suggestions", []),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            model_used=data.get("model_used", ""),
        )


@dataclass
class CodeFile:
    path: str
    content: str

    def to_dict(self) -> dict:
        return {"path": self.path, "content": self.content}

    @classmethod
    def from_dict(cls, data: dict) -> CodeFile:
        return cls(path=data["path"], content=data["content"])


@dataclass
class TestResult:
    passed: bool
    total: int
    failures: int
    output: str
    errors: list[str]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "total": self.total,
            "failures": self.failures,
            "output": self.output,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TestResult:
        return cls(
            passed=data["passed"],
            total=data.get("total", 0),
            failures=data.get("failures", 0),
            output=data.get("output", ""),
            errors=data.get("errors", []),
        )


@dataclass
class FailureAnalysis:
    root_cause: str          # "code_bug" | "test_bug" | "import_error" | "unknown"
    explanation: str
    suggested_fix_target: str  # "code" | "tests" | "both"
    specific_issues: list[str]

    def to_dict(self) -> dict:
        return {
            "root_cause": self.root_cause,
            "explanation": self.explanation,
            "suggested_fix_target": self.suggested_fix_target,
            "specific_issues": self.specific_issues,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FailureAnalysis:
        return cls(
            root_cause=data.get("root_cause", "unknown"),
            explanation=data.get("explanation", ""),
            suggested_fix_target=data.get("suggested_fix_target", "code"),
            specific_issues=data.get("specific_issues", []),
        )


def validate_transition(current: TaskStatus, target: TaskStatus) -> bool:
    """Check if a state transition is valid."""
    return target in TRANSITIONS.get(current, set())
