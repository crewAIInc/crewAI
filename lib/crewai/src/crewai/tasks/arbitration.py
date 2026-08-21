"""Deterministic arbitration for agent deliverables.

Provides a lightweight, non-LLM escrow/arbitration layer that evaluates task
outputs against hard constraints (Pydantic contracts, optional CEL rules, and
deadlines) before a task is treated as resolved.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import json
from typing import Any, TypeVar

from pydantic import BaseModel, Field, ValidationError

from crewai.tasks.task_output import TaskOutput


class ArbitrationStatus(str, Enum):
    """Outcome of a deterministic arbitration evaluation."""

    APPROVED = "APPROVED"
    DISPUTED = "DISPUTED"


class Violation(BaseModel):
    """A single hard-constraint violation found during arbitration."""

    field: str = Field(description="Dotted path to the offending field.")
    constraint: str = Field(description="Machine-readable constraint type.")
    message: str = Field(description="Human-readable explanation of the violation.")
    expected: str | None = Field(
        default=None, description="What the contract required."
    )
    received: Any = Field(
        default=None, description="The actual value that was submitted."
    )


class ArbitrationResult(BaseModel):
    """Structured result of evaluating a deliverable against a contract."""

    status: ArbitrationStatus
    contract_name: str
    evaluated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    violations: list[Violation] = Field(default_factory=list)
    validated_payload: dict[str, Any] | None = None

    @property
    def is_approved(self) -> bool:
        """Whether the deliverable satisfied every hard constraint."""
        return self.status is ArbitrationStatus.APPROVED

    def to_retry_instructions(self) -> str:
        """Format violations as agent-facing retry feedback."""
        if self.is_approved:
            return "No violations. Contract satisfied."
        lines = [
            f"DISPUTED: {len(self.violations)} violation(s) against "
            f"contract '{self.contract_name}':"
        ]
        for index, violation in enumerate(self.violations, start=1):
            detail = (
                f"  {index}. field='{violation.field}' "
                f"constraint='{violation.constraint}' -> {violation.message}"
            )
            if violation.expected:
                detail += f" (expected: {violation.expected})"
            lines.append(detail)
        return "\n".join(lines)


ContractT = TypeVar("ContractT", bound=BaseModel)


class ArbitrationEngine:
    """Deterministic evaluator for agent-to-agent deliverables.

    Validates a payload against a Pydantic contract, optional CEL boolean rules,
    and an optional deadline. Outcomes are APPROVED or DISPUTED with precise
    machine-readable violations — no LLM judging is involved.
    """

    def evaluate(
        self,
        contract: type[ContractT],
        payload: dict[str, Any] | str | TaskOutput | BaseModel,
        *,
        rules: list[str] | None = None,
        deadline: datetime | None = None,
        evaluated_at: datetime | None = None,
    ) -> ArbitrationResult:
        """Evaluate ``payload`` against ``contract`` and optional constraints.

        Args:
            contract: Pydantic model describing the required deliverable shape.
            payload: Dict, JSON string, TaskOutput, or BaseModel to evaluate.
            rules: Optional CEL expressions that must evaluate to ``True``.
                Expressions are evaluated with ``{"output": <validated dict>}``.
            deadline: Optional UTC/aware deadline; missed deadlines dispute.
            evaluated_at: Evaluation timestamp (defaults to now, UTC).

        Returns:
            ArbitrationResult with APPROVED or DISPUTED status.
        """
        contract_name = contract.__name__
        when = evaluated_at or datetime.now(timezone.utc)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)

        violations: list[Violation] = []

        if deadline is not None:
            deadline_aware = (
                deadline
                if deadline.tzinfo is not None
                else deadline.replace(tzinfo=timezone.utc)
            )
            if when > deadline_aware:
                violations.append(
                    Violation(
                        field="<deadline>",
                        constraint="deadline_missed",
                        message="Deliverable was submitted after the deadline.",
                        expected=deadline_aware.isoformat(),
                        received=when.isoformat(),
                    )
                )

        parsed = self._coerce_payload(payload)
        if isinstance(parsed, Violation):
            violations.append(parsed)
            return ArbitrationResult(
                status=ArbitrationStatus.DISPUTED,
                contract_name=contract_name,
                evaluated_at=when,
                violations=violations,
            )

        try:
            validated = contract.model_validate(parsed)
        except ValidationError as exc:
            violations.extend(self._translate_errors(exc))
            return ArbitrationResult(
                status=ArbitrationStatus.DISPUTED,
                contract_name=contract_name,
                evaluated_at=when,
                violations=violations,
            )

        validated_payload = validated.model_dump(mode="json")
        if rules:
            violations.extend(self._evaluate_rules(rules, validated_payload))

        if violations:
            return ArbitrationResult(
                status=ArbitrationStatus.DISPUTED,
                contract_name=contract_name,
                evaluated_at=when,
                violations=violations,
                validated_payload=validated_payload,
            )

        return ArbitrationResult(
            status=ArbitrationStatus.APPROVED,
            contract_name=contract_name,
            evaluated_at=when,
            validated_payload=validated_payload,
        )

    @staticmethod
    def _coerce_payload(
        payload: dict[str, Any] | str | TaskOutput | BaseModel,
    ) -> dict[str, Any] | Violation:
        if isinstance(payload, TaskOutput):
            if payload.json_dict is not None:
                return payload.json_dict
            if payload.pydantic is not None:
                return payload.pydantic.model_dump(mode="json")
            return ArbitrationEngine._safe_parse(payload.raw)

        if isinstance(payload, BaseModel):
            return payload.model_dump(mode="json")

        return ArbitrationEngine._safe_parse(payload)

    @staticmethod
    def _safe_parse(payload: dict[str, Any] | str) -> dict[str, Any] | Violation:
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, str):
            text = payload.strip()
            if not text:
                return Violation(
                    field="<root>",
                    constraint="empty_payload",
                    message="Payload is empty.",
                    expected="A valid JSON object.",
                    received="",
                )
            try:
                loaded = json.loads(text)
            except json.JSONDecodeError as exc:
                return Violation(
                    field="<root>",
                    constraint="invalid_json",
                    message=f"Payload is not valid JSON: {exc.msg}.",
                    expected="A valid JSON object.",
                    received=text[:200],
                )
            if not isinstance(loaded, dict):
                return Violation(
                    field="<root>",
                    constraint="invalid_root_type",
                    message="Payload JSON root must be an object.",
                    expected="JSON object",
                    received=type(loaded).__name__,
                )
            return loaded
        return Violation(
            field="<root>",
            constraint="invalid_payload_type",
            message="Payload must be a dict, JSON string, TaskOutput, or BaseModel.",
            expected="dict | str | TaskOutput | BaseModel",
            received=type(payload).__name__,
        )

    @staticmethod
    def _translate_errors(exc: ValidationError) -> list[Violation]:
        violations: list[Violation] = []
        for err in exc.errors():
            field_path = ".".join(str(loc) for loc in err["loc"]) or "<root>"
            received = err.get("input")
            if isinstance(received, (dict, list)):
                received = json.dumps(received)[:200]
            violations.append(
                Violation(
                    field=field_path,
                    constraint=str(err["type"]),
                    message=err["msg"],
                    expected=None,
                    received=received,
                )
            )
        return violations

    @staticmethod
    def _evaluate_rules(
        rules: list[str], validated_payload: dict[str, Any]
    ) -> list[Violation]:
        """Evaluate CEL boolean rules against the validated payload."""
        from typing import cast

        try:
            from celpy import Environment
            from celpy.adapter import CELJSONEncoder, json_to_cel
            from celpy.evaluation import Context
        except Exception as exc:
            return [
                Violation(
                    field="<rule>",
                    constraint="rule_evaluation_error",
                    message=f"Failed to import CEL runtime for rule evaluation: {exc}",
                    expected="cel-python to be installed.",
                    received=None,
                )
            ]

        violations: list[Violation] = []
        environment = Environment()
        context = cast(Context, json_to_cel({"output": validated_payload}))

        for rule in rules:
            expression = rule.strip()
            if not expression:
                continue
            try:
                program = environment.program(environment.compile(expression))
                result = program.evaluate(context)
                normalized = json.loads(json.dumps(result, cls=CELJSONEncoder))
            except Exception as exc:
                violations.append(
                    Violation(
                        field="<rule>",
                        constraint="rule_evaluation_error",
                        message=f"Failed to evaluate rule {expression!r}: {exc}",
                        expected="A valid CEL boolean expression.",
                        received=expression,
                    )
                )
                continue

            if normalized is not True:
                violations.append(
                    Violation(
                        field="<rule>",
                        constraint="rule_failed",
                        message=f"Rule not satisfied: {expression}",
                        expected="true",
                        received=normalized,
                    )
                )
        return violations


class ArbitrationGuardrail:
    """Task guardrail that deterministically arbitrates deliverables.

    Plug into ``Task(guardrail=...)`` or ``Task(guardrails=[...])``. On dispute,
    returns retry instructions describing each hard-constraint violation.

    Examples:
        >>> from pydantic import BaseModel, Field
        >>> class Deliverable(BaseModel):
        ...     price: float = Field(ge=0, le=180)
        ...     room_type: str
        >>> guardrail = ArbitrationGuardrail(
        ...     Deliverable,
        ...     rules=["output.price >= 0.0 && output.price <= 180.0"],
        ... )
        >>> task = Task(..., guardrail=guardrail)
    """

    def __init__(
        self,
        contract: type[BaseModel],
        *,
        rules: list[str] | None = None,
        deadline: datetime | None = None,
        engine: ArbitrationEngine | None = None,
    ) -> None:
        """Initialize the arbitration guardrail.

        Args:
            contract: Pydantic model the deliverable must satisfy.
            rules: Optional CEL boolean expressions over ``output.*``.
            deadline: Optional deadline after which outputs are disputed.
            engine: Optional custom ArbitrationEngine instance.
        """
        self.contract = contract
        self.rules = list(rules) if rules else None
        self.deadline = deadline
        self.engine = engine or ArbitrationEngine()

    @property
    def description(self) -> str:
        """Description used in guardrail event logging."""
        parts = [f"ArbitrationGuardrail({self.contract.__name__})"]
        if self.rules:
            parts.append(f"rules={len(self.rules)}")
        if self.deadline is not None:
            parts.append(f"deadline={self.deadline.isoformat()}")
        return " ".join(parts)

    def __call__(self, task_output: TaskOutput) -> tuple[bool, Any]:
        """Arbitrate a task output against the configured hard constraints.

        Args:
            task_output: The task output to evaluate.

        Returns:
            ``(True, raw_or_json)`` when approved, otherwise
            ``(False, retry_instructions)``.
        """
        result = self.engine.evaluate(
            self.contract,
            task_output,
            rules=self.rules,
            deadline=self.deadline,
        )
        if result.is_approved:
            if result.validated_payload is not None:
                return True, json.dumps(result.validated_payload)
            return True, task_output.raw
        return False, result.to_retry_instructions()
