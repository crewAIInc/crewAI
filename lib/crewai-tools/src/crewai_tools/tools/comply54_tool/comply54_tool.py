"""
comply54 compliance tools for CrewAI agents.

Provides pre-execution African regulatory enforcement (CBN, NDPA 2023,
POPIA, KDPA, NFIU AML, and 18 other jurisdiction packs) with zero runtime
dependencies beyond comply54 itself — no OPA binary, no API calls, fully
offline in-process Rego evaluation via regopy.

Three integration patterns:

1. Explicit self-check (agent decides when to call):

    from crewai_tools import Comply54ComplianceTool
    from comply54.sectors import NigeriaFintechCompliance

    tool = Comply54ComplianceTool(compliance=NigeriaFintechCompliance())
    agent = Agent(role="Fintech Agent", tools=[tool], ...)

2. Automatic pre-execution guard (transparent to the agent):

    from crewai_tools import comply54_guard_tools
    from comply54.sectors import NigeriaFintechCompliance

    guarded = comply54_guard_tools(
        NigeriaFintechCompliance(),
        [transfer_funds_tool, approve_payment_tool],
        context={"kyc_tier": 3},
    )
    agent = Agent(role="Fintech Agent", tools=guarded, ...)

3. Output-level task guardrail (blocks PII leakage before delivery):

    from comply54.crewai import Comply54TaskGuardrail
    guardrail = Comply54TaskGuardrail(NigeriaFintechCompliance())
    task = Task(..., guardrail=guardrail, agent=agent)
"""

from __future__ import annotations

import json
from typing import Any, List, Optional, Type

from pydantic import BaseModel, Field
from crewai.tools import BaseTool


def _ensure_comply54_installed() -> None:
    try:
        from comply54.sectors._base import SectorCompliance  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Missing optional dependency 'comply54'. Install with:\n"
            "  pip install crewai-tools[comply54]\n"
            "or\n"
            "  pip install comply54"
        ) from exc


class _ComplianceInput(BaseModel):
    """Input schema for Comply54ComplianceTool."""

    action: str = Field(
        description=(
            "The action or tool name the agent intends to perform "
            "(e.g. 'transfer_funds', 'approve_loan', 'send_sms')."
        )
    )
    params: dict = Field(
        default_factory=dict,
        description="Parameters for the action (e.g. amount, recipient, account number).",
    )
    output: str = Field(
        default="",
        description="Proposed output text to check for PII leakage or policy violations.",
    )
    context: dict = Field(
        default_factory=dict,
        description=(
            "Session context used to evaluate dynamic rules "
            "(e.g. {'kyc_tier': 3, 'customer_verified': True})."
        ),
    )


class Comply54ComplianceTool(BaseTool):
    """
    Check an agent action against African financial regulations before executing.

    Returns a JSON decision — ``allow``, ``deny``, ``escalate``, or ``audit`` —
    with the triggered rule, exact regulatory citation, and a stable audit ID.

    Covers Nigeria (CBN, NDPA 2023, NIIRA 2025, NFIU AML, BVN/NIN),
    Kenya (KDPA), South Africa (POPIA), Ghana, Rwanda, Egypt, Tanzania,
    Uganda, Ethiopia, Mauritius — 21 policy packs across 12 jurisdictions.

    No API key required. Evaluation runs fully offline via in-process Rego
    (regopy); no OPA binary or network call needed.

    Args:
        compliance: A ``SectorCompliance`` instance from comply54
                    (e.g. ``NigeriaFintechCompliance()``).

    Usage::

        from crewai_tools import Comply54ComplianceTool
        from comply54.sectors import NigeriaFintechCompliance

        tool = Comply54ComplianceTool(compliance=NigeriaFintechCompliance())
        agent = Agent(role="Fintech Agent", tools=[tool], ...)
    """

    name: str = "comply54_compliance_check"
    description: str = (
        "Check whether an agent action complies with applicable African regulatory "
        "requirements before executing it. Returns a JSON decision: allow / deny / "
        "escalate / audit — with the triggered rule and exact regulatory citation. "
        "Call this tool before any regulated financial, health, or data-processing action."
    )
    args_schema: Type[BaseModel] = _ComplianceInput
    package_dependencies: List[str] = ["comply54"]

    compliance: Any = None
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, compliance: Any, **kwargs: Any) -> None:
        _ensure_comply54_installed()
        super().__init__(
            compliance=compliance,
            name=f"comply54_{type(compliance).__name__.lower()}",
            description=(
                f"comply54 compliance check — {compliance.name}. "
                f"Jurisdictions: {', '.join(compliance.jurisdictions)}. "
                "Returns JSON with overall decision (allow/deny/escalate/audit), "
                "triggered rule, and exact regulatory citations."
            ),
            **kwargs,
        )

    def _run(
        self,
        action: str,
        params: Optional[dict] = None,
        output: str = "",
        context: Optional[dict] = None,
    ) -> str:
        result = self.compliance.check(
            action=action,
            params=params or {},
            output=output,
            context=context or {},
        )
        return json.dumps(_result_to_dict(result))


class _FallbackInput(BaseModel):
    """Generic input for tools without a defined args_schema."""

    input: str = Field(default="", description="Tool input.")


class Comply54GuardedTool(BaseTool):
    """
    Wrap any CrewAI BaseTool with automatic pre-execution comply54 enforcement.

    The original tool's name, description, and args_schema are preserved —
    the agent's interface is unchanged. If comply54 returns ``deny`` (or
    ``escalate`` when ``block_on_escalate=True``), the inner tool is not called
    and a structured error JSON is returned instead.

    Args:
        inner_tool:        The CrewAI ``BaseTool`` to wrap.
        compliance:        A ``SectorCompliance`` instance.
        context:           Default session context applied to every evaluation.
        block_on_escalate: Treat ``escalate`` decisions as blocks. Default ``False``.

    Usage::

        from crewai_tools import Comply54GuardedTool
        from comply54.sectors import NigeriaFintechCompliance

        guarded = Comply54GuardedTool(
            TransferFundsTool(),
            NigeriaFintechCompliance(),
            context={"kyc_tier": 3},
        )
        agent = Agent(role="Fintech Agent", tools=[guarded], ...)
    """

    name: str = ""
    description: str = ""
    package_dependencies: List[str] = ["comply54"]

    inner_tool: Any = None
    compliance: Any = None
    guard_context: dict = Field(default_factory=dict)
    block_on_escalate: bool = False
    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        inner_tool: Any,
        compliance: Any,
        context: Optional[dict] = None,
        block_on_escalate: bool = False,
        **kwargs: Any,
    ) -> None:
        _ensure_comply54_installed()
        inner_schema = getattr(inner_tool, "args_schema", _FallbackInput)

        super().__init__(
            name=inner_tool.name,
            description=(
                f"[comply54 guarded] {inner_tool.description} "
                f"— enforces {compliance.name} "
                f"({', '.join(compliance.jurisdictions)})"
            ),
            inner_tool=inner_tool,
            compliance=compliance,
            guard_context=context or {},
            block_on_escalate=block_on_escalate,
            args_schema=inner_schema,
            **kwargs,
        )

    def _run(self, **kwargs: Any) -> str:
        result = self.compliance.check(
            action=self.inner_tool.name,
            params=kwargs,
            output="",
            context=self.guard_context,
        )
        is_blocked = result.overall == "deny" or (
            self.block_on_escalate and result.overall == "escalate"
        )

        if is_blocked:
            violation = result.primary_violation
            reason = (
                violation.messages[0]
                if violation and violation.messages
                else "Compliance check failed"
            )
            return json.dumps({
                "blocked": True,
                "decision": result.overall,
                "reason": reason,
                "regulation": violation.regulation if violation else self.compliance.name,
                "rule_triggered": violation.rule_triggered if violation else None,
                "citations": [
                    {
                        "document": c.document,
                        "section": c.section,
                        "authority": c.authority,
                        "year": c.year,
                    }
                    for c in (violation.citations if violation else [])
                ],
                "audit_id": result.audit_id,
                "jurisdictions": self.compliance.jurisdictions,
            })

        return self.inner_tool._run(**kwargs)


def comply54_guard_tools(
    compliance: Any,
    tools: list,
    context: Optional[dict] = None,
    block_on_escalate: bool = False,
) -> list:
    """
    Wrap a list of CrewAI tools with comply54 pre-execution enforcement.

    Each tool's original name, description, and args_schema are preserved.
    Compliance is checked before every execution; denied calls return a
    structured error JSON without calling the inner tool.

    Args:
        compliance:        A ``SectorCompliance`` instance.
        tools:             List of CrewAI ``BaseTool`` instances to guard.
        context:           Default session context applied to every evaluation
                           (e.g. ``{"kyc_tier": 3, "customer_verified": True}``).
        block_on_escalate: Treat ``escalate`` decisions as blocks. Default ``False``.

    Returns:
        List of ``Comply54GuardedTool`` instances, one per input tool.

    Usage::

        from crewai_tools import comply54_guard_tools
        from comply54.sectors import NigeriaFintechCompliance

        guarded = comply54_guard_tools(
            NigeriaFintechCompliance(),
            [transfer_funds_tool, approve_payment_tool],
            context={"kyc_tier": 3},
        )
        agent = Agent(role="Fintech Agent", tools=guarded, ...)
    """
    return [
        Comply54GuardedTool(
            inner_tool=tool,
            compliance=compliance,
            context=context,
            block_on_escalate=block_on_escalate,
        )
        for tool in tools
    ]


def _result_to_dict(result: Any) -> dict:
    return {
        "overall": result.overall,
        "blocked": result.blocked,
        "audit_id": result.audit_id,
        "violations": [
            {
                "pack": d.pack,
                "regulation": d.regulation,
                "jurisdiction": d.jurisdiction,
                "action": d.action,
                "messages": d.messages,
                "rule_triggered": d.rule_triggered,
                "citations": [
                    {
                        "document": c.document,
                        "section": c.section,
                        "authority": c.authority,
                        "year": c.year,
                    }
                    for c in d.citations
                ],
            }
            for d in result.violations
        ],
    }
