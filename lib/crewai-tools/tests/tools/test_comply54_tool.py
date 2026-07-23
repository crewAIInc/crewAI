"""
Unit tests for Comply54ComplianceTool and Comply54GuardedTool.

All tests mock the comply54 library so no real Rego evaluation or
comply54 installation is required in CI.
"""

import json
from typing import Any, Optional, Type
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from crewai_tools.tools.comply54_tool.comply54_tool import (
    Comply54ComplianceTool,
    Comply54GuardedTool,
    comply54_guard_tools,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_compliance(overall: str = "allow", blocked: bool = False) -> MagicMock:
    """Build a minimal SectorCompliance mock."""
    violation = MagicMock()
    violation.messages = ["Transaction exceeds cap"]
    violation.regulation = "CBN NIP Framework"
    violation.rule_triggered = "cbn_nip_single_transfer_cap"
    violation.citations = []

    result = MagicMock()
    result.overall = overall
    result.blocked = blocked
    result.audit_id = "c54-test-001"
    result.violations = [violation] if blocked else []
    result.primary_violation = violation if blocked else None

    compliance = MagicMock()
    compliance.name = "Nigeria Fintech Compliance"
    compliance.jurisdictions = ["NG"]
    compliance.check.return_value = result
    return compliance


class _TransferInput(BaseModel):
    amount: float = Field(description="Amount in NGN.")
    recipient: str = Field(description="Recipient account number.")


class _MockInnerTool(BaseTool):
    name: str = "transfer_funds"
    description: str = "Transfer funds between accounts."
    args_schema: Type[BaseModel] = _TransferInput

    def _run(self, amount: float, recipient: str) -> str:
        return json.dumps({"status": "ok", "amount": amount, "recipient": recipient})


# ── Comply54ComplianceTool ────────────────────────────────────────────────────

@pytest.fixture
def mock_sector_compliance_import():
    """Patch the comply54 import check inside the tool constructors."""
    with patch(
        "crewai_tools.tools.comply54_tool.comply54_tool.__import__",
        side_effect=lambda name, *a, **kw: MagicMock() if "comply54" in name else __import__(name, *a, **kw),
    ):
        yield


def _build_compliance_tool(overall: str = "allow", blocked: bool = False):
    compliance = _make_compliance(overall=overall, blocked=blocked)
    with patch("crewai_tools.tools.comply54_tool.comply54_tool.__builtins__"):
        pass
    # Patch only the import guard inside __init__
    with patch(
        "crewai_tools.tools.comply54_tool.comply54_tool.importlib",
        create=True,
    ):
        pass

    with patch.dict("sys.modules", {"comply54": MagicMock(), "comply54.sectors._base": MagicMock()}):
        tool = Comply54ComplianceTool.__new__(Comply54ComplianceTool)
        tool.compliance = compliance
        tool.name = f"comply54_{type(compliance).__name__.lower()}"
        tool.description = "comply54 compliance check"
        tool.guard_context = {}
        tool.block_on_escalate = False
        return tool, compliance


class TestComply54ComplianceTool:
    def test_allow_decision_returns_json(self):
        compliance = _make_compliance(overall="allow", blocked=False)
        with patch.dict("sys.modules", {"comply54": MagicMock(), "comply54.sectors._base": MagicMock()}):
            tool = Comply54ComplianceTool(compliance=compliance)

        result = json.loads(tool._run(action="transfer_funds", params={"amount": 5000}))

        assert result["overall"] == "allow"
        assert result["blocked"] is False
        assert result["audit_id"] == "c54-test-001"
        compliance.check.assert_called_once_with(
            action="transfer_funds",
            params={"amount": 5000},
            output="",
            context={},
        )

    def test_deny_decision_returns_json(self):
        compliance = _make_compliance(overall="deny", blocked=True)
        with patch.dict("sys.modules", {"comply54": MagicMock(), "comply54.sectors._base": MagicMock()}):
            tool = Comply54ComplianceTool(compliance=compliance)

        result = json.loads(tool._run(action="transfer_funds", params={"amount": 15_000_000}))

        assert result["overall"] == "deny"
        assert result["blocked"] is True
        assert len(result["violations"]) == 1
        assert result["violations"][0]["regulation"] == "CBN NIP Framework"

    def test_missing_comply54_raises_import_error(self):
        with patch.dict("sys.modules", {"comply54": None, "comply54.sectors._base": None}):
            with pytest.raises(ImportError, match="comply54"):
                Comply54ComplianceTool(compliance=MagicMock())

    def test_name_derived_from_compliance_class(self):
        compliance = _make_compliance()
        compliance.__class__.__name__ = "NigeriaFintechCompliance"
        with patch.dict("sys.modules", {"comply54": MagicMock(), "comply54.sectors._base": MagicMock()}):
            tool = Comply54ComplianceTool(compliance=compliance)
        assert "nigeria" in tool.name.lower() or "compliance" in tool.name.lower()


# ── Comply54GuardedTool ───────────────────────────────────────────────────────

class TestComply54GuardedTool:
    def _make_guarded(self, overall: str = "allow", blocked: bool = False, block_on_escalate: bool = False):
        compliance = _make_compliance(overall=overall, blocked=blocked)
        inner = _MockInnerTool()
        with patch.dict("sys.modules", {"comply54": MagicMock(), "comply54.sectors._base": MagicMock()}):
            guarded = Comply54GuardedTool(
                inner,
                compliance,
                context={"kyc_tier": 3},
                block_on_escalate=block_on_escalate,
            )
        return guarded, compliance, inner

    def test_allow_passes_through_to_inner_tool(self):
        guarded, compliance, _ = self._make_guarded(overall="allow", blocked=False)

        result = json.loads(guarded._run(amount=5000.0, recipient="0123456789"))

        assert result["status"] == "ok"
        assert result["amount"] == 5000.0

    def test_deny_blocks_inner_tool(self):
        guarded, compliance, inner = self._make_guarded(overall="deny", blocked=True)

        result = json.loads(guarded._run(amount=15_000_000.0, recipient="0123456789"))

        assert result["blocked"] is True
        assert result["decision"] == "deny"
        assert "CBN NIP Framework" in result["regulation"]
        assert result["audit_id"] == "c54-test-001"

    def test_escalate_passes_through_by_default(self):
        guarded, _, _ = self._make_guarded(overall="escalate", blocked=False, block_on_escalate=False)

        result = json.loads(guarded._run(amount=5000.0, recipient="0123456789"))

        assert result["status"] == "ok"

    def test_escalate_blocked_when_block_on_escalate_true(self):
        compliance = _make_compliance(overall="escalate", blocked=False)
        # Primary violation still set for escalate case
        violation = MagicMock()
        violation.messages = ["Requires manual review"]
        violation.regulation = "NFIU AML"
        violation.rule_triggered = "nfiu_high_risk_flag"
        violation.citations = []
        compliance.check.return_value.primary_violation = violation
        compliance.check.return_value.overall = "escalate"

        inner = _MockInnerTool()
        with patch.dict("sys.modules", {"comply54": MagicMock(), "comply54.sectors._base": MagicMock()}):
            guarded = Comply54GuardedTool(inner, compliance, block_on_escalate=True)

        result = json.loads(guarded._run(amount=5000.0, recipient="0123456789"))

        assert result["blocked"] is True
        assert result["decision"] == "escalate"

    def test_preserves_inner_tool_name(self):
        guarded, _, inner = self._make_guarded()
        assert guarded.name == inner.name

    def test_preserves_inner_tool_args_schema(self):
        guarded, _, inner = self._make_guarded()
        assert guarded.args_schema is _TransferInput

    def test_missing_comply54_raises_import_error(self):
        with patch.dict("sys.modules", {"comply54": None, "comply54.sectors._base": None}):
            with pytest.raises(ImportError, match="comply54"):
                Comply54GuardedTool(_MockInnerTool(), MagicMock())


# ── comply54_guard_tools ──────────────────────────────────────────────────────

class TestComply54GuardTools:
    def test_returns_one_guarded_tool_per_input(self):
        compliance = _make_compliance()
        tools = [_MockInnerTool(), _MockInnerTool()]
        tools[1].name = "approve_loan"

        with patch.dict("sys.modules", {"comply54": MagicMock(), "comply54.sectors._base": MagicMock()}):
            guarded = comply54_guard_tools(compliance, tools, context={"kyc_tier": 3})

        assert len(guarded) == 2
        assert all(isinstance(t, Comply54GuardedTool) for t in guarded)

    def test_empty_list_returns_empty(self):
        compliance = _make_compliance()
        with patch.dict("sys.modules", {"comply54": MagicMock(), "comply54.sectors._base": MagicMock()}):
            guarded = comply54_guard_tools(compliance, [])
        assert guarded == []

    def test_context_propagated_to_all_tools(self):
        compliance = _make_compliance()
        ctx = {"kyc_tier": 3, "customer_verified": True}
        tools = [_MockInnerTool()]

        with patch.dict("sys.modules", {"comply54": MagicMock(), "comply54.sectors._base": MagicMock()}):
            guarded = comply54_guard_tools(compliance, tools, context=ctx)

        assert guarded[0].guard_context == ctx
