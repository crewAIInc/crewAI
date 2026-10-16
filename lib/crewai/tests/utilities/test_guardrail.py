from typing import Any

from pydantic import ValidationError
import pytest

from crewai.utilities.guardrail import GuardrailResult


@pytest.mark.parametrize(
    "payload",
    [
        {"success": True, "result": "accepted", "error": "rejected"},
        {"success": True, "result": "accepted", "error": ""},
        {"success": False, "result": "accepted", "error": "rejected"},
        {"success": False, "result": 0, "error": "rejected"},
    ],
)
def test_guardrail_result_rejects_payload_for_opposite_outcome(
    payload: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        GuardrailResult(**payload)


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ((True, "accepted"), GuardrailResult(success=True, result="accepted")),
        ((True, ""), GuardrailResult(success=True, result="")),
        ((True, None), GuardrailResult(success=True, result=None)),
        ((False, "rejected"), GuardrailResult(success=False, error="rejected")),
        ((False, ""), GuardrailResult(success=False, error="")),
        ((False, None), GuardrailResult(success=False, error=None)),
    ],
)
def test_guardrail_result_from_tuple_preserves_selected_payload(
    result: tuple[bool, Any | str | None], expected: GuardrailResult
) -> None:
    assert GuardrailResult.from_tuple(result) == expected
