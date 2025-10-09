from unittest.mock import Mock

import pytest
from crewai.llm import LLM
from crewai.tasks.hallucination_guardrail import HallucinationGuardrail
from crewai.tasks.task_output import TaskOutput


def test_hallucination_guardrail_initialization():
    """Test that the hallucination guardrail initializes correctly with all parameters."""
    mock_llm = Mock(spec=LLM)

    guardrail = HallucinationGuardrail(context="Test reference context", llm=mock_llm)

    assert guardrail.context == "Test reference context"
    assert guardrail.llm == mock_llm
    assert guardrail.threshold is None
    assert guardrail.tool_response == ""

    guardrail = HallucinationGuardrail(
        context="Test reference context",
        llm=mock_llm,
        threshold=8.5,
        tool_response="Sample tool response",
    )

    assert guardrail.context == "Test reference context"
    assert guardrail.llm == mock_llm
    assert guardrail.threshold == 8.5
    assert guardrail.tool_response == "Sample tool response"


def test_hallucination_guardrail_no_op_behavior():
    """Test that the guardrail always returns True in the open-source version."""
    mock_llm = Mock(spec=LLM)
    guardrail = HallucinationGuardrail(
        context="Test reference context",
        llm=mock_llm,
        threshold=9.0,
    )

    task_output = TaskOutput(
        raw="Sample task output",
        description="Test task",
        expected_output="Expected output",
        agent="Test Agent",
    )

    result, output = guardrail(task_output)

    assert result is True
    assert output == "Sample task output"


def test_hallucination_guardrail_description():
    """Test that the guardrail provides the correct description for event logging."""
    guardrail = HallucinationGuardrail(
        context="Test reference context", llm=Mock(spec=LLM)
    )

    assert guardrail.description == "HallucinationGuardrail (no-op)"


@pytest.mark.parametrize(
    "context,task_output_text,threshold,tool_response",
    [
        (
            "Earth orbits the Sun once every 365.25 days.",
            "It takes Earth approximately one year to go around the Sun.",
            None,
            "",
        ),
        (
            "Python was created by Guido van Rossum in 1991.",
            "Python is a programming language developed by Guido van Rossum.",
            7.5,
            "",
        ),
        (
            "The capital of France is Paris.",
            "Paris is the largest city and capital of France.",
            9.0,
            "Geographic API returned: France capital is Paris",
        ),
    ],
)
def test_hallucination_guardrail_always_passes(
    context, task_output_text, threshold, tool_response
):
    """Test that the guardrail always passes regardless of configuration in open-source version."""
    mock_llm = Mock(spec=LLM)

    guardrail = HallucinationGuardrail(
        context=context, llm=mock_llm, threshold=threshold, tool_response=tool_response
    )

    task_output = TaskOutput(
        raw=task_output_text,
        description="Test task",
        expected_output="Expected output",
        agent="Test Agent",
    )

    result, output = guardrail(task_output)

    assert result is True
    assert output == task_output_text
