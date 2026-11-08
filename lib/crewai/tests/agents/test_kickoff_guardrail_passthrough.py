"""#8788: a pass-through guardrail keeps the response_format result."""
from pydantic import BaseModel

from crewai.lite_agent_output import LiteAgentOutput
from crewai.utilities.guardrail import process_guardrail


class Verdict(BaseModel):
    approved: bool
    reason: str


def _pass_through_guardrail(output):
    """Approves by returning the output it was given."""
    return True, output


def _replacement_guardrail(output):
    """Approves with a rewritten output."""
    return True, LiteAgentOutput(raw="rewritten", agent_role="Reviewer")


def test_pass_through_guardrail_keeps_the_response_format_result() -> None:
    output = LiteAgentOutput(raw="reviewed NDA", agent_role="Reviewer")
    output.pydantic = Verdict(approved=True, reason="ok")

    result = process_guardrail(
        output=output,
        guardrail=_pass_through_guardrail,
        retry_count=0,
    )

    if result.result is not None and isinstance(result.result, BaseModel):
        if result.result is output:
            # The guardrail passed the output back unchanged; response_format
            # result must survive, not be overwritten by the LiteAgentOutput.
            pass
        else:
            output.pydantic = result.result

    assert isinstance(output.pydantic, Verdict)
    assert output.pydantic.approved is True


def test_replacement_guardrail_replaces_the_result() -> None:
    output = LiteAgentOutput(raw="reviewed NDA", agent_role="Reviewer")
    output.pydantic = Verdict(approved=True, reason="ok")

    result = process_guardrail(
        output=output,
        guardrail=_replacement_guardrail,
        retry_count=0,
    )

    if result.result is not None and isinstance(result.result, BaseModel):
        if result.result is output:
            pass
        else:
            output.pydantic = result.result

    assert isinstance(output.pydantic, LiteAgentOutput)
    assert output.pydantic.raw == "rewritten"
