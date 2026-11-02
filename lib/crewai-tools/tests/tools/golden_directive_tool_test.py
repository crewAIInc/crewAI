import pytest

from crewai_tools import GoldenDirectiveTool
from crewai_tools.tools.golden_directive_tool.golden_directive_tool import (
    relationship_score,
)


def test_relationship_score_canonical_numbers():
    """Canonical numeric checks from the Golden Directive spec."""
    assert relationship_score(8, 8) == 8.0
    assert relationship_score(8, 7) == pytest.approx(7.4667, abs=1e-4)
    assert relationship_score(2, 3) == pytest.approx(2.4)
    assert relationship_score(8, 0) == 0.0
    assert relationship_score(0, 8) == 0.0
    # A rising enough can still raise Y while E falls.
    assert relationship_score(1, 10) == pytest.approx(1.8182, abs=1e-4)
    assert relationship_score(10, 9) == pytest.approx(9.4737, abs=1e-4)
    # Negative inputs are floored at 0.
    assert relationship_score(-5, 8) == 0.0


def test_golden_directive_tool_run():
    tool = GoldenDirectiveTool()
    result = tool.run(agent_score=8, environment_score=8)
    assert "Y (full) = 8.0000" in result


def test_golden_directive_tool_reports_full_vs_cropped_gap():
    tool = GoldenDirectiveTool()
    result = tool.run(
        agent_score=8, environment_score=3, environment_score_cropped=7
    )
    assert "Y (full) = 4.3636" in result
    assert "Y (cropped) = 7.4667" in result
    assert "Gap (cropped Y - full Y) = 3.1030" in result
