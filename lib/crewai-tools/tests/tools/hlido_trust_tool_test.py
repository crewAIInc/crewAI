from unittest.mock import MagicMock, patch

from crewai_tools.tools.hlido_trust_tool.hlido_trust_tool import HlidoTrustTool


def _fake_verdict(*, recommended: bool, summary: str):
    v = MagicMock()
    v.recommended.return_value = recommended
    v.summary.return_value = summary
    return v


@patch("crewai_tools.tools.hlido_trust_tool.hlido_trust_tool.HlidoClient")
def test_hlido_trust_tool_pass(mock_client_cls):
    instance = mock_client_cls.return_value
    instance.trust_check.return_value = _fake_verdict(
        recommended=True,
        summary="Aider — VITAL (92/100), confidence high. Evidence: https://hlido.eu/reviews/aider/",
    )

    tool = HlidoTrustTool()
    result = tool.run(slug="aider", min_score=70)

    assert "PASS" in result
    assert "Aider" in result
    instance.trust_check.assert_called_once_with("aider")


@patch("crewai_tools.tools.hlido_trust_tool.hlido_trust_tool.HlidoClient")
def test_hlido_trust_tool_fails_closed_on_red_flag(mock_client_cls):
    mock_client_cls.return_value.trust_check.return_value = _fake_verdict(
        recommended=False,
        summary="X — VITAL (95/100) — red flags: hallucinates deletions. Evidence: https://hlido.eu/reviews/x/",
    )

    result = HlidoTrustTool().run(slug="x", min_score=70)

    assert "FAIL" in result
