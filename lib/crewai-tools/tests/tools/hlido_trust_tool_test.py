from unittest.mock import MagicMock, patch

from crewai_tools.tools.hlido_trust_tool.hlido_trust_tool import (
    HlidoRecommendTool,
    HlidoTrustCheckTool,
    _tier_for_score,
)


def _mock_response(status_code=200, json_data=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status.return_value = None
    return resp


def test_tier_for_score():
    assert _tier_for_score(95) == "VITAL"
    assert _tier_for_score(70) == "STEADY"
    assert _tier_for_score(40) == "FADING"
    assert _tier_for_score(10) == "FLATLINE"


def test_trust_check_defaults():
    tool = HlidoTrustCheckTool()
    assert tool.name == "Hlido Trust Check"
    assert tool.base_url == "https://hlido.eu"


@patch("crewai_tools.tools.hlido_trust_tool.hlido_trust_tool.requests.get")
def test_trust_check_pass(mock_get):
    mock_get.return_value = _mock_response(
        json_data={
            "slug": "aider",
            "name": "Aider",
            "score": 88,
            "tier": "STEADY",
            "what_it_does_well": ["Surgical diff edits"],
            "red_flags": [],
        }
    )
    result = HlidoTrustCheckTool().run(slug="aider", min_score=70)
    assert '"gate": "PASS"' in result
    assert '"verdict": "APPROVE"' in result
    assert "Aider" in result


@patch("crewai_tools.tools.hlido_trust_tool.hlido_trust_tool.requests.get")
def test_trust_check_fail(mock_get):
    mock_get.return_value = _mock_response(
        json_data={"slug": "weak", "name": "Weak", "score": 45, "tier": "FADING"}
    )
    result = HlidoTrustCheckTool().run(slug="weak", min_score=70)
    assert '"gate": "FAIL"' in result
    assert '"verdict": "REJECT"' in result


@patch("crewai_tools.tools.hlido_trust_tool.hlido_trust_tool.requests.get")
def test_trust_check_unknown_slug(mock_get):
    mock_get.return_value = _mock_response(status_code=404)
    result = HlidoTrustCheckTool().run(slug="does-not-exist")
    assert "No Hlido review found" in result


def test_trust_check_empty_slug():
    result = HlidoTrustCheckTool().run(slug="  ")
    assert "slug is required" in result


@patch("crewai_tools.tools.hlido_trust_tool.hlido_trust_tool.requests.get")
def test_recommend_ranks_by_score(mock_get):
    mock_get.return_value = _mock_response(
        json_data={
            "items": [
                {
                    "slug": "a",
                    "name": "A",
                    "category": "AI coding agent",
                    "score": 82,
                    "tier": "STEADY",
                    "lane": "reviewed",
                },
                {
                    "slug": "b",
                    "name": "B",
                    "category": "AI coding agent",
                    "score": 91,
                    "tier": "VITAL",
                    "lane": "reviewed",
                },
                {
                    "slug": "c",
                    "name": "C",
                    "category": "Voice",
                    "score": 95,
                    "tier": "VITAL",
                    "lane": "reviewed",
                },
                {
                    "slug": "d",
                    "name": "D",
                    "category": "AI coding agent",
                    "score": 60,
                    "tier": "FADING",
                    "lane": "reviewed",
                },
            ]
        }
    )
    result = HlidoRecommendTool().run(need="coding agent", min_score=70, limit=5)
    # Highest-scoring matching reviewed agent first; below-threshold and
    # non-matching-category excluded.
    assert result.index('"slug": "b"') < result.index('"slug": "a"')
    assert '"slug": "d"' not in result
    assert '"slug": "c"' not in result


@patch("crewai_tools.tools.hlido_trust_tool.hlido_trust_tool.requests.get")
def test_recommend_no_match(mock_get):
    mock_get.return_value = _mock_response(json_data={"items": []})
    result = HlidoRecommendTool().run(need="nonexistent category")
    assert "No Hlido-reviewed agents" in result
