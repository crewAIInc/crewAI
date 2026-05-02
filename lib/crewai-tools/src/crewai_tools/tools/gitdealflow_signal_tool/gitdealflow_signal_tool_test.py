"""Smoke tests for GitDealFlowSignalTool.

These tests use mocking to avoid hitting the live API in CI. A separate set of
integration tests (not included here) exercises the live endpoints — the public
API is unauthenticated and stable, so live tests are easy to run locally.
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from crewai_tools.tools.gitdealflow_signal_tool.gitdealflow_signal_tool import (
    SECTOR_SLUGS,
    GitDealFlowSignalTool,
)


@pytest.fixture
def tool() -> GitDealFlowSignalTool:
    return GitDealFlowSignalTool()


def _mock_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.status = 200
    resp.read.return_value = json.dumps(payload).encode("utf-8")
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = None
    return resp


def test_unknown_action_returns_friendly_error(tool: GitDealFlowSignalTool) -> None:
    result = tool._run(action="bogus")
    assert "Unknown action" in result
    assert "trending" in result


def test_sector_requires_slug(tool: GitDealFlowSignalTool) -> None:
    result = tool._run(action="sector")
    assert "sector_slug" in result


def test_sector_validates_slug(tool: GitDealFlowSignalTool) -> None:
    result = tool._run(action="sector", sector_slug="not-a-real-sector")
    assert "Unknown sector" in result
    assert "ai-ml" in result


def test_startup_requires_name(tool: GitDealFlowSignalTool) -> None:
    result = tool._run(action="startup")
    assert "startup_name" in result


def test_trending_formats_top_results(tool: GitDealFlowSignalTool) -> None:
    payload = {
        "sectors": [
            {
                "slug": "ai-ml",
                "startups": [
                    {
                        "name": "ExampleAI",
                        "commitVelocityChange": 87.5,
                        "signalType": "breakout",
                        "contributors": 42,
                    },
                    {
                        "name": "AnotherAI",
                        "commitVelocityChange": 12.0,
                        "signalType": "steady",
                        "contributors": 8,
                    },
                ],
            }
        ]
    }
    with patch("urllib.request.urlopen", return_value=_mock_response(payload)):
        result = tool._run(action="trending", limit=5)
    assert "ExampleAI" in result
    assert "+87.5%" in result
    assert "breakout" in result
    assert "VC Deal Flow Signal" in result


def test_summary_counts_startups(tool: GitDealFlowSignalTool) -> None:
    payload = {
        "period": "2026-Q2",
        "asOf": "2026-05-01",
        "sectors": [
            {"slug": "ai-ml", "startups": [{"name": "A"}, {"name": "B"}]},
            {"slug": "fintech", "startups": [{"name": "C"}]},
        ],
    }
    with patch("urllib.request.urlopen", return_value=_mock_response(payload)):
        result = tool._run(action="summary")
    assert "Sectors covered: 2" in result
    assert "Tracked startups: 3" in result
    assert "2026-Q2" in result


def test_sector_slugs_constant_matches_documented_count() -> None:
    assert len(SECTOR_SLUGS) == 20
    assert "ai-ml" in SECTOR_SLUGS
    assert "fintech" in SECTOR_SLUGS
