"""Tests for the Serply job-search result formatting."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from crewai_tools.tools.serply_api_tool.serply_job_search_tool import (
    SerplyJobSearchTool,
)


def test_hybrid_flag_uses_its_own_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """A remote, non-hybrid job must not be reported as hybrid."""
    response = MagicMock()
    response.json.return_value = {
        "jobs": [
            {
                "position": "Backend Engineer",
                "employer": "Acme",
                "location": "New York, NY",
                "link": "https://example.test/jobs/1",
                "highlights": ["Python"],
                "is_remote": True,
                "is_hybrid": False,
            }
        ]
    }
    monkeypatch.setattr(
        "crewai_tools.tools.serply_api_tool.serply_job_search_tool.requests.request",
        MagicMock(return_value=response),
    )

    tool = SerplyJobSearchTool.model_construct(
        headers={}, request_url="https://api.example.test/job/search/"
    )
    output = tool._run(search_query="backend engineer")

    assert "Is Remote: True" in output
    assert "Is Hybrid: False" in output
