from unittest.mock import Mock, patch

from crewai_tools import SerplyJobSearchTool
from crewai_tools.tools.rag.rag_tool import Adapter
import pytest


@pytest.fixture
def tool(monkeypatch: pytest.MonkeyPatch) -> SerplyJobSearchTool:
    monkeypatch.setenv("SERPLY_API_KEY", "synthetic-key")
    return SerplyJobSearchTool(adapter=Mock(spec=Adapter))


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("highlights", ["Full-time role", ["Full-time", "Health insurance"]])
@pytest.mark.parametrize("remote,hybrid", [(False, True), (True, False), (False, False)])
def test_formats_jobs_without_losing_work_arrangement(
    tool: SerplyJobSearchTool,
    nested: bool,
    highlights: str | list[str],
    remote: bool,
    hybrid: bool,
) -> None:
    details = {"employer": "Synthetic Company", "is_remote": remote, "is_hybrid": hybrid}
    job = {"position": "Engineer", "link": "https://example.com/job", "highlights": highlights}
    if nested:
        job["description"] = details
    else:
        job.update(details)
        job["location"] = "Test city"

    with patch(
        "crewai_tools.tools.serply_api_tool.serply_job_search_tool.requests.request",
        return_value=Mock(json=Mock(return_value={"jobs": [job]})),
    ):
        result = tool.run(search_query="engineer")

    assert "Position: Engineer" in result
    assert "Employer: Synthetic Company" in result
    assert "Link: https://example.com/job" in result
    assert f"Is Remote: {remote}" in result
    assert f"Is Hybrid: {hybrid}" in result
    assert (highlights if isinstance(highlights, str) else ", ".join(highlights)) in result
    if not nested:
        assert "Location: Test city" in result


def test_no_jobs_returns_empty_result(tool: SerplyJobSearchTool) -> None:
    with patch(
        "crewai_tools.tools.serply_api_tool.serply_job_search_tool.requests.request",
        return_value=Mock(json=Mock(return_value={"jobs": []})),
    ):
        assert tool.run(search_query="engineer") == ""
