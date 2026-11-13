from typing import Any
from unittest.mock import patch

from crewai_tools import ScrapegraphScrapeTool
import pytest


sdk = pytest.importorskip("scrapegraph_py")


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("configured_prompt", [None, "Extract product prices"])
@pytest.mark.parametrize("runtime_prompt", [None, "Extract article headings"])
def test_prompt_configuration_reaches_sdk_request(
    structured: bool, configured_prompt: str | None, runtime_prompt: str | None
) -> None:
    requests: list[dict[str, Any]] = []

    def respond(method: str, url: str, request_kwargs: dict[str, Any]) -> dict[str, Any]:
        requests.append(request_kwargs["json"])
        return {"result": {"heading": "Offline page"}}

    key = "sgai-00000000-0000-0000-0000-000000000000"
    client = sdk.Client(api_key=key, mock=True, mock_handler=respond)
    with patch("scrapegraph_py.Client", return_value=client):
        tool = ScrapegraphScrapeTool(api_key=key, user_prompt=configured_prompt)
        arguments = {"website_url": "https://example.com/article"}
        if runtime_prompt is not None:
            arguments["user_prompt"] = runtime_prompt
        if structured:
            result = tool.to_structured_tool().invoke(arguments)
        else:
            result = tool.run(**arguments)

    assert result == {"result": {"heading": "Offline page"}}
    assert len(requests) == 1
    expected = runtime_prompt or configured_prompt or "Extract the main content of the webpage"
    assert requests[0]["user_prompt"] == expected
    assert requests[0]["website_url"] == "https://example.com/article"


def test_fixed_url_keeps_configured_prompt() -> None:
    requests: list[dict[str, Any]] = []

    def respond(method: str, url: str, request_kwargs: dict[str, Any]) -> dict[str, Any]:
        requests.append(request_kwargs["json"])
        return {"result": "Offline page"}

    key = "sgai-00000000-0000-0000-0000-000000000000"
    client = sdk.Client(api_key=key, mock=True, mock_handler=respond)
    with patch("scrapegraph_py.Client", return_value=client):
        tool = ScrapegraphScrapeTool(
            api_key=key, website_url="https://example.com/article",
            user_prompt="Extract product prices",
        )
        assert tool.run() == {"result": "Offline page"}
    assert requests[0]["user_prompt"] == "Extract product prices"
