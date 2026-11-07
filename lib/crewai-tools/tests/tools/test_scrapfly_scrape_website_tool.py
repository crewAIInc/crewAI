from unittest.mock import patch

from crewai_tools.tools.scrapfly_scrape_website_tool.scrapfly_scrape_website_tool import (
    ScrapflyScrapeWebsiteTool,
)
import pytest


@pytest.mark.parametrize(
    ("kwargs", "expected_key"),
    [
        ({}, "env-key"),
        ({"api_key": None}, "env-key"),
        ({"api_key": ""}, "env-key"),
        ({"api_key": "explicit-key"}, "explicit-key"),
    ],
)
def test_api_key_resolution(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, str | None], expected_key: str
) -> None:
    monkeypatch.setenv("SCRAPFLY_API_KEY", "env-key")
    with patch("scrapfly.ScrapflyClient") as client:
        tool = ScrapflyScrapeWebsiteTool(**kwargs)

    client.assert_called_once_with(key=expected_key)
    assert tool.scrapfly is client.return_value
