from unittest.mock import patch

from crewai_tools import SpiderTool
import pytest


pytest.importorskip("spider")


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("mode", ["scrape", "crawl"])
def test_default_and_run_time_urls_reach_spider_request(
    configured: bool, override: bool, structured: bool, mode: str
) -> None:
    default_url = "https://example.com/default"
    runtime_url = "https://example.com/override"
    tool = SpiderTool(
        api_key="synthetic-key",
        website_url=default_url if configured else None,
        custom_params={"depth": 2},
        log_failures=False,
    )
    arguments = {"mode": mode}
    if override:
        arguments["website_url"] = runtime_url
    response = [{"content": "Offline page"}]
    with patch.object(tool.spider, "api_post", return_value=response) as post:
        if not configured and not override:
            with pytest.raises(ValueError, match="website_url"):
                if structured:
                    tool.to_structured_tool().invoke(arguments)
                else:
                    tool.run(**arguments)
            post.assert_not_called()
            return
        if structured:
            result = tool.to_structured_tool().invoke(arguments)
        else:
            result = tool.run(**arguments)

    assert result == response
    assert post.call_count == 1
    assert post.call_args.args[0] == "crawl"
    payload = post.call_args.args[1]
    assert payload["url"] == (runtime_url if override else default_url)
    assert payload["limit"] == (5 if mode == "crawl" else 1)
    assert payload["depth"] == 2
    assert payload["return_format"] == "markdown"
