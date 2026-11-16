"""Tests for the SocialScrape tools.

No request leaves the machine: the SocialScrape client is given an
``httpx.MockTransport`` that answers with small sample responses.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

socialscrape = pytest.importorskip("socialscrape")

from crewai_tools import (  # noqa: E402
    SocialScrapeCommentsTool,
    SocialScrapeEndpointTool,
    SocialScrapePostsTool,
    SocialScrapePostTool,
    SocialScrapeProfileTool,
    SocialScrapeSearchTool,
    SocialScrapeTranscriptTool,
)
from crewai_tools.tools.socialscrape_tool import socialscrape_base_tool as base  # noqa: E402
from crewai_tools.tools.socialscrape_tool.socialscrape_base_tool import (  # noqa: E402
    endpoint_path,
)


KEY = "ss_test_key_do_not_leak"


def _envelope(data: Any, **extra: Any) -> dict[str, Any]:
    return {
        "ok": True,
        "credits_charged": 1,
        "credits_remaining": 4999,
        "cached": False,
        "request_id": "req_test",
        "data": data,
        **extra,
    }


class FakeAPI:
    """Records requests and answers from a queue per path."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.answers: dict[str, tuple[int, dict[str, Any]]] = {}

    def add(self, path: str, body: dict[str, Any], status: int = 200) -> None:
        self.answers[path] = (status, body)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        status, body = self.answers.get(
            request.url.path,
            (
                404,
                {
                    "ok": False,
                    "request_id": "req_test",
                    "error": {
                        "code": "not_found",
                        "message": "No such endpoint.",
                        "retryable": False,
                    },
                },
            ),
        )
        return httpx.Response(status, json=body)

    def params(self) -> dict[str, str]:
        return dict(self.requests[-1].url.params)


@pytest.fixture
def api() -> FakeAPI:
    return FakeAPI()


@pytest.fixture
def client(api: FakeAPI):
    return socialscrape.SocialScrape(
        api_key=KEY,
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(api)),
    )


PROFILE = {
    "object": "profile",
    "platform": "tiktok",
    "handle": "nasa",
    "name": "NASA",
    "stats": {"followers": 100},
}


def test_profile_calls_the_platform_endpoint_with_the_key(api, client):
    api.add("/v1/tiktok/profile", _envelope(PROFILE))
    out = json.loads(
        SocialScrapeProfileTool(client=client).run(platform="tiktok", handle="nasa")
    )

    assert out["data"]["name"] == "NASA"
    assert (out["credits_charged"], out["credits_remaining"]) == (1, 4999)
    request = api.requests[0]
    assert request.method == "GET"
    assert request.url.host == "api.socialscrape.dev"
    assert request.headers["x-api-key"] == KEY
    assert api.params() == {"handle": "nasa", "trim": "true"}


def test_tool_made_from_the_environment_uses_the_fixed_base_url(
    api, monkeypatch
):
    monkeypatch.setenv("SOCIALSCRAPE_API_KEY", "ss_env_key")
    monkeypatch.setenv("SOCIALSCRAPE_BASE_URL", "https://evil.example.test")
    made: dict[str, Any] = {}
    real_init = socialscrape.SocialScrape.__init__

    def init(self, **kwargs):
        made.update(kwargs)
        real_init(
            self,
            **kwargs,
            http_client=httpx.Client(transport=httpx.MockTransport(api)),
        )

    monkeypatch.setattr(socialscrape.SocialScrape, "__init__", init)
    api.add("/v1/tiktok/profile", _envelope(PROFILE))
    tool = SocialScrapeProfileTool()
    tool.run(platform="tiktok", handle="nasa")

    assert made["api_key"] == "ss_env_key"
    assert made["base_url"] == "https://api.socialscrape.dev"
    assert api.requests[0].headers["x-api-key"] == "ss_env_key"


def test_key_never_appears_in_repr_or_dump():
    tool = SocialScrapeProfileTool(api_key=KEY)
    assert KEY not in repr(tool)
    assert KEY not in tool.model_dump_json()


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("SOCIALSCRAPE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SOCIALSCRAPE_API_KEY"):
        SocialScrapeProfileTool().run(platform="tiktok", handle="nasa")


def test_missing_sdk_gives_an_install_hint(monkeypatch):
    monkeypatch.setattr(base, "SOCIALSCRAPE_AVAILABLE", False)
    with pytest.raises(ImportError, match="socialscrape-sdk"):
        SocialScrapeProfileTool(api_key=KEY).run(platform="tiktok", handle="nasa")


def test_unknown_platform_and_missing_identifier_cost_no_request(api, client):
    tool = SocialScrapeProfileTool(client=client)
    assert tool.run(platform="myspace", handle="tom").startswith("Error: No profile lookup")
    assert tool.run(platform="tiktok").startswith("Error: Pass one of handle")
    assert api.requests == []


def test_x_is_an_alias_for_twitter(api, client):
    api.add("/v1/twitter/profile", _envelope(PROFILE))
    SocialScrapeProfileTool(client=client).run(platform="X", handle="nasa")
    assert api.requests[0].url.path == "/v1/twitter/profile"


def test_posts_pass_the_cursor_on(api, client):
    api.add(
        "/v1/youtube/channel/videos",
        _envelope({"items": [{"id": "a"}], "cursor": "c2", "has_more": True}),
    )
    tool = SocialScrapePostsTool(client=client)
    tool.run(platform="youtube", handle="@NASA", cursor="c1")
    assert api.params() == {"handle": "@NASA", "cursor": "c1", "trim": "true"}


def test_post_reads_the_platform_from_the_link(api, client):
    api.add("/v1/youtube/video", _envelope({"object": "post"}))
    tool = SocialScrapePostTool(client=client)
    tool.run(url="https://youtu.be/dQw4w9WgXcQ")
    assert api.requests[0].url.path == "/v1/youtube/video"
    assert api.params()["url"] == "https://youtu.be/dQw4w9WgXcQ"

    out = tool.run(url="https://example.com/p/1")
    assert out.startswith("Error: No platform recognised")
    assert len(api.requests) == 1


def test_a_look_alike_host_is_not_taken_for_a_platform(api, client):
    out = SocialScrapePostTool(client=client).run(url="https://notyoutube.com/watch?v=1")
    assert out.startswith("Error: No platform recognised")
    assert api.requests == []


def test_post_by_id_needs_a_platform(api, client):
    api.add("/v1/tiktok/video", _envelope({"object": "post"}))
    tool = SocialScrapePostTool(client=client)
    assert tool.run(id="123").startswith("Error: Pass a url")
    tool.run(platform="tiktok", id="123")
    assert api.params() == {"id": "123", "trim": "true"}


def test_comments(api, client):
    api.add("/v1/youtube/video/comments", _envelope({"items": [], "has_more": False}))
    SocialScrapeCommentsTool(client=client).run(
        url="https://www.youtube.com/watch?v=dQw4w9WgXcQ", cursor="c1"
    )
    assert api.requests[0].url.path == "/v1/youtube/video/comments"
    assert api.params()["cursor"] == "c1"
    # Comments exist for fewer platforms than posts do.
    out = SocialScrapeCommentsTool(client=client).run(url="https://bsky.app/profile/a/post/b")
    assert out.startswith("Error: No platform recognised")


def test_transcript_leaves_segments_out_unless_asked(api, client):
    data = {
        "object": "transcript",
        "source": "captions",
        "text": "Hello there.",
        "segments": [{"start": 0.0, "end": 1.0, "text": "Hello there."}],
    }
    api.add("/v1/transcript", _envelope(data))
    tool = SocialScrapeTranscriptTool(client=client)
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    out = json.loads(tool.run(url=url))
    assert out["data"]["text"] == "Hello there."
    assert "segments" not in out["data"]
    assert api.params() == {"url": url, "trim": "true"}

    out = json.loads(
        tool.run(
            url=url,
            include_segments=True,
            allow_long_transcription=True,
            language="en",
        )
    )
    assert len(out["data"]["segments"]) == 1
    assert api.params() == {
        "url": url,
        "language": "en",
        "allow_long_transcription": "true",
        "trim": "true",
    }


def test_search(api, client):
    api.add("/v1/reddit/search", _envelope({"items": [], "has_more": False}))
    SocialScrapeSearchTool(client=client).run(platform="reddit", query="keyboards")
    assert api.params() == {"query": "keyboards", "trim": "true"}
    assert SocialScrapeSearchTool(client=client).run(
        platform="github", query="x"
    ).startswith("Error: No search lookup")
    assert len(api.requests) == 1


def test_an_api_error_is_the_answer(api, client):
    api.add(
        "/v1/tiktok/profile",
        {
            "ok": False,
            "request_id": "req_test",
            "error": {
                "code": "not_found",
                "message": "No TikTok account with that handle.",
                "retryable": False,
            },
        },
        404,
    )
    out = SocialScrapeProfileTool(client=client).run(platform="tiktok", handle="nobody")
    assert out.startswith("Error: not_found: No TikTok account with that handle.")
    assert KEY not in out


def test_cached_results_and_settings(api, client):
    api.add("/v1/tiktok/profile", _envelope(PROFILE, cached=True, credits_charged=0))
    tool = SocialScrapeProfileTool(client=client, max_age_days=7, trim=False)
    out = json.loads(tool.run(platform="tiktok", handle="nasa"))
    assert out["cached"] is True and out["credits_charged"] == 0
    assert api.params() == {"handle": "nasa", "max_age_days": "7.0"}


def test_long_answers_are_cut_and_say_so(api, client):
    api.add("/v1/transcript", _envelope({"text": "x" * 5_000}))
    tool = SocialScrapeTranscriptTool(client=client, max_output_chars=1_000)
    out = tool.run(url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert len(out) < 1_300
    assert "Answer cut at 1000 characters" in out


def test_endpoint_tool_runs_any_endpoint_by_id(api, client):
    api.add("/v1/facebook/ad-library/search", _envelope({"items": []}))
    tool = SocialScrapeEndpointTool(client=client)
    tool.run(endpoint="facebook.ad_library.search", params={"query": "running shoes"})
    assert api.requests[0].url.path == "/v1/facebook/ad-library/search"
    assert api.params() == {"query": "running shoes", "trim": "true"}


@pytest.mark.parametrize(
    "endpoint",
    [
        "../../etc/passwd",
        "https://evil.example.test/v1/x",
        "tiktok/profile",
        "tiktok.profile?x=1",
        "a.b.c.d.e.f",
        "",
    ],
)
def test_endpoint_tool_rejects_anything_but_an_endpoint_id(api, client, endpoint):
    out = SocialScrapeEndpointTool(client=client).run(endpoint=endpoint)
    assert out.startswith("Error: Invalid endpoint id")
    assert api.requests == []


def test_endpoint_path_maps_ids_to_api_paths():
    assert endpoint_path("transcript") == "/v1/transcript"
    assert endpoint_path("Apple_Music.artist") == "/v1/apple-music/artist"
    assert (
        endpoint_path("google.ad_library.company.ads")
        == "/v1/google/ad-library/company/ads"
    )


def test_tools_are_valid_crewai_tools_with_readable_schemas():
    tools = [
        SocialScrapeProfileTool(api_key=KEY),
        SocialScrapePostsTool(api_key=KEY),
        SocialScrapePostTool(api_key=KEY),
        SocialScrapeCommentsTool(api_key=KEY),
        SocialScrapeTranscriptTool(api_key=KEY),
        SocialScrapeSearchTool(api_key=KEY),
        SocialScrapeEndpointTool(api_key=KEY),
    ]
    assert len({t.name for t in tools}) == 7
    for tool in tools:
        assert tool.description
        assert tool.env_vars[0].name == "SOCIALSCRAPE_API_KEY"
        assert "socialscrape-sdk" in tool.package_dependencies
        assert KEY not in json.dumps(tool.args_schema.model_json_schema())
