"""Contract tests for the native MrScraper integration."""

import hashlib
import json
from pathlib import Path
from typing import Any

from crewai import Agent
from crewai.tools import BaseTool
from crewai_tools import (
    MrScraperCrawlWebsiteUrlsTool,
    MrScraperCreateListingScraperTool,
    MrScraperCreatePromptScraperTool,
    MrScraperCreateWebsiteCrawlScraperTool,
    MrScraperExtractListingsTool,
    MrScraperExtractPageByPromptTool,
    MrScraperExtractStructuredDataTool,
    MrScraperFetchRenderedHtmlTool,
    MrScraperGetAccountInfoTool,
    MrScraperGetLatestResultsTool,
    MrScraperGetResultDetailTool,
    MrScraperGetResultsTool,
    MrScraperRunExistingScraperBatchTool,
    MrScraperRunExistingScraperTool,
    MrScraperSearchGoogleSerpTool,
    create_mrscraper_toolkit,
)
from crewai_tools.generate_tool_specs import ToolSpecExtractor
from crewai_tools.tools.mrscraper.client import MrScraperClient
from crewai_tools.tools.mrscraper.extraction import load_structured_data_prompts
from crewai_tools.tools.mrscraper.payloads import (
    append_output_schema,
    general_payload,
    listing_payload,
    map_payload,
)
from crewai_tools.tools.mrscraper.schemas import (
    FetchRenderedHtmlInput,
    GetResultsInput,
    ListingScraperInput,
    MapScraperInput,
    RunExistingScraperBatchInput,
    RunExistingScraperInput,
    SearchGoogleSerpInput,
)
from pydantic import BaseModel, ValidationError
import pytest
import requests


FAKE_TOKEN = "conspicuously-fake-mrscraper-token"

TOOL_CLASSES = (
    MrScraperGetAccountInfoTool,
    MrScraperCrawlWebsiteUrlsTool,
    MrScraperSearchGoogleSerpTool,
    MrScraperExtractPageByPromptTool,
    MrScraperExtractListingsTool,
    MrScraperExtractStructuredDataTool,
    MrScraperFetchRenderedHtmlTool,
    MrScraperGetResultsTool,
    MrScraperGetLatestResultsTool,
    MrScraperGetResultDetailTool,
    MrScraperCreatePromptScraperTool,
    MrScraperCreateListingScraperTool,
    MrScraperCreateWebsiteCrawlScraperTool,
    MrScraperRunExistingScraperTool,
    MrScraperRunExistingScraperBatchTool,
)

TOOL_NAMES = (
    "mrscraper_get_account_info",
    "mrscraper_crawl_website_urls",
    "mrscraper_search_google_serp",
    "mrscraper_extract_page_by_prompt",
    "mrscraper_extract_listings",
    "mrscraper_extract_structured_data",
    "mrscraper_fetch_rendered_html",
    "mrscraper_get_results",
    "mrscraper_get_latest_results",
    "mrscraper_get_result_detail",
    "mrscraper_create_prompt_scraper",
    "mrscraper_create_listing_scraper",
    "mrscraper_create_website_crawl_scraper",
    "mrscraper_run_existing_scraper",
    "mrscraper_run_existing_scraper_batch",
)


class FakeResponse:
    def __init__(
        self,
        value: Any = None,
        *,
        text: str | None = None,
        status_code: int = 200,
        content_type: str = "application/json",
    ) -> None:
        """Initialize a deterministic response double."""
        self.value = value
        self.text = text if text is not None else json.dumps(value)
        self.status_code = status_code
        self.headers = {"Content-Type": content_type}

    def json(self) -> Any:
        """Return the configured JSON value."""
        return self.value


class FakeSession:
    def __init__(
        self,
        response: FakeResponse | None = None,
        error: requests.RequestException | None = None,
    ) -> None:
        """Initialize a session double with a response or transport error."""
        self.response = response or FakeResponse({"ok": True})
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        """Record a request before returning or raising the configured result."""
        self.calls.append({"method": method, "url": url, **kwargs})
        if self.error is not None:
            raise self.error
        return self.response


def make_client(
    response: FakeResponse | None = None,
    error: requests.RequestException | None = None,
) -> tuple[MrScraperClient, FakeSession]:
    """Build a client and expose its deterministic session double."""
    session = FakeSession(response=response, error=error)
    return MrScraperClient(FAKE_TOKEN, session=session), session  # type: ignore[arg-type]


def test_all_tools_are_public_independent_base_tools() -> None:
    """Expose every integration operation as a distinct BaseTool."""
    client, _ = make_client()
    tools = [tool_class(client=client) for tool_class in TOOL_CLASSES]

    assert len(tools) == 15
    assert all(isinstance(tool, BaseTool) for tool in tools)
    assert tuple(tool.name for tool in tools) == TOOL_NAMES
    assert len({tool.description for tool in tools}) == 15
    assert all(tool.description.strip() for tool in tools)
    assert all(tool.args_schema is not BaseTool._ArgsSchemaPlaceholder for tool in tools)


def test_toolkit_returns_all_groups_names_and_fresh_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Select fresh tool instances by group or public tool name."""
    monkeypatch.setenv("MRSCRAPER_API_TOKEN", FAKE_TOKEN)
    first = create_mrscraper_toolkit()
    second = create_mrscraper_toolkit()

    assert len(first) == 15
    assert [tool.name for tool in first] == list(TOOL_NAMES)
    assert all(left is not right for left, right in zip(first, second, strict=True))
    assert [tool.name for tool in create_mrscraper_toolkit(groups=["Results"])] == [
        "mrscraper_get_results",
        "mrscraper_get_latest_results",
        "mrscraper_get_result_detail",
    ]
    assert len(create_mrscraper_toolkit(groups=["Discovery", "Extraction"])) == 6
    selected = create_mrscraper_toolkit(
        tool_names=["mrscraper_get_account_info", "mrscraper_get_result_detail"]
    )
    assert [tool.name for tool in selected] == [
        "mrscraper_get_account_info",
        "mrscraper_get_result_detail",
    ]
    with pytest.raises(ValueError, match="groups or tool_names"):
        create_mrscraper_toolkit(groups=["Account"], tool_names=[TOOL_NAMES[0]])
    with pytest.raises(ValueError, match="Unknown MrScraper toolkit groups"):
        create_mrscraper_toolkit(groups=["unknown"])


def test_agent_receives_15_independent_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """Attach all independent MrScraper tools to an Agent."""
    monkeypatch.setenv("MRSCRAPER_API_TOKEN", FAKE_TOKEN)
    tools = create_mrscraper_toolkit()
    agent = Agent(
        role="MrScraper contract test",
        goal="Verify independent tool discovery",
        backstory="A deterministic test agent",
        tools=tools,
    )

    assert len(agent.tools) == 15
    assert [tool.name for tool in agent.tools] == list(TOOL_NAMES)
    assert "operation" not in {
        field for tool in agent.tools for field in tool.args_schema.model_fields
    }


def test_schema_required_defaults_enums_constraints_and_descriptions() -> None:
    """Publish strict, documented schemas with the expected defaults."""
    search_schema = SearchGoogleSerpInput.model_json_schema()
    assert search_schema["required"] == ["query"]
    assert search_schema["properties"]["page"]["default"] == 1
    assert search_schema["properties"]["page"]["minimum"] == 1
    assert search_schema["properties"]["format"]["enum"] == ["json", "html"]

    rendered = FetchRenderedHtmlInput.model_json_schema()["properties"]
    assert rendered["max_retries"]["minimum"] == 0
    assert {item.get("minimum") for item in rendered["token_cap"]["anyOf"]} == {
        1,
        None,
    }
    assert ["full", "top"] in [
        item.get("enum") for item in rendered["screenshot_mode"]["anyOf"]
    ]

    listing = ListingScraperInput.model_json_schema()
    assert listing["required"] == ["url"]
    assert listing["properties"]["max_pages"]["default"] == 1
    assert listing["properties"]["max_pages"]["minimum"] == 1

    for tool_class in TOOL_CLASSES:
        schema = tool_class.model_fields["args_schema"].default.model_json_schema()
        assert schema["type"] == "object"
        for field in schema.get("properties", {}).values():
            assert field.get("description")


@pytest.mark.parametrize(
    ("schema", "values"),
    [
        (SearchGoogleSerpInput, {"query": "x", "page": True}),
        (SearchGoogleSerpInput, {"query": "x", "region": "USA"}),
        (MapScraperInput, {"url": "https://example.com", "limit": False}),
        (ListingScraperInput, {"url": "https://example.com", "max_pages": 1.5}),
        (FetchRenderedHtmlInput, {"url": "https://example.com", "max_retries": -1}),
        (RunExistingScraperBatchInput, {"scraper_type": "ai", "scraper_id": "id", "urls": []}),
        (RunExistingScraperBatchInput, {"scraper_type": "ai", "scraper_id": "id", "urls": [" "]}),
    ],
)
def test_strict_schema_validation(schema: Any, values: dict[str, Any]) -> None:
    """Reject coercion, invalid bounds, and empty URL batches."""
    with pytest.raises(ValidationError):
        schema.model_validate(values)


def test_credentials_are_required_metadata_only_and_secret_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MRSCRAPER_API_TOKEN", raising=False)
    with pytest.raises(ValueError, match="MRSCRAPER_API_TOKEN is required") as exc:
        MrScraperGetAccountInfoTool()
    assert FAKE_TOKEN not in str(exc.value)

    client, _ = make_client()
    for tool_class in TOOL_CLASSES:
        tool = tool_class(client=client)
        assert [(item.name, item.description, item.required) for item in tool.env_vars] == [
            ("MRSCRAPER_API_TOKEN", "MrScraper API token", True)
        ]
        schema_text = json.dumps(tool.args_schema.model_json_schema())
        assert "MRSCRAPER_API_TOKEN" not in schema_text
        assert FAKE_TOKEN not in schema_text
        assert FAKE_TOKEN not in repr(tool)
        assert FAKE_TOKEN not in tool.model_dump_json()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"café": False, "zero": 0}, '{"café":false,"zero":0}'),
        ([1, False, 0], "[1,false,0]"),
        (False, "false"),
        (0, "0"),
        (None, "null"),
    ],
)
def test_client_preserves_json_shapes_and_primary_auth(value: Any, expected: str) -> None:
    client, session = make_client(FakeResponse(value))
    result = client.request("GET", "primary", "/api/v1/test")

    assert result == expected
    call = session.calls[0]
    assert call["url"] == "https://api.app.mrscraper.com/api/v1/test"
    assert call["headers"]["x-api-token"] == FAKE_TOKEN
    assert call["headers"]["Accept"] == "application/json"
    assert call["timeout"] == (10, 660)


def test_serp_host_bearer_payload_and_html_preservation() -> None:
    html = "<!doctype html>\n<p>exact & unquoted</p>"
    client, session = make_client(
        FakeResponse(text=html, content_type="text/html; charset=utf-8")
    )
    tool = MrScraperSearchGoogleSerpTool(client=client)

    result = tool.run(query="hotels", format="html", render_js=False)

    assert result == html
    call = session.calls[0]
    assert call["url"] == (
        "https://sync.scraper.mrscraper.com/api/google/serp/v2/sync"
    )
    assert call["headers"]["Authorization"] == f"Bearer {FAKE_TOKEN}"
    assert "x-api-token" not in call["headers"]
    assert call["json"]["renderJs"] is False


def test_rendered_page_query_body_split_and_boolean_text() -> None:
    client, session = make_client(FakeResponse(text="<html />", content_type="text/html"))
    tool = MrScraperFetchRenderedHtmlTool(client=client)

    assert tool.run(
        url="https://target.example",
        max_retries=0,
        screenshot=False,
        html=False,
        markdown=False,
        block_resources=False,
        home_page=False,
        return_cookie=False,
        super_mode=False,
    ) == "<html />"
    call = session.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == "https://api.mrscraper.com/"
    assert call["params"] == {
        "token": FAKE_TOKEN,
        "browserRendering": "true",
        "timeout": 300,
        "geoCode": "us",
        "html": "false",
        "markdown": "false",
        "proxyCountry": "us",
    }
    assert call["json"] == {
        "url": "https://target.example",
        "maxRetries": 0,
    }

    tool.run(url="https://target.example", screenshot=True, screenshot_mode="top")
    assert session.calls[1]["params"]["screenshot"] == "top"

    tool.run(
        url="https://target.example",
        screenshot=True,
        token_cap=30,
        wait_for_selector="#ready",
        wait_until="networkidle",
        block_resources=True,
        home_page=True,
        return_cookie=True,
        super_mode=True,
    )
    advanced = session.calls[2]
    assert advanced["params"]["screenshot"] == "full"
    assert advanced["params"]["waitForSelector"] == "#ready"
    assert advanced["params"]["waitUntil"] == "networkidle"
    assert advanced["params"]["blockResources"] == "true"
    assert advanced["params"]["returnCookie"] == "true"
    assert advanced["params"]["super"] == "true"
    assert advanced["json"]["tokenCap"] == 30
    assert advanced["json"]["homePage"] is True


def test_general_listing_map_payloads_and_schema_append_once() -> None:
    schema = {"name": "string", "enabled": False, "count": 0}
    general = general_payload(
        url="https://example.com",
        prompt="Extract",
        output_schema=schema,
        mode="Cheap",
        proxy_country=None,
    )
    listing = listing_payload(
        url="https://example.com",
        prompt=None,
        output_schema={},
        max_pages=1,
        proxy_country="ID",
    )
    mapping = map_payload(
        url="https://example.com",
        max_depth=0,
        max_pages=0,
        limit=1,
        include_patterns=None,
        exclude_patterns=None,
    )

    assert general["graph"] == "general"
    assert general["message"] == (
        "Extract\n\nReturn the output as JSON matching this schema:\n"
        '{"name":"string","enabled":false,"count":0}'
    )
    assert general["message"].count("Return the output as JSON matching this schema:") == 1
    assert "output_schema" not in general
    assert listing["graph"] == "listing"
    assert listing["message"] == "Return each item as JSON matching this schema:\n{}"
    assert mapping == {
        "graph": "map",
        "url": "https://example.com",
        "maxDepth": 0,
        "maxPages": 0,
        "limit": 1,
    }
    assert append_output_schema("p", None, "label") == "p"


def test_structured_presets_are_exact_and_selected_without_category() -> None:
    preset_path = (
        Path(__file__).resolve().parents[3]
        / "src/crewai_tools/tools/mrscraper/structured_data_prompts.json"
    )
    assert hashlib.sha256(preset_path.read_bytes()).hexdigest() == (
        "3d9c15e8ebe7ad8cb04281251311200c1d3413452f14f252dc9ed3a8aae8533a"
    )
    prompts = load_structured_data_prompts()
    assert set(prompts) == {
        "article",
        "forumThread",
        "hotel",
        "jobPosting",
        "post",
        "product",
        "property",
        "restaurant",
        "socialMediaProfile",
        "tourAttraction",
    }

    client, session = make_client()
    tool = MrScraperExtractStructuredDataTool(client=client)
    for category, expected_prompt in prompts.items():
        tool.run(url="https://example.com", category=category)
        body = session.calls[-1]["json"]
        assert body["message"] == expected_prompt
        assert "category" not in body


def test_result_filters_sort_and_encoded_detail_id() -> None:
    client, session = make_client()
    MrScraperGetResultsTool(client=client).run(
        scraper_id="scraper/id", page=0, page_size=0, sort_order="ASC"
    )
    assert session.calls[0]["params"] == {
        "filters[scraperId]": "scraper/id",
        "page": 0,
        "pageSize": 0,
        "sort": "createdAt",
        "sortOrder": "ASC",
    }

    MrScraperGetLatestResultsTool(client=client).run(scraper_id="id", count=0)
    assert session.calls[1]["params"]["page"] == 1
    assert session.calls[1]["params"]["pageSize"] == 0
    assert session.calls[1]["params"]["sortOrder"] == "DESC"

    MrScraperGetResultDetailTool(client=client).run(result_id="a/b ?#")
    assert session.calls[2]["url"].endswith("/api/v1/results/a%2Fb%20%3F%23")


@pytest.mark.parametrize(
    "values",
    [
        {"scraper_type": "manual", "scraper_id": "id", "url": "u", "agent_type": "general"},
        {"scraper_type": "manual", "scraper_id": "id", "url": "u", "max_depth": 2},
        {"scraper_type": "ai", "scraper_id": "id", "url": "u", "cookie_jar": "x"},
        {"scraper_type": "ai", "scraper_id": "id", "url": "u", "stream": False},
        {"scraper_type": "ai", "scraper_id": "id", "url": "u", "agent_type": "map", "screenshot": False},
        {"scraper_type": "ai", "scraper_id": "id", "url": "u", "agent_type": "general", "max_pages": 2},
    ],
)
def test_single_run_rejects_incompatible_conditional_fields(values: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="do not accept"):
        RunExistingScraperInput.model_validate(values)


def test_ai_single_run_endpoints_defaults_and_zero_false_preservation() -> None:
    client, session = make_client()
    tool = MrScraperRunExistingScraperTool(client=client)

    tool.run(
        scraper_type="ai",
        scraper_id="id",
        url="https://example.com",
        agent_type="general",
        bypass_proxy=False,
        html=False,
    )
    general = session.calls[0]
    assert general["url"].endswith("/api/v1/scrapers-ai-rerun")
    assert general["json"]["bypassProxy"] is False
    assert general["json"]["html"] is False
    assert "agent_type" not in general["json"]

    tool.run(
        scraper_type="ai",
        scraper_id="id",
        url="https://example.com",
        agent_type="listing",
        max_pages=1,
        timeout=1,
        stream=False,
    )
    listing = session.calls[1]["json"]
    assert listing["maxPages"] == 1
    assert listing["timeout"] == 1
    assert listing["stream"] is False

    tool.run(
        scraper_type="ai",
        scraper_id="id",
        url="https://example.com",
        agent_type="map",
        max_depth=0,
        max_pages=1,
        limit=1,
    )
    mapping = session.calls[2]["json"]
    assert mapping["maxDepth"] == 0
    assert mapping["maxPages"] == 1
    assert mapping["limit"] == 1
    assert "bypassProxy" not in mapping


def test_single_run_omits_unsupplied_advanced_options() -> None:
    client, session = make_client()
    tool = MrScraperRunExistingScraperTool(client=client)

    tool.run(scraper_type="ai", scraper_id="ai-id", url="https://example.com")
    assert session.calls[0]["json"] == {
        "scraperId": "ai-id",
        "url": "https://example.com",
        "maxRetry": 3,
    }

    tool.run(scraper_type="manual", scraper_id="manual-id", url="https://example.com")
    assert session.calls[1]["json"] == {
        "scraperId": "manual-id",
        "url": "https://example.com",
        "maxRetry": 3,
    }


def test_manual_single_run_endpoint_json_values_and_screenshot_string() -> None:
    client, session = make_client()
    tool = MrScraperRunExistingScraperTool(client=client)
    cookies = [{"name": "session", "value": "x"}]
    paginator = {"selector": "a.next", "maxPages": 0}

    tool.run(
        scraper_type="manual",
        scraper_id="id",
        url="https://example.com",
        max_retry=0,
        bypass_proxy=False,
        cookies=cookies,
        paginator=paginator,
        screenshot=False,
        token_cap=0,
    )

    call = session.calls[0]
    assert call["url"].endswith("/api/v1/scrapers-manual-rerun")
    assert call["json"]["maxRetry"] == 0
    assert call["json"]["bypassProxy"] is False
    assert call["json"]["cookies"] == cookies
    assert call["json"]["paginator"] == paginator
    assert call["json"]["screenshot"] == "false"
    assert call["json"]["tokenCap"] == 0
    assert "agentType" not in call["json"]


@pytest.mark.parametrize(
    ("scraper_type", "endpoint"),
    [
        ("ai", "/api/v1/scrapers-ai-rerun/bulk"),
        ("manual", "/api/v1/scrapers-manual-rerun/bulk"),
    ],
)
def test_batch_endpoint_and_array_payload(scraper_type: str, endpoint: str) -> None:
    client, session = make_client()
    tool = MrScraperRunExistingScraperBatchTool(client=client)
    urls = ["https://example.com/1", "https://example.com/2"]

    tool.run(scraper_type=scraper_type, scraper_id="id", urls=urls)

    assert session.calls[0]["url"].endswith(endpoint)
    assert session.calls[0]["json"] == {"scraperId": "id", "urls": urls}


def test_non_2xx_and_transport_errors_are_truncated_and_redacted() -> None:
    secret_body = f"token={FAKE_TOKEN} " + "x" * 2000
    client, _ = make_client(FakeResponse(text=secret_body, status_code=502))
    with pytest.raises(RuntimeError) as http_exc:
        client.request("POST", "rendered", "/")
    message = str(http_exc.value)
    assert "HTTP 502" in message
    assert FAKE_TOKEN not in message
    assert "[REDACTED]" in message
    assert len(message) < 1100

    rendered_url = f"https://api.mrscraper.com/?token={FAKE_TOKEN}&browserRendering=true"
    client, _ = make_client(error=requests.RequestException(rendered_url))
    with pytest.raises(RuntimeError) as transport_exc:
        client.request("POST", "rendered", "/")
    assert FAKE_TOKEN not in str(transport_exc.value)
    assert "token=[REDACTED]" in str(transport_exc.value)


def test_success_response_cannot_echo_the_secret() -> None:
    client, _ = make_client(FakeResponse({"echo": FAKE_TOKEN}))
    result = client.request("GET", "primary", "/api/v1/test")

    assert result == '{"echo":"[REDACTED]"}'
    assert FAKE_TOKEN not in result


def test_generated_discovery_specs_include_all_tools_without_secrets() -> None:
    specs = ToolSpecExtractor().extract_all_tools()
    by_class = {spec["name"]: spec for spec in specs}

    for tool_class in TOOL_CLASSES:
        spec = by_class[tool_class.__name__]
        assert spec["humanized_name"] in TOOL_NAMES
        assert spec["run_params_schema"]["type"] == "object"
        assert spec["env_vars"] == [
            {
                "name": "MRSCRAPER_API_TOKEN",
                "description": "MrScraper API token",
                "required": True,
                "default": None,
            }
        ]
        rendered = json.dumps(spec)
        assert FAKE_TOKEN not in rendered
        assert "api_token" not in spec["run_params_schema"].get("properties", {})


@pytest.mark.parametrize(
    ("tool_class", "input_schema"),
    [
        (MrScraperFetchRenderedHtmlTool, FetchRenderedHtmlInput),
        (MrScraperRunExistingScraperTool, RunExistingScraperInput),
    ],
)
def test_generated_run_schemas_match_runtime_schemas(
    tool_class: type[BaseTool], input_schema: type[BaseModel]
) -> None:
    """Keep generated discovery defaults aligned with direct tool invocation."""
    specs = ToolSpecExtractor().extract_all_tools()
    by_class = {spec["name"]: spec for spec in specs}

    assert by_class[tool_class.__name__]["run_params_schema"] == (
        input_schema.model_json_schema()
    )
