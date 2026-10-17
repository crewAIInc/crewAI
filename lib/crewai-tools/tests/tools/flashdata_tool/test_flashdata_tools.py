import asyncio
import copy
import json
from pathlib import Path

from crewai.tools.structured_tool import ToolUsageLimitExceededError
import httpx
import pytest

from crewai_tools.tools import flashdata_tool as integration
from crewai_tools.tools.flashdata_tool import _base
from crewai_tools.tools.flashdata_tool._client import Client, FlashDataError


CAPABILITIES = json.loads(Path(__file__).with_name("capabilities.json").read_text())
REALTIME = [item for item in CAPABILITIES if item["source"] != "youtube_download"]
CLASSES = [getattr(integration, name) for name in integration.__all__]


def tool_class(source):
    return next(cls for cls in CLASSES if getattr(cls, "source", None) == source)


@pytest.fixture
def mock_api(monkeypatch):
    requests = []
    handler = None

    def set_handler(callback):
        nonlocal handler
        handler = callback
        return requests

    def transport(request):
        requests.append(request)
        return handler(request)

    monkeypatch.setenv("FLASHDATA_API_KEY", "mock-private-api-key")
    monkeypatch.setattr(
        _base,
        "Client",
        lambda key: Client(key, transport=httpx.MockTransport(transport)),
    )
    return set_handler


@pytest.mark.parametrize("capability", REALTIME, ids=lambda item: item["source"])
@pytest.mark.parametrize("structured", [False, True])
def test_real_tool_paths_preserve_api_contract(capability, structured, mock_api):
    response = copy.deepcopy(capability["outputExample"])
    response["new_provider_field"] = {"retained": True}
    requests = mock_api(lambda request: httpx.Response(200, json=response))
    inputs = copy.deepcopy(capability["sample"])
    original = copy.deepcopy(inputs)
    tool = tool_class(capability["source"])()
    result = (
        tool.to_structured_tool().invoke(inputs) if structured else tool.run(**inputs)
    )
    assert result == response
    assert inputs == original
    assert len(requests) == 1
    payload = json.loads(requests[0].content)
    expected = tool.input_model.model_validate(inputs).model_dump(
        exclude_none=True, by_alias=True
    )
    primary = next(key for key in ("query", "video_id", "handle") if key in expected)
    assert payload == {
        "source": capability["source"],
        primary: expected.pop(primary),
        "params": expected,
    }
    assert requests[0].headers["X-API-Key"] == "mock-private-api-key"


@pytest.mark.parametrize("cls", CLASSES, ids=lambda cls: cls.__name__)
def test_checkpoint_round_trip_excludes_secrets(cls, monkeypatch):
    monkeypatch.setenv("FLASHDATA_API_KEY", "mock-private-api-key")
    monkeypatch.setenv("FLASHDATA_DOWNLOAD_STORAGE", "mock-private-storage")
    tool = cls()
    data = tool.model_dump(mode="json")
    encoded = json.dumps(data)
    assert "mock-private" not in encoded
    restored = cls.model_validate(data)
    assert type(restored) is cls
    assert restored.args_schema.model_json_schema() == tool.args_schema.model_json_schema()
    assert restored.api_key_env == "FLASHDATA_API_KEY"
    properties = tool.args_schema.model_json_schema()["properties"]
    assert not {
        "api_key",
        "api_key_env",
        "storage",
        "storage_env",
        "allow_submit",
        "source",
    } & set(properties)
    assert not restored.cache_function({}, {})


def test_nested_filter_aliases_survive_structured_tool_conversion(mock_api):
    requests = mock_api(lambda request: httpx.Response(200, json={"results": []}))
    integration.FlashDataYouTubeSearchTool().to_structured_tool().invoke(
        {"query": "ocean", "filters": {"360": True, "4k": True, "3d": False}}
    )
    assert json.loads(requests[0].content)["params"]["filters"] == {
        "360": True,
        "3d": False,
        "4k": True,
    }


def test_async_paths_use_real_tools_without_extra_requests(mock_api):
    requests = mock_api(lambda request: httpx.Response(200, json={"results": []}))
    result = asyncio.run(integration.FlashDataGoogleSearchTool().arun(query="ocean"))
    other = asyncio.run(
        integration.FlashDataGoogleNewsTool()
        .to_structured_tool()
        .ainvoke({"query": "ocean"})
    )
    assert result == other == {"results": []}
    assert len(requests) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query": " "},
        {"query": "a", "num": True},
        {"query": "a", "num": 101},
        {"query": "a", "source": "youtube_download"},
    ],
)
def test_invalid_inputs_never_send_requests(kwargs, mock_api):
    requests = mock_api(lambda request: pytest.fail("Unexpected network request"))
    with pytest.raises(FlashDataError):
        integration.FlashDataGoogleSearchTool().run(**kwargs)
    assert not requests


def test_google_omitted_options_and_caption_defaults(mock_api):
    requests = mock_api(lambda request: httpx.Response(200, json={"results": []}))
    integration.FlashDataGoogleSearchTool().run(query="test")
    assert json.loads(requests[0].content)["params"] == {}
    integration.FlashDataYouTubeCaptionsTool().run(video_id="dQw4w9WgXcQ")
    integration.FlashDataYouTubeTranscriptTool().run(video_id="dQw4w9WgXcQ")
    assert json.loads(requests[1].content)["params"]["auto"] is False
    assert json.loads(requests[2].content)["params"]["auto"] is True


@pytest.mark.parametrize("status", [401, 402, 403, 429, 500])
def test_http_errors_redacted_without_retry(status, mock_api):
    requests = mock_api(
        lambda request: httpx.Response(status, json={"error": "private-response"})
    )
    with pytest.raises(FlashDataError) as error:
        integration.FlashDataGoogleSearchTool().run(query="test")
    assert str(status) in str(error.value)
    assert "private-response" not in str(error.value)
    assert len(requests) == 1


def test_timeout_reports_uncertain_billing_without_retry(mock_api):
    def handler(request):
        raise httpx.ReadTimeout("private-response", request=request)

    requests = mock_api(handler)
    with pytest.raises(FlashDataError, match="already have been accepted") as error:
        integration.FlashDataGoogleSearchTool().run(query="test")
    assert "private-response" not in str(error.value)
    assert len(requests) == 1


def test_usage_limit_is_enforced_by_crewai(mock_api):
    requests = mock_api(lambda request: httpx.Response(200, json={"results": []}))
    tool = integration.FlashDataGoogleSearchTool(max_usage_count=1).to_structured_tool()
    tool.invoke({"query": "first"})
    with pytest.raises(ToolUsageLimitExceededError, match="maximum usage limit"):
        tool.invoke({"query": "second"})
    assert len(requests) == 1


def test_download_requires_explicit_operator_opt_in(monkeypatch):
    monkeypatch.delenv("FLASHDATA_API_KEY", raising=False)
    with pytest.raises(FlashDataError, match="allow_submit=True"):
        integration.FlashDataYouTubeDownloadSubmitTool().run(video_id="dQw4w9WgXcQ")


def test_download_submits_once_and_does_not_return_storage(mock_api, monkeypatch):
    storage = {
        "storage_type": "s3",
        "storage_url": "customer-owned-bucket",
        "access_key": "private-access",
        "secret_key": "private-secret",
    }
    monkeypatch.setenv("FLASHDATA_DOWNLOAD_STORAGE", json.dumps(storage))
    requests = mock_api(
        lambda request: httpx.Response(
            200,
            json={
                "id": "job_saved",
                "status": "pending",
                "source": "youtube_download",
                "params": {"storage": storage},
            },
        )
    )
    result = (
        integration.FlashDataYouTubeDownloadSubmitTool(allow_submit=True)
        .to_structured_tool()
        .invoke(
            {
                "video_id": "dQw4w9WgXcQ",
                "download_type": "audio",
                "output_format": "m4a",
            }
        )
    )
    assert result == {
        "id": "job_saved",
        "status": "pending",
        "source": "youtube_download",
    }
    assert len(requests) == 1
    assert json.loads(requests[0].content)["params"]["storage"] == storage


@pytest.mark.parametrize(
    "source,job_id", [("google_search", "job_saved"), ("youtube_download", "job_other")]
)
def test_download_reads_reject_other_jobs(source, job_id, mock_api):
    requests = mock_api(
        lambda request: httpx.Response(
            200, json={"job": {"source": source, "id": job_id, "status": "completed"}}
        )
    )
    with pytest.raises(FlashDataError):
        integration.FlashDataYouTubeDownloadStatusTool().run(job_id="job_saved")
    assert len(requests) == 1


@pytest.mark.parametrize("trusted", [True, False])
def test_download_result_only_reads_and_never_forwards_key(trusted, mock_api):
    def handler(request):
        if request.url.path == "/v2/jobs/job_saved":
            return httpx.Response(
                200,
                json={
                    "job": {
                        "id": "job_saved",
                        "source": "youtube_download",
                        "status": "completed",
                        "params": {"storage": "private-storage"},
                    }
                },
            )
        if request.url.path == "/v2/api-access":
            return httpx.Response(
                200, json={"result_origin": "https://results.example.test"}
            )
        if request.url.path == "/v2/jobs/job_saved/results":
            host = "results.example.test" if trusted else "untrusted.example.test"
            return httpx.Response(
                302, headers={"Location": "https://" + host + "/manifest"}
            )
        assert request.url.host == "results.example.test"
        assert "X-API-Key" not in request.headers
        return httpx.Response(200, json={"files": [{"name": "audio.m4a"}]})

    requests = mock_api(handler)
    tool = integration.FlashDataYouTubeDownloadResultTool()
    if trusted:
        result = tool.run(job_id="job_saved")
        assert result["output"] == {"files": [{"name": "audio.m4a"}]}
        assert "private-storage" not in json.dumps(result)
        assert len(requests) == 4
    else:
        with pytest.raises(FlashDataError, match="trusted FlashData storage"):
            tool.run(job_id="job_saved")
        assert len(requests) == 3
    assert all(request.method == "GET" for request in requests)
