"""Exercise native CrewAI dispatch with the real MCP SDK and an offline transport."""

import asyncio
import builtins
import importlib.util
import json
import logging
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

from crewai_tools import BaizhiExtractTool, BaizhiScrapeTool, BaizhiSearchTool
import httpx
import pytest


KEY = "synthetic-baizhi-test-key"
CASES = [
    (
        BaizhiSearchTool,
        {"query": "test", "filter": {"domains": ["example.com"]}},
        "websearch_search",
    ),
    (BaizhiScrapeTool, {"url": "https://example.com/page"}, "web_scrape"),
    (
        BaizhiExtractTool,
        {"url": "https://example.com/page", "fields": {"title": "string"}},
        "web_extract",
    ),
]


def dispatch_tool(tool, arguments, dispatch):
    if dispatch == "run":
        return tool.run(**arguments)
    if dispatch == "arun":
        return asyncio.run(tool.arun(**arguments))
    if dispatch == "invoke":
        return tool.to_structured_tool().invoke(arguments)
    return asyncio.run(tool.to_structured_tool().ainvoke(arguments))


@pytest.fixture
def server(monkeypatch):
    """Only the HTTP transport is mocked; CrewAI and MCP execute unchanged."""
    state = {
        "calls": [],
        "requests": [],
        "closed": False,
        "started": asyncio.Event(),
        "status": 200,
        "delay": False,
        "result": {
            "content": [{"type": "text", "text": "fallback"}],
            "structuredContent": {"items": ["result"]},
            "isError": False,
        },
    }
    original = httpx.AsyncClient

    async def handler(request):
        state["requests"].append(str(request.url))
        assert str(request.url) == "https://agent-toolkit.app.baizhi.cloud/mcp"
        assert request.headers["authorization"] == f"Bearer {KEY}"
        if request.method == "DELETE":
            state["closed"] = True
            return httpx.Response(204)
        if request.method == "GET":
            return httpx.Response(405)
        message = json.loads(request.content)
        method = message["method"]
        if method == "initialize":
            if state.get("malformed"):
                return httpx.Response(200, json={"jsonrpc": "2.0", "id": [KEY], "result": {"echo": KEY}})
            if state["status"] != 200:
                return httpx.Response(state["status"], headers=state.get("headers", {}), text=KEY)
            return httpx.Response(
                200,
                headers={"mcp-session-id": "test-session"},
                json={
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {
                        "protocolVersion": "2025-11-25",
                        "capabilities": {},
                        "serverInfo": {"name": "offline-baizhi", "version": "1"},
                    },
                },
            )
        if "id" not in message:
            return httpx.Response(202)
        if method == "tools/list":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {
                        "tools": [
                            {"name": name, "inputSchema": {"type": "object"}}
                            for _, _, name in CASES
                        ]
                    },
                },
            )
        assert method == "tools/call"
        state["calls"].append(message["params"])
        state["started"].set()
        if state["delay"]:
            await asyncio.Event().wait()
        return httpx.Response(
            200, json={"jsonrpc": "2.0", "id": message["id"], "result": state["result"]}
        )

    def client(**kwargs):
        assert kwargs["follow_redirects"] is False
        assert kwargs["trust_env"] is False
        return original(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    monkeypatch.delenv("BAIZHI_API_KEY", raising=False)
    with patch(
        "socket.socket.connect", side_effect=AssertionError("Network is forbidden")
    ):
        yield state


@pytest.mark.parametrize("tool_class,arguments,remote_name", CASES)
@pytest.mark.parametrize("dispatch", ["run", "arun", "invoke", "ainvoke"])
def test_native_dispatch(server, tool_class, arguments, remote_name, dispatch):
    tool = tool_class(api_key=KEY)
    output = dispatch_tool(tool, arguments, dispatch)
    assert json.loads(output) == {"items": ["result"]}
    assert len(server["calls"]) == 1
    call = server["calls"][0]
    assert call["name"] == remote_name
    assert call["arguments"] == tool.args_schema.model_validate(arguments).model_dump(
        mode="json", exclude_none=True
    )
    if remote_name in ("web_scrape", "web_extract"):
        assert call["arguments"]["download"] is False
        download_schema = tool.args_schema.model_json_schema()["properties"]["download"]
        assert download_schema["const"] is False
    assert server["closed"]
    assert KEY not in tool.model_dump_json()
    assert KEY not in repr(tool)
    assert "api_key" not in tool.args_schema.model_json_schema()["properties"]
    assert tool.current_usage_count == 1


@pytest.mark.parametrize(
    "result,expected",
    [
        (
            {
                "content": [
                    {"type": "text", "text": "one"},
                    {"type": "text", "text": "two"},
                ]
            },
            "one\ntwo",
        ),
        (
            {
                "content": [{"type": "text", "text": "fallback"}],
                "structuredContent": {},
            },
            "{}",
        ),
    ],
)
def test_content_fallback(server, result, expected):
    server["result"] = result
    assert BaizhiSearchTool(api_key=KEY).run(query="test") == expected


@pytest.mark.parametrize("failure", ["tool", "http"])
def test_errors_are_sanitized_without_retry(server, failure, caplog):
    if failure == "tool":
        server["result"] = {"isError": True, "content": [{"type": "text", "text": KEY}]}
    else:
        server["status"] = 401
    with pytest.raises(RuntimeError) as error:
        BaizhiSearchTool(api_key=KEY).run(query="test")
    assert KEY not in str(error.value)
    assert KEY not in caplog.text
    assert len(server["calls"]) == (1 if failure == "tool" else 0)


def test_missing_key_is_local_and_environment_key_is_not_serialized(
    server, monkeypatch
):
    with pytest.raises(ValueError, match="BAIZHI_API_KEY"):
        BaizhiSearchTool().run(query="test")
    assert not server["calls"]
    monkeypatch.setenv("BAIZHI_API_KEY", KEY)
    tool = BaizhiSearchTool()
    assert KEY not in tool.model_dump_json()
    assert json.loads(tool.run(query="test")) == {"items": ["result"]}


@pytest.mark.parametrize(
    "tool_class,arguments",
    [
        (BaizhiSearchTool, {"query": "  "}),
        (BaizhiSearchTool, {"query": "test", "count": 51}),
        (BaizhiSearchTool, {"query": "test", "api_key": "agent-controlled"}),
        (BaizhiScrapeTool, {"url": "file:///etc/passwd"}),
        (BaizhiExtractTool, {"url": "https://example.com"}),
        (BaizhiExtractTool, {"url": "https://example.com", "instruction": " "}),
        (
            BaizhiExtractTool,
            {"url": "https://example.com", "fields": {"title": "object"}},
        ),
    ],
)
def test_invalid_inputs_do_not_call_service(server, tool_class, arguments):
    with pytest.raises(ValueError):
        tool_class(api_key=KEY).run(**arguments)
    assert not server["calls"]


def test_timeout_does_not_retry(server):
    server["delay"] = True
    with pytest.raises(TimeoutError, match="outcome may be unknown"):
        BaizhiSearchTool(api_key=KEY, timeout=0.05).run(query="test")
    assert len(server["calls"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("structured", [False, True])
async def test_cancellation_propagates(server, structured):
    server["delay"] = True
    tool = BaizhiSearchTool(api_key=KEY)
    coroutine = (
        tool.to_structured_tool().ainvoke({"query": "test"})
        if structured
        else tool.arun(query="test")
    )
    task = asyncio.create_task(coroutine)
    await asyncio.wait_for(server["started"].wait(), timeout=1)
    assert len(server["calls"]) == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(server["calls"]) == 1


def test_non_text_result_is_not_silently_dropped(server):
    server["result"] = {
        "content": [{"type": "image", "data": "", "mimeType": "image/png"}]
    }
    with pytest.raises(RuntimeError, match="non-text"):
        BaizhiSearchTool(api_key=KEY).run(query="test")


def test_invalid_key_is_not_exposed(server):
    with pytest.raises(ValueError) as error:
        BaizhiSearchTool(api_key=KEY + "\n")
    assert KEY not in str(error.value)


def test_public_exports_are_discoverable_in_tool_specs(server):
    from crewai_tools import tools
    from crewai_tools.generate_tool_specs import ToolSpecExtractor

    extractor = ToolSpecExtractor()
    for tool_class, _, _ in CASES:
        assert getattr(tools, tool_class.__name__) is tool_class
        extractor.extract_tool_info(tool_class)
    assert len(extractor.tools_spec) == 3
    for spec in extractor.tools_spec:
        assert spec["env_vars"][0]["name"] == "BAIZHI_API_KEY"
        assert "api_key" not in spec["run_params_schema"]["properties"]
        assert "api_key" not in spec["init_params_schema"]["properties"]


@pytest.mark.parametrize("value", ["https://example.com/path", "example.com/path", "localhost", "user@example.com"])
def test_domain_filter_rejects_non_domains(server, value):
    with pytest.raises(ValueError):
        BaizhiSearchTool(api_key=KEY).run(query="test", filter={"domains": [value]})
    assert not server["calls"]


@pytest.mark.parametrize("tool_class", [BaizhiScrapeTool, BaizhiExtractTool])
@pytest.mark.parametrize("dispatch", ["run", "arun", "invoke", "ainvoke"])
def test_urls_with_embedded_credentials_are_rejected(
    server, tool_class, dispatch, caplog, capsys
):
    username = "userx"
    password = "secretx"
    arguments = {"url": f"https://{username}:{password}@example.com"}
    if tool_class is BaizhiExtractTool:
        arguments["fields"] = {"title": "string"}
    with pytest.raises(ValueError, match="embedded credentials") as error:
        dispatch_tool(tool_class(api_key=KEY), arguments, dispatch)
    captured = capsys.readouterr()
    diagnostics = (
        str(error.value)
        + "".join(traceback.format_exception(error.value))
        + caplog.text
        + captured.out
        + captured.err
    )
    assert username not in diagnostics
    assert password not in diagnostics
    assert arguments["url"] not in diagnostics
    assert not server["requests"]
    assert not server["calls"]


@pytest.mark.parametrize("tool_class", [BaizhiScrapeTool, BaizhiExtractTool])
@pytest.mark.parametrize("dispatch", ["run", "arun", "invoke", "ainvoke"])
def test_downloads_are_disabled(server, tool_class, dispatch):
    arguments = {"url": "https://example.com", "download": True}
    if tool_class is BaizhiExtractTool:
        arguments["instruction"] = "Extract the title"
    with pytest.raises(ValueError, match="download"):
        dispatch_tool(tool_class(api_key=KEY), arguments, dispatch)
    assert not server["requests"]
    assert not server["calls"]


@pytest.mark.parametrize("tool_class,arguments,_remote_name", CASES)
def test_mcp_dependency_is_loaded_only_at_call_time(
    server, monkeypatch, tool_class, arguments, _remote_name
):
    from crewai_tools.tools.baizhi_tools import baizhi_tools

    original_import = builtins.__import__

    def import_without_mcp(name, *args, **kwargs):
        if name == "mcp" or name.startswith("mcp."):
            raise ModuleNotFoundError("Synthetic missing optional MCP dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_mcp)
    spec = importlib.util.spec_from_file_location(
        "baizhi_without_mcp", Path(baizhi_tools.__file__)
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    # Import a fresh module without adding duplicate process-wide log filters.
    with patch.object(logging.Logger, "addFilter"):
        spec.loader.exec_module(module)
    tool = getattr(module, tool_class.__name__)(api_key=KEY)
    assert tool.args_schema.model_json_schema()["properties"]
    with pytest.raises(ImportError, match=r"uv add 'crewai-tools\[mcp\]'"):
        tool.run(**arguments)
    assert not server["requests"]
    assert not server["calls"]


@pytest.mark.parametrize("structured", [True, False])
def test_successful_result_key_echo_is_redacted(server, structured):
    server["result"] = {"content": [{"type": "text", "text": KEY}]}
    if structured:
        server["result"]["structuredContent"] = {KEY: [KEY]}
    output = BaizhiSearchTool(api_key=KEY).run(query="test")
    assert KEY not in output
    assert "[REDACTED]" in output


def test_malformed_sdk_response_does_not_leak_in_default_logs(server, caplog):
    server["malformed"] = True
    with pytest.raises(TimeoutError):
        BaizhiSearchTool(api_key=KEY, timeout=0.1).run(query="test")
    assert KEY not in caplog.text
    assert "Error parsing JSON response" in caplog.text


def test_sdk_logging_outside_call_is_unchanged(server, caplog):
    logger = logging.getLogger("mcp.client.streamable_http")
    logger.warning("unrelated %s", "diagnostic")
    assert "unrelated diagnostic" in caplog.text
    assert caplog.records[-1].args == ("diagnostic",)


@pytest.mark.parametrize("location", ["https://example.com/redirect", "https://agent-toolkit.app.baizhi.cloud/other"])
def test_redirects_never_follow_credentials(server, location):
    server.update(status=307, headers={"Location": location})
    with pytest.raises(RuntimeError):
        BaizhiSearchTool(api_key=KEY).run(query="test")
    assert server["requests"] == ["https://agent-toolkit.app.baizhi.cloud/mcp"]
