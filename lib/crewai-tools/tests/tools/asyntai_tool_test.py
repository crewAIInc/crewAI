import json
from unittest.mock import Mock, patch

import pytest
import requests

from crewai_tools.tools.asyntai_tool import (
    AsyntaiAddTextTool,
    AsyntaiAddUrlTool,
    AsyntaiAskWebsiteTool,
    AsyntaiGetConversationTool,
    AsyntaiGetLeadsTool,
    AsyntaiListKnowledgeTool,
    AsyntaiListWebsitesTool,
)


REQUEST = "crewai_tools.tools.asyntai_tool.asyntai_tool.requests.request"
ALL_TOOLS = [
    AsyntaiAddTextTool,
    AsyntaiAddUrlTool,
    AsyntaiAskWebsiteTool,
    AsyntaiGetConversationTool,
    AsyntaiGetLeadsTool,
    AsyntaiListKnowledgeTool,
    AsyntaiListWebsitesTool,
]


def _response(payload, status_code=200):
    response = Mock(spec=requests.Response)
    response.status_code = status_code
    response.json.return_value = payload
    return response


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("ASYNTAI_API_KEY", "test_api_key")


def test_key_from_argument_wins_over_env():
    tool = AsyntaiListWebsitesTool(api_key="param_key")
    with patch(REQUEST, return_value=_response({"success": True})) as request:
        tool.run()
    assert request.call_args.kwargs["headers"]["Authorization"] == "Bearer param_key"


def test_key_is_not_in_repr():
    assert "secret_key" not in repr(AsyntaiListWebsitesTool(api_key="secret_key"))


def test_every_tool_declares_the_env_var():
    for tool_class in ALL_TOOLS:
        names = [env_var.name for env_var in tool_class().env_vars]
        assert names == ["ASYNTAI_API_KEY"]


def test_ask_sends_bearer_key_and_drops_none():
    with patch(
        REQUEST, return_value=_response({"success": True, "response": "Hi"})
    ) as request:
        AsyntaiAskWebsiteTool().run(question="Hello")

    assert request.call_args.args == ("POST", "https://asyntai.com/api/v1/chat/")
    kwargs = request.call_args.kwargs
    assert kwargs["headers"]["Authorization"] == "Bearer test_api_key"
    assert kwargs["json"] == {"message": "Hello"}
    assert kwargs["timeout"] == 60


def test_ask_returns_answer_and_session():
    with patch(
        REQUEST,
        return_value=_response(
            {"success": True, "response": "3 days.", "session_id": "api_abc"}
        ),
    ):
        result = json.loads(AsyntaiAskWebsiteTool().run(question="How fast?"))
    assert result == {"answer": "3 days.", "session_id": "api_abc"}


def test_ask_passes_session_and_website():
    with patch(
        REQUEST, return_value=_response({"success": True, "response": "Yes"})
    ) as request:
        AsyntaiAskWebsiteTool().run(question="UK?", session_id="s9", website_id=7)
    assert request.call_args.kwargs["json"] == {
        "message": "UK?",
        "session_id": "s9",
        "website_id": 7,
    }


def test_default_website_and_override():
    tool = AsyntaiAskWebsiteTool(website_id=12)
    with patch(
        REQUEST, return_value=_response({"success": True, "response": "Ok"})
    ) as request:
        tool.run(question="Hi")
        assert request.call_args.kwargs["json"]["website_id"] == 12
        tool.run(question="Hi", website_id=99)
        assert request.call_args.kwargs["json"]["website_id"] == 99


def test_custom_base_url_and_timeout():
    tool = AsyntaiListWebsitesTool(
        api_base_url="https://custom.example.com/api/v1/", timeout=30
    )
    with patch(REQUEST, return_value=_response({"success": True})) as request:
        tool.run()
    assert request.call_args.args[1] == "https://custom.example.com/api/v1/websites/"
    assert request.call_args.kwargs["timeout"] == 30


def test_no_api_key_makes_no_request(monkeypatch):
    monkeypatch.delenv("ASYNTAI_API_KEY")
    calls = [
        (AsyntaiAddTextTool, {"title": "t", "content": "c"}),
        (AsyntaiAddUrlTool, {"url": "https://example.com"}),
        (AsyntaiAskWebsiteTool, {"question": "Hi"}),
        (AsyntaiGetConversationTool, {"session_id": "s1"}),
        (AsyntaiGetLeadsTool, {}),
        (AsyntaiListKnowledgeTool, {}),
        (AsyntaiListWebsitesTool, {}),
    ]
    with patch(REQUEST) as request:
        for tool_class, arguments in calls:
            assert "ASYNTAI_API_KEY not set" in tool_class().run(**arguments)
    request.assert_not_called()


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (401, {"success": False}, "rejected the API key"),
        (403, {"success": False}, "needs a paid plan"),
        (429, {"success": False, "limit_reached": True}, "message limit"),
        (400, {"success": False, "error": "Website not found"}, "Website not found"),
        (500, {"success": False}, "Asyntai API error (HTTP 500)"),
        (200, ["unexpected"], "unexpected response body"),
    ],
)
def test_api_errors_become_text(status, body, expected):
    with patch(REQUEST, return_value=_response(body, status)):
        result = AsyntaiAskWebsiteTool().run(question="Hi")
    assert result.startswith("Error asking Asyntai:")
    assert expected in result


def test_non_json_response():
    response = Mock(spec=requests.Response)
    response.status_code = 502
    response.json.side_effect = ValueError("no json")
    with patch(REQUEST, return_value=response):
        result = AsyntaiListWebsitesTool().run()
    assert "non-JSON response (HTTP 502)" in result


def test_network_error():
    with patch(REQUEST, side_effect=requests.ConnectionError("connection refused")):
        result = AsyntaiAskWebsiteTool().run(question="Hi")
    assert result == "Error asking Asyntai: connection refused"


def test_list_websites():
    websites = [{"id": 1, "domain": "example.com", "is_primary": True}]
    with patch(
        REQUEST, return_value=_response({"success": True, "websites": websites})
    ) as request:
        assert json.loads(AsyntaiListWebsitesTool().run()) == websites
    assert request.call_args.args == ("GET", "https://asyntai.com/api/v1/websites/")


def test_add_text():
    with patch(
        REQUEST, return_value=_response({"success": True, "context_id": "ctx_1"})
    ) as request:
        result = json.loads(
            AsyntaiAddTextTool().run(title="Returns", content="30 days.", website_id=3)
        )
    assert result["context_id"] == "ctx_1"
    assert request.call_args.args[1].endswith("/knowledge/text/")
    assert request.call_args.kwargs["json"] == {
        "title": "Returns",
        "content": "30 days.",
        "website_id": 3,
    }


def test_add_url():
    with patch(
        REQUEST, return_value=_response({"success": True, "status": "queued"})
    ) as request:
        result = json.loads(AsyntaiAddUrlTool().run(url="https://example.com/faq"))
    assert result["status"] == "queued"
    assert request.call_args.args[1].endswith("/knowledge/url/")
    assert request.call_args.kwargs["json"] == {"url": "https://example.com/faq"}


def test_list_knowledge():
    entries = [{"id": "ctx_1", "title": "Returns"}]
    with patch(
        REQUEST, return_value=_response({"success": True, "entries": entries})
    ) as request:
        assert json.loads(AsyntaiListKnowledgeTool().run(limit=10)) == entries
    assert request.call_args.kwargs["params"] == {"limit": 10}


def test_get_leads_with_filters():
    leads = [{"email": "visitor@example.com", "session_id": "s1"}]
    with patch(
        REQUEST, return_value=_response({"success": True, "leads": leads})
    ) as request:
        result = json.loads(
            AsyntaiGetLeadsTool().run(
                limit=5, email="visitor@example.com", since="2026-01-31"
            )
        )
    assert result == leads
    assert request.call_args.args == ("GET", "https://asyntai.com/api/v1/leads/")
    assert request.call_args.kwargs["params"] == {
        "limit": 5,
        "email": "visitor@example.com",
        "since": "2026-01-31",
    }


def test_get_leads_empty():
    with patch(REQUEST, return_value=_response({"success": True, "leads": []})):
        assert json.loads(AsyntaiGetLeadsTool().run()) == []


def test_get_conversation():
    messages = [{"role": "user", "text": "Hi"}]
    with patch(
        REQUEST, return_value=_response({"success": True, "messages": messages})
    ) as request:
        result = json.loads(AsyntaiGetConversationTool().run(session_id="s1", limit=20))
    assert result == messages
    assert request.call_args.kwargs["params"] == {"session_id": "s1", "limit": 20}
