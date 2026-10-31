import json

import pytest

from crewai_tools import ViaSocketTools


CATALOG = {
    "service": {"name": "Slack"},
    "actions": [
        {
            "name": "Send Message",
            "description": "Send a message.",
            "action_id": "act1",
            "action_version_id": "ver1",
            "input_schema": {
                "steps": {"root": ["channel", "text"]},
                "blocks": {
                    "channel": {"key": "channel", "type": "string", "label": "Channel", "required": True},
                    "text": {"key": "text", "type": "string", "label": "Message", "required": True},
                },
            },
        }
    ],
    "triggers": [],
}


@pytest.fixture
def viasocket_api(monkeypatch):
    """Answers viaSocket's catalog, the user's flows and the action run; records what was run."""
    viasocket_client = pytest.importorskip("viasocket._client")
    runs = []

    def transport(method, url, headers, body, timeout):
        if "recommend/services" in url:
            reply = {"data": {"plugins": {"slack": {"rowid": "rowbu58rc"}}}}
        elif url.endswith("/func/scriK4LFg2kc"):
            reply = CATALOG
        elif "/integrations" in url:
            flow = {"id": "scriSLACK", "status": "active", "service_id": "rowbu58rc", "auth_id": "a1"}
            reply = {"success": True, "data": {"flows": [flow]}}
        else:
            runs.append(json.loads(body))
            reply = {"success": True, "data": {"ok": True}}
        return 200, json.dumps(reply)

    monkeypatch.setattr(viasocket_client, "_urllib_transport", transport)
    for name, value in (
        ("VIASOCKET_ORG_ID", "4160"),
        ("VIASOCKET_PROJECT_ID", "proj1"),
        ("VIASOCKET_EMBED_SECRET", "s3cret"),
    ):
        monkeypatch.setenv(name, value)
    return runs


def test_tools_run_the_action_on_the_users_connection(viasocket_api):
    tools = ViaSocketTools(user="user_1", apps=["slack"], actions={"slack": ["send-message"]})

    assert [tool.name for tool in tools] == ["slack_send-message"]
    assert tools[0].run(channel="C1", text="hi") == '{"ok": true}'
    assert viasocket_api == [{"action_version_id": "ver1", "inputData": {"channel": "C1", "text": "hi"}}]


def test_missing_credentials_are_named(monkeypatch):
    pytest.importorskip("viasocket")
    for name in ("VIASOCKET_ORG_ID", "VIASOCKET_PROJECT_ID", "VIASOCKET_EMBED_SECRET"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError, match="VIASOCKET_ORG_ID, VIASOCKET_PROJECT_ID, VIASOCKET_EMBED_SECRET is not set"):
        ViaSocketTools(user="user_1", apps=["slack"])
