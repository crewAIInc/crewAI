import json
from unittest.mock import patch

import pytest
import requests

from crewai_tools.tools.darkmoon_tool.darkmoon_tool import (
    DarkmoonGetFindingsTool,
    DarkmoonListCampaignsTool,
    DarkmoonRunPentestTool,
    DarkmoonToolError,
)


MODULE = "crewai_tools.tools.darkmoon_tool.darkmoon_tool"
BASE = "http://darkmoon.test:8000"

FINDING = {
    "title": "Reflected XSS in /search",
    "severity": "high",
    "cvss_score": 7.4,
    "category": "xss_reflected",
    "status": "confirmed",
}


class _Response:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


def _response(status=200, body=None):
    return _Response(status, body)


class FakeDarkmoon:
    """Routes requests.request calls to canned Dashboard API responses."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __call__(self, method, url, headers=None, json=None, timeout=None):
        path = url[len(BASE) :]
        self.calls.append((method, path, headers, json))
        handler = self.routes.get((method, path.split("?")[0]))
        if handler is None:
            return _response(404, {"detail": "Not found"})
        return handler(path, json) if not isinstance(handler, _Response) else handler


@pytest.fixture(autouse=True)
def darkmoon_env(monkeypatch):
    monkeypatch.setenv("DARKMOON_BASE_URL", BASE + "/")
    monkeypatch.setenv("DARKMOON_USERNAME", "admin")
    monkeypatch.setenv("DARKMOON_PASSWORD", "secret")


LOGIN = _response(200, {"token": "jwt-123"})


def test_missing_settings_raise(monkeypatch):
    monkeypatch.delenv("DARKMOON_BASE_URL")
    monkeypatch.delenv("DARKMOON_PASSWORD")
    with pytest.raises(DarkmoonToolError) as exc:
        DarkmoonListCampaignsTool().run()
    assert "DARKMOON_BASE_URL" in str(exc.value)
    assert "DARKMOON_PASSWORD" in str(exc.value)


def test_list_campaigns_logs_in_and_sends_bearer():
    fake = FakeDarkmoon(
        {
            ("POST", "/api/v1/auth/login"): LOGIN,
            ("GET", "/api/v1/campaigns"): _response(
                200, {"data": [{"id": "camp_1"}, {"id": "camp_2"}], "total": 2}
            ),
        }
    )
    with patch(f"{MODULE}.requests.request", fake):
        result = json.loads(DarkmoonListCampaignsTool().run())

    assert result == {"total": 2, "campaigns": [{"id": "camp_1"}, {"id": "camp_2"}]}
    login = fake.calls[0]
    assert login[0:2] == ("POST", "/api/v1/auth/login")
    assert login[3] == {"username": "admin", "password": "secret"}
    assert "Authorization" not in login[2]
    assert fake.calls[1][2]["Authorization"] == "Bearer jwt-123"


def test_login_is_cached_across_calls():
    fake = FakeDarkmoon(
        {
            ("POST", "/api/v1/auth/login"): LOGIN,
            ("GET", "/api/v1/campaigns"): _response(200, {"data": []}),
        }
    )
    tool = DarkmoonListCampaignsTool()
    with patch(f"{MODULE}.requests.request", fake):
        tool.run()
        tool.run()
    assert [c[1] for c in fake.calls].count("/api/v1/auth/login") == 1


def test_get_findings_returns_findings_and_stats():
    fake = FakeDarkmoon(
        {
            ("POST", "/api/v1/auth/login"): LOGIN,
            ("GET", "/api/v1/vulnerabilities"): _response(
                200, {"data": [FINDING], "total": 1, "stats": {"high": 1}}
            ),
        }
    )
    with patch(f"{MODULE}.requests.request", fake):
        result = json.loads(DarkmoonGetFindingsTool().run(campaign_id="camp 1"))

    assert result["campaign_id"] == "camp 1"
    assert result["total"] == 1
    assert result["stats"] == {"high": 1}
    assert result["findings"] == [FINDING]
    assert fake.calls[-1][1] == "/api/v1/vulnerabilities?campaign_id=camp%201"


def test_api_error_detail_is_surfaced_without_secrets():
    fake = FakeDarkmoon(
        {("POST", "/api/v1/auth/login"): _response(401, {"detail": "Bad credentials"})}
    )
    with patch(f"{MODULE}.requests.request", fake):
        with pytest.raises(DarkmoonToolError) as exc:
            DarkmoonListCampaignsTool().run()
    assert "401" in str(exc.value)
    assert "Bad credentials" in str(exc.value)
    assert "secret" not in str(exc.value)


def test_network_failure_is_wrapped():
    with patch(
        f"{MODULE}.requests.request", side_effect=requests.ConnectionError("refused")
    ):
        with pytest.raises(DarkmoonToolError) as exc:
            DarkmoonListCampaignsTool().run()
    assert "Request to Darkmoon failed" in str(exc.value)


def _run_routes(log_events, campaigns_before, campaigns_after):
    campaign_lists = iter([campaigns_before, campaigns_after])
    return {
        ("POST", "/api/v1/auth/login"): LOGIN,
        ("GET", "/api/v1/campaigns"): lambda p, b: _response(
            200, {"data": next(campaign_lists)}
        ),
        ("POST", "/api/v1/run/campaign"): lambda p, b: _response(
            200, {"run_id": "run_9", "pid": 42}
        ),
        ("GET", "/api/v1/run/logs/run_9"): lambda p, b: _response(
            200, {"data": next(log_events)}
        ),
        ("GET", "/api/v1/vulnerabilities"): _response(
            200, {"data": [FINDING], "total": 1, "stats": {"high": 1}}
        ),
    }


def test_run_pentest_waits_and_returns_findings():
    log_events = iter(
        [
            [{"type": "run_started"}],
            [{"type": "run_started"}, {"type": "run_completed"}],
        ]
    )
    fake = FakeDarkmoon(
        _run_routes(
            log_events,
            [{"id": "camp_old", "date": "2026-01-01"}],
            [
                {"id": "camp_old", "date": "2026-01-01"},
                {"id": "camp_new_app.test", "date": "2026-10-02"},
            ],
        )
    )
    with patch(f"{MODULE}.requests.request", fake), patch(f"{MODULE}.time.sleep") as sleep:
        result = json.loads(
            DarkmoonRunPentestTool().run(
                target="app.test", focus="auth, injection", severity="high"
            )
        )

    start = next(c for c in fake.calls if c[1] == "/api/v1/run/campaign")
    assert start[3] == {
        "target": "app.test",
        "focus": ["auth", "injection"],
        "severity": "high",
    }
    assert sleep.call_count == 1
    assert result["run_id"] == "run_9"
    assert result["campaign_id"] == "camp_new_app.test"
    assert result["timed_out"] is False
    assert result["total"] == 1
    assert result["findings"] == [FINDING]


def test_run_pentest_without_waiting_returns_run_id_only():
    fake = FakeDarkmoon(_run_routes(iter([]), [], []))
    with patch(f"{MODULE}.requests.request", fake):
        result = json.loads(
            DarkmoonRunPentestTool().run(target="app.test", wait_for_completion=False)
        )
    assert result == {"status": "started", "run_id": "run_9", "target": "app.test"}
    assert not any("/run/logs/" in c[1] for c in fake.calls)


def test_run_pentest_times_out_and_reports_it():
    log_events = iter([[{"type": "run_started"}]] * 50)
    fake = FakeDarkmoon(_run_routes(log_events, [], []))
    clock = iter([0.0, 0.0, 10.0, 10.0, 10.0])
    with patch(f"{MODULE}.requests.request", fake), patch(
        f"{MODULE}.time.sleep"
    ), patch(f"{MODULE}.time.monotonic", lambda: next(clock)):
        result = json.loads(
            DarkmoonRunPentestTool().run(target="app.test", timeout_seconds=5)
        )
    assert result["timed_out"] is True
    assert result["campaign_id"] is None
    assert result["findings"] == []


def test_run_pentest_rejects_blank_target():
    with pytest.raises(DarkmoonToolError):
        DarkmoonRunPentestTool()._run(target="   ")


def test_tools_declare_env_vars():
    names = {e.name for e in DarkmoonRunPentestTool().env_vars}
    assert names == {"DARKMOON_BASE_URL", "DARKMOON_USERNAME", "DARKMOON_PASSWORD"}


def test_run_pentest_never_reports_a_preexisting_campaign():
    log_events = iter([[{"type": "run_completed"}]])
    campaigns = [{"id": "camp_old", "date": "2026-01-01"}]
    fake = FakeDarkmoon(_run_routes(log_events, campaigns, campaigns))
    with patch(f"{MODULE}.requests.request", fake), patch(f"{MODULE}.time.sleep"):
        result = json.loads(DarkmoonRunPentestTool().run(target="app.test"))
    assert result["campaign_id"] is None
    assert result["findings"] == []
