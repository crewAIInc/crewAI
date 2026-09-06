import json
from unittest.mock import MagicMock, patch

import requests

from crewai_tools.tools.dns_doctor_tool.dns_doctor_tool import (
    DnsDoctorDmarcUpgradeTool,
    DnsDoctorPropagationTool,
    DnsDoctorScanTool,
)

REPORT = {
    "domain": "example.com",
    "checks": [{"check": "dmarc", "status": "fail", "title": "No DMARC record found"}],
    "next_steps": {"report_url": "https://dnsdoctor.dev/scan/example.com"},
}
_PATH = "crewai_tools.tools.dns_doctor_tool.dns_doctor_tool.requests.post"


def _response(status: int, payload=None):
    response = MagicMock()
    response.status_code = status
    response.json.return_value = payload
    return response


def test_scan_posts_the_domain_and_relays_the_body_verbatim():
    with patch(_PATH, return_value=_response(200, REPORT)) as post:
        out = DnsDoctorScanTool().run(domain="example.com")
    assert json.loads(out) == REPORT
    (_url,) = post.call_args.args
    assert _url == "https://dnsdoctor.dev/api/v1/scan"
    assert post.call_args.kwargs["json"] == {"domain": "example.com"}
    assert post.call_args.kwargs["headers"]["User-Agent"].startswith(
        "dnsdoctor-crewai/"
    )


def test_dmarc_upgrade_uses_its_own_endpoint():
    body = {"record": None, "rationale": "reporting first", "current_policy": "none"}
    with patch(_PATH, return_value=_response(200, body)) as post:
        out = DnsDoctorDmarcUpgradeTool().run(domain="example.com")
    assert json.loads(out) == body
    assert post.call_args.args[0] == "https://dnsdoctor.dev/api/v1/dmarc-upgrade"


def test_propagation_sends_the_endpoints_field_names_and_omits_an_absent_expectation():
    with patch(_PATH, return_value=_response(200, {"verdict": "consistent"})) as post:
        DnsDoctorPropagationTool().run(name="www.example.com", record_type="NS")
    assert post.call_args.args[0] == "https://dnsdoctor.dev/api/tools/propagation-check"
    assert post.call_args.kwargs["json"] == {
        "name": "www.example.com",
        "record_type": "NS",
    }
    with patch(_PATH, return_value=_response(200, {"verdict": "propagated"})) as post:
        DnsDoctorPropagationTool().run(
            name="www.example.com", expected_value="203.0.113.10"
        )
    assert post.call_args.kwargs["json"] == {
        "name": "www.example.com",
        "record_type": "A",
        "expected_value": "203.0.113.10",
    }


def test_transport_failures_are_messages_that_are_not_verdicts():
    with patch(_PATH, return_value=_response(429, {"detail": "slow down"})):
        assert "not a verdict" in DnsDoctorScanTool().run(domain="example.com")
    with patch(_PATH, return_value=_response(503, {"detail": "unavailable"})):
        assert "not a verdict" in DnsDoctorScanTool().run(domain="example.com")
    with patch(_PATH, side_effect=requests.ConnectionError("down")):
        assert "not a verdict" in DnsDoctorScanTool().run(domain="example.com")
    paid = _response(402, {"x402Version": 1, "accepts": []})
    paid.text = '{"x402Version": 1, "accepts": []}'
    paid.headers = {"PAYMENT-REQUIRED": "eyJ4NDAyVmVyc2lvbiI6Mn0"}
    with patch(_PATH, return_value=paid):
        out = DnsDoctorScanTool().run(domain="example.com")
    assert "x402" in out and "not a verdict" in out
    # the offer travels with the text, so an x402-capable caller can pay and retry
    assert '"accepts": []' in out and "eyJ4NDAyVmVyc2lvbiI6Mn0" in out


def test_a_refusal_relays_the_apis_own_detail():
    with patch(_PATH, return_value=_response(422, {"detail": "malformed domain"})):
        assert "malformed domain" in DnsDoctorScanTool().run(domain="not a domain")


def test_token_is_sent_only_when_set(monkeypatch):
    monkeypatch.setenv("DNSDOCTOR_API_TOKEN", "dnsd_example")
    with patch(_PATH, return_value=_response(200, {})) as post:
        DnsDoctorScanTool().run(domain="example.com")
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer dnsd_example"
    monkeypatch.delenv("DNSDOCTOR_API_TOKEN")
    with patch(_PATH, return_value=_response(200, {})) as post:
        DnsDoctorScanTool().run(domain="example.com")
    assert "Authorization" not in post.call_args.kwargs["headers"]
