import os
from unittest.mock import MagicMock, patch

import pytest
import requests as requests_lib

from crewai_tools.tools.precleared_tool.precleared_tool import (
    PreclearedTool,
    PreclearedToolSchema,
)


def _mock_response(status_code: int = 200, json_data: dict | None = None) -> MagicMock:
    resp = MagicMock(spec=requests_lib.Response)
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 400
    resp.json.return_value = json_data if json_data is not None else {}
    return resp


OK_ANSWER = {
    "verdict": "ok_to_email",
    "confidence": "high",
    "facts": {"recipient_country": "GB", "recipient_type": "limited_company"},
    "reasons": [
        {
            "rule_key": "uk_corporate_subscriber_ok",
            "text": "Limited companies are corporate subscribers, so PECR's consent rule for unsolicited marketing email does not apply to them.",
            "source_title": "PECR regulation 22",
            "source_url": "https://www.legislation.gov.uk/uksi/2003/2426/regulation/22",
        }
    ],
    "footer_text": "Say who you are, give a valid contact address and offer a free, simple way to opt out.",
    "next_step": "Send, with a footer that meets the footer requirements.",
    "receipt": {"id": "rcpt_test123"},
    "for_your_report": "Precleared cleared jane@acme.co.uk for a cold email under UK PECR (receipt rcpt_test123).",
}


@pytest.fixture(autouse=True)
def _no_key_by_default():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PRECLEARED_API_KEY", None)
        yield


def test_schema_defaults():
    args = PreclearedToolSchema(recipient_email="jane@acme.co.uk")
    assert args.sender_country == "GB"
    assert args.address_source == "unknown"
    assert args.existing_customer is False
    assert args.pitch is None


def test_run_posts_check_and_formats_answer():
    tool = PreclearedTool()
    with patch(
        "crewai_tools.tools.precleared_tool.precleared_tool.requests.post",
        return_value=_mock_response(200, OK_ANSWER),
    ) as post:
        result = tool._run(
            recipient_email="jane@acme.co.uk",
            sender_country="gb",
            address_source="published",
            pitch="Bookkeeping software for small firms",
            company_name="Acme Ltd",
        )

    post.assert_called_once()
    url = post.call_args.args[0]
    kwargs = post.call_args.kwargs
    assert url == "https://precleared.dev/api/v1/check?src=crewai"
    assert kwargs["json"] == {
        "recipient_email": "jane@acme.co.uk",
        "sender_country": "GB",
        "address_source": "published",
        "existing_customer": False,
        "pitch": "Bookkeeping software for small firms",
        "company_name": "Acme Ltd",
    }
    assert "Authorization" not in kwargs["headers"]
    assert kwargs["headers"]["User-Agent"].startswith("crewai-tools")
    assert kwargs["timeout"] == 40.0

    assert result.startswith("Verdict: ok_to_email")
    assert "Recipient country: GB" in result
    assert "Rule: Limited companies are corporate subscribers" in result
    assert "(PECR regulation 22)" in result
    assert "Footer must include: Say who you are" in result
    assert "Next step: Send, with a footer" in result
    assert "Receipt: rcpt_test123 (https://precleared.dev/r/rcpt_test123)" in result
    assert "For your report: Precleared cleared" in result
    assert "not legal advice" in result


def test_run_sends_key_when_set():
    tool = PreclearedTool()
    with patch.dict(os.environ, {"PRECLEARED_API_KEY": "pc_test_key"}):
        with patch(
            "crewai_tools.tools.precleared_tool.precleared_tool.requests.post",
            return_value=_mock_response(200, OK_ANSWER),
        ) as post:
            tool._run(recipient_email="jane@acme.co.uk")
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer pc_test_key"


def test_run_uppercases_recipient_country_and_drops_empty_optionals():
    tool = PreclearedTool()
    with patch(
        "crewai_tools.tools.precleared_tool.precleared_tool.requests.post",
        return_value=_mock_response(200, OK_ANSWER),
    ) as post:
        tool._run(
            recipient_email="jane@acme.co.uk",
            recipient_country="de",
            recipient_name="",
            existing_customer=True,
        )
    body = post.call_args.kwargs["json"]
    assert body["recipient_country"] == "DE"
    assert body["existing_customer"] is True
    assert "recipient_name" not in body


def test_run_requires_recipient_email():
    tool = PreclearedTool()
    with pytest.raises(ValueError, match="recipient_email is required"):
        tool._run(recipient_email="  ")


def test_consent_needed_answer_keeps_rule_and_receipt():
    answer = dict(OK_ANSWER)
    answer["verdict"] = "consent_needed"
    answer["reasons"] = [
        {
            "rule_key": "uk_sole_trader_consent",
            "text": "Sole traders are individual subscribers under PECR, so consent is needed first.",
            "source_title": "PECR regulation 22",
        }
    ]
    answer["next_step"] = "Do not send. Ask for consent through another channel first."
    tool = PreclearedTool()
    with patch(
        "crewai_tools.tools.precleared_tool.precleared_tool.requests.post",
        return_value=_mock_response(200, answer),
    ):
        result = tool._run(recipient_email="bob@bobsplumbing.co.uk")
    assert result.startswith("Verdict: consent_needed")
    assert "Sole traders are individual subscribers" in result
    assert "Next step: Do not send." in result


def test_daily_limit_returns_operator_message_instead_of_raising():
    tool = PreclearedTool()
    with patch(
        "crewai_tools.tools.precleared_tool.precleared_tool.requests.post",
        return_value=_mock_response(
            429,
            {
                "error": "daily_limit",
                "message": "The free daily allowance for your network is used up.",
                "operator_message": "Get a free key at https://precleared.dev/get-key for an allowance of your own.",
                "get_key_url": "https://precleared.dev/get-key",
            },
        ),
    ):
        result = tool._run(recipient_email="jane@acme.co.uk")
    assert "Precleared did not check this address (daily_limit)" in result
    assert "For the person running this agent: Get a free key" in result
    assert "https://precleared.dev/get-key" in result
    assert "treat the address as consent needed" in result


def test_payment_required_mentions_buy_url():
    tool = PreclearedTool()
    with patch(
        "crewai_tools.tools.precleared_tool.precleared_tool.requests.post",
        return_value=_mock_response(
            402,
            {
                "error": "payment_required",
                "message": "This key has no checks left.",
                "buy_url": "https://precleared.dev/buy?key=pc_abc",
            },
        ),
    ):
        result = tool._run(recipient_email="jane@acme.co.uk")
    assert "payment_required" in result
    assert "https://precleared.dev/buy?key=pc_abc" in result


def test_network_error_returns_text_not_exception():
    tool = PreclearedTool()
    with patch(
        "crewai_tools.tools.precleared_tool.precleared_tool.requests.post",
        side_effect=requests_lib.ConnectionError("boom"),
    ):
        result = tool._run(recipient_email="jane@acme.co.uk")
    assert "could not be reached" in result
    assert "ConnectionError" in result


def test_unreadable_body_is_reported():
    resp = _mock_response(502)
    resp.json.side_effect = ValueError("not json")
    tool = PreclearedTool()
    with patch(
        "crewai_tools.tools.precleared_tool.precleared_tool.requests.post",
        return_value=resp,
    ):
        result = tool._run(recipient_email="jane@acme.co.uk")
    assert "HTTP 502" in result
    assert "consent needed" in result


def test_tool_metadata():
    tool = PreclearedTool()
    assert tool.name == "Precleared cold email check"
    assert tool.args_schema is PreclearedToolSchema
    assert [env.name for env in tool.env_vars] == ["PRECLEARED_API_KEY"]
    assert tool.env_vars[0].required is False
