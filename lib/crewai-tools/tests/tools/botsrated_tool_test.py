import base64
import json
from unittest.mock import MagicMock, patch

import pytest
from crewai_tools.tools.botsrated_tool.botsrated_tool import (
    BASE_USDC,
    BOTSRATED_PAY_TO,
    BotsRatedCheckTool,
    BotsRatedPreflightTool,
)

SELLER_PAY_TO = "0xe9030014F5DAe217d0A152f02A043567b16c1aBf"
# Trimmed from a real GET https://botsrated.com/v1/preflight response.
PREFLIGHT = {
    "found": True,
    "host": "blockrun.ai",
    "latest_probe": {
        "ts": "2026-10-23T13:06:35Z",
        "url": "https://blockrun.ai/api/v1/chat/completions",
        "status": 402,
        "valid_402": True,
        "latency_ms": 1417,
        "live_amount": "2000",
        "live_payTo": SELLER_PAY_TO,
        "live_network": "eip155:8453",
        "receipt_url": "https://botsrated.com/receipt/05889891d26734d55948?host=blockrun.ai&day=2026-10-23",
    },
    "listing": {"payTo": SELLER_PAY_TO, "network": "eip155:8453"},
    "freshness": {"stale": False},
}
TERMS = {
    "x402Version": 2,
    "accepts": [
        {
            "scheme": "exact",
            "network": "eip155:8453",
            "amount": "2000",
            "asset": BASE_USDC,
            "payTo": BOTSRATED_PAY_TO,
        }
    ],
}


def _resp(status: int, body: dict, headers: dict | None = None) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.ok = 200 <= status < 300
    r.json.return_value = body
    r.headers = headers or {}
    return r


def _quote(terms: dict = TERMS) -> MagicMock:
    return _resp(
        402,
        {},
        {"payment-required": base64.b64encode(json.dumps(terms).encode()).decode()},
    )


@patch("crewai_tools.tools.botsrated_tool.botsrated_tool.requests.get")
def test_preflight_matches_observed(get):
    get.return_value = _resp(200, PREFLIGHT)
    out = json.loads(
        BotsRatedPreflightTool().run(
            url=PREFLIGHT["latest_probe"]["url"], pay_to=SELLER_PAY_TO.lower()
        )
    )
    assert out["pay_to_check"] == "matches_observed"
    assert "url=https%3A%2F%2Fblockrun.ai" in get.call_args.args[0]


@patch("crewai_tools.tools.botsrated_tool.botsrated_tool.requests.get")
def test_preflight_not_seen_and_other_network(get):
    get.return_value = _resp(200, PREFLIGHT)
    dead = "0x000000000000000000000000000000000000dEaD"
    assert (
        json.loads(BotsRatedPreflightTool().run(url="blockrun.ai", pay_to=dead))[
            "pay_to_check"
        ]
        == "not_seen"
    )
    other = json.loads(
        BotsRatedPreflightTool().run(
            url="blockrun.ai", pay_to=dead, network="eip155:84532"
        )
    )
    assert other["pay_to_check"] == "not_compared"


@patch("crewai_tools.tools.botsrated_tool.botsrated_tool.requests.get")
def test_preflight_not_covered(get):
    get.return_value = _resp(404, {"error": {"code": "host_not_covered"}})
    assert json.loads(BotsRatedPreflightTool().run(url="x.example"))["covered"] is False


@patch("crewai_tools.tools.botsrated_tool.botsrated_tool.requests.get")
def test_check_pays_and_enforces_budget(get):
    get.return_value = _quote()
    paying_get = MagicMock(return_value=_resp(200, {"host": "blockrun.ai"}))
    tool = BotsRatedCheckTool(budget_usd="0.003", paying_get=paying_get)
    assert json.loads(tool.run(host="blockrun.ai"))["remaining_usd"] == pytest.approx(
        0.001
    )
    assert json.loads(tool.run(host="blockrun.ai"))["error"] == "budget_exceeded"
    paying_get.assert_called_once_with(
        "https://botsrated.com/v1/check?host=blockrun.ai"
    )


@patch("crewai_tools.tools.botsrated_tool.botsrated_tool.requests.get")
def test_check_no_payment_on_free_404_or_bad_terms(get):
    paying_get = MagicMock()
    tool = BotsRatedCheckTool(budget_usd="1", paying_get=paying_get)
    get.return_value = _resp(404, {"error": {"code": "host_not_covered"}})
    assert json.loads(tool.run(host="x.example"))["error"] == "no_quote"
    get.return_value = _quote(
        {"accepts": [{**TERMS["accepts"][0], "payTo": SELLER_PAY_TO}]}
    )
    assert json.loads(tool.run(host="blockrun.ai"))["error"] == "terms_rejected"
    paying_get.assert_not_called()


@patch("crewai_tools.tools.botsrated_tool.botsrated_tool.requests.get")
def test_check_failure_releases_budget(get):
    get.return_value = _quote()
    tool = BotsRatedCheckTool(
        budget_usd="0.002",
        paying_get=MagicMock(side_effect=[_resp(500, {}), _resp(200, {})]),
    )
    assert json.loads(tool.run(host="blockrun.ai"))["error"] == "paid_call_failed"
    assert json.loads(tool.run(host="blockrun.ai"))["remaining_usd"] == 0


def test_check_requires_positive_budget():
    with pytest.raises(ValueError):
        BotsRatedCheckTool(budget_usd="0", paying_get=MagicMock())
