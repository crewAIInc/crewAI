import base64
import json
from unittest.mock import MagicMock, patch

import pytest
from crewai_tools.tools.botsrated_tool.botsrated_tool import (
    BASE_USDC,
    BOTSRATED_PAY_TO,
    BotsRatedCheckTool,
    BotsRatedPreflightTool,
    botsrated_payment_policy,
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
            "amount": "1000",
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
def test_preflight_listing_alone_does_not_confirm(get):
    listed = "0x1111111111111111111111111111111111111111"
    body = {**PREFLIGHT, "listing": {"payTo": listed, "network": "eip155:8453"}}
    get.return_value = _resp(200, body)
    out = json.loads(BotsRatedPreflightTool().run(url="blockrun.ai", pay_to=listed))
    assert out["pay_to_check"] == "listing_only"
    no_probe = {**PREFLIGHT, "latest_probe": None}
    get.return_value = _resp(200, no_probe)
    out = json.loads(
        BotsRatedPreflightTool().run(url="blockrun.ai", pay_to=SELLER_PAY_TO)
    )
    assert out["pay_to_check"] == "listing_only"


@patch("crewai_tools.tools.botsrated_tool.botsrated_tool.requests.get")
def test_preflight_not_covered(get):
    get.return_value = _resp(404, {"error": {"code": "host_not_covered"}})
    assert json.loads(BotsRatedPreflightTool().run(url="x.example"))["covered"] is False


@patch("crewai_tools.tools.botsrated_tool.botsrated_tool.requests.get")
def test_check_pays_and_enforces_budget(get):
    get.return_value = _quote()
    paying_get = MagicMock(return_value=_resp(200, {"host": "blockrun.ai"}))
    tool = BotsRatedCheckTool(budget_usd="0.0015", paying_get=paying_get)
    first = json.loads(tool.run(host="blockrun.ai"))
    assert first["remaining_usd"] == pytest.approx(0.0005)
    assert first["charged"] is True
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
def test_check_failure_keeps_reservation(get):
    # A payment may settle before the paid call fails, so the budget is not refunded.
    get.return_value = _quote()
    paying_get = MagicMock(side_effect=[_resp(500, {}), _resp(200, {})])
    tool = BotsRatedCheckTool(budget_usd="0.001", paying_get=paying_get)
    first = json.loads(tool.run(host="blockrun.ai"))
    assert first["error"] == "paid_call_failed"
    assert first["may_have_been_charged"] is True
    assert tool.remaining_usd == 0
    assert json.loads(tool.run(host="blockrun.ai"))["error"] == "budget_exceeded"
    assert paying_get.call_count == 1


@patch("crewai_tools.tools.botsrated_tool.botsrated_tool.requests.get")
def test_check_exception_keeps_reservation(get):
    get.return_value = _quote()
    tool = BotsRatedCheckTool(
        budget_usd="0.001", paying_get=MagicMock(side_effect=TimeoutError("read"))
    )
    assert json.loads(tool.run(host="blockrun.ai"))["error"] == "paid_call_failed"
    assert tool.remaining_usd == 0


@patch("crewai_tools.tools.botsrated_tool.botsrated_tool.requests.get")
def test_check_rejects_quote_with_any_foreign_option(get):
    foreign = {**TERMS["accepts"][0], "amount": "500000", "payTo": SELLER_PAY_TO}
    get.return_value = _quote({"accepts": [foreign, TERMS["accepts"][0]]})
    paying_get = MagicMock()
    tool = BotsRatedCheckTool(budget_usd="1", paying_get=paying_get)
    assert json.loads(tool.run(host="blockrun.ai"))["error"] == "terms_rejected"
    get.return_value = _quote({"accepts": []})
    assert json.loads(tool.run(host="blockrun.ai"))["error"] == "terms_rejected"
    paying_get.assert_not_called()


@patch("crewai_tools.tools.botsrated_tool.botsrated_tool.requests.get")
def test_check_quote_does_not_follow_redirects(get):
    get.return_value = _resp(302, {})
    paying_get = MagicMock()
    tool = BotsRatedCheckTool(budget_usd="1", paying_get=paying_get)
    assert json.loads(tool.run(host="blockrun.ai"))["error"] == "no_quote"
    assert get.call_args.kwargs["allow_redirects"] is False
    paying_get.assert_not_called()


def test_payment_policy_keeps_only_botsrated_terms():
    class Req:
        def __init__(self, **kw):
            self.__dict__.update(kw)

        def get_amount(self):
            return self.amount

    ok = Req(scheme="exact", network="eip155:8453", asset=BASE_USDC,
             pay_to=BOTSRATED_PAY_TO, amount="1000")
    pricey = Req(**{**ok.__dict__, "amount": "500000"})
    other_to = Req(**{**ok.__dict__, "pay_to": SELLER_PAY_TO})
    other_net = Req(**{**ok.__dict__, "network": "eip155:1"})
    assert botsrated_payment_policy(2, [pricey, other_to, other_net, ok]) == [ok]
    assert botsrated_payment_policy(2, [pricey]) == []


FREE = {
    "host": "blockrun.ai",
    "charged": False,
    "free_quota": {"route": "check", "used": 1, "limit": 100, "remaining": 99},
}


@patch("crewai_tools.tools.botsrated_tool.botsrated_tool.requests.get")
def test_check_free_quota_answer_is_not_charged(get):
    get.return_value = _resp(200, FREE)
    paying_get = MagicMock()
    tool = BotsRatedCheckTool(budget_usd="0.001", paying_get=paying_get)
    out = json.loads(tool.run(host="blockrun.ai"))
    assert out["charged"] is False and out["remaining_usd"] == pytest.approx(0.001)
    paying_get.assert_not_called()


@patch("crewai_tools.tools.botsrated_tool.botsrated_tool.requests.get")
def test_check_rejects_quote_above_price(get):
    get.return_value = _quote({"accepts": [{**TERMS["accepts"][0], "amount": "2000"}]})
    paying_get = MagicMock()
    tool = BotsRatedCheckTool(budget_usd="1", paying_get=paying_get)
    assert json.loads(tool.run(host="blockrun.ai"))["error"] == "terms_rejected"
    paying_get.assert_not_called()


def test_check_requires_positive_budget():
    with pytest.raises(ValueError):
        BotsRatedCheckTool(budget_usd="0", paying_get=MagicMock())
