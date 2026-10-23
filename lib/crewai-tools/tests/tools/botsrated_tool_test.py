import base64
import json
from unittest.mock import MagicMock, patch

import requests

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


MOD = "crewai_tools.tools.botsrated_tool.botsrated_tool"


@patch(f"{MOD}._sign", return_value="sig")
@patch(f"{MOD}.requests.get")
def test_check_signs_validated_quote_and_enforces_budget(get, sign):
    get.side_effect = [_quote(), _resp(200, {"host": "blockrun.ai"}), _quote()]
    client = MagicMock()
    tool = BotsRatedCheckTool(budget_usd="0.0015", x402_client=client)
    first = json.loads(tool.run(host="blockrun.ai"))
    assert first["charged"] is True
    assert first["remaining_usd"] == pytest.approx(0.0005)
    sign.assert_called_once_with(client, TERMS)
    paid = get.call_args_list[1]
    assert paid.args[0] == "https://botsrated.com/v1/check?host=blockrun.ai"
    assert paid.kwargs["headers"]["PAYMENT-SIGNATURE"] == "sig"
    assert paid.kwargs["allow_redirects"] is False
    assert json.loads(tool.run(host="blockrun.ai"))["error"] == "budget_exceeded"
    assert sign.call_count == 1


@patch(f"{MOD}._sign")
@patch(f"{MOD}.requests.get")
def test_check_no_signature_on_404_bad_terms_or_redirect(get, sign):
    tool = BotsRatedCheckTool(budget_usd="1", x402_client=MagicMock())
    get.return_value = _resp(404, {"error": {"code": "host_not_covered"}})
    assert json.loads(tool.run(host="x.example"))["error"] == "no_quote"
    get.return_value = _resp(302, {})
    assert json.loads(tool.run(host="blockrun.ai"))["error"] == "no_quote"
    assert get.call_args.kwargs["allow_redirects"] is False
    for accepts in (
        [{**TERMS["accepts"][0], "payTo": SELLER_PAY_TO}],
        [{**TERMS["accepts"][0], "amount": "2000"}],
        [{**TERMS["accepts"][0], "amount": "500000", "payTo": SELLER_PAY_TO}, TERMS["accepts"][0]],
        [],
    ):
        get.return_value = _quote({"accepts": accepts})
        assert json.loads(tool.run(host="blockrun.ai"))["error"] == "terms_rejected"
    sign.assert_not_called()
    assert tool.remaining_usd == 1


@patch(f"{MOD}._sign", return_value="sig")
@patch(f"{MOD}.requests.get")
def test_check_failure_after_signing_keeps_reservation(get, sign):
    get.side_effect = [_quote(), _resp(500, {}), _quote()]
    tool = BotsRatedCheckTool(budget_usd="0.001", x402_client=MagicMock())
    first = json.loads(tool.run(host="blockrun.ai"))
    assert first["error"] == "paid_call_failed"
    assert first["may_have_been_charged"] is True
    assert tool.remaining_usd == 0
    assert json.loads(tool.run(host="blockrun.ai"))["error"] == "budget_exceeded"


@patch(f"{MOD}._sign", return_value="sig")
@patch(f"{MOD}.requests.get")
def test_check_transport_error_after_signing_keeps_reservation(get, sign):
    get.side_effect = [_quote(), requests.Timeout("read")]
    tool = BotsRatedCheckTool(budget_usd="0.001", x402_client=MagicMock())
    out = json.loads(tool.run(host="blockrun.ai"))
    assert out["error"] == "paid_call_failed" and out["may_have_been_charged"] is True
    assert tool.remaining_usd == 0


@patch(f"{MOD}._sign", return_value="sig")
@patch(f"{MOD}.requests.get")
def test_check_paid_free_quota_body_does_not_refund(get, sign):
    get.side_effect = [_quote(), _resp(200, FREE)]
    tool = BotsRatedCheckTool(budget_usd="0.001", x402_client=MagicMock())
    assert json.loads(tool.run(host="blockrun.ai"))["charged"] is True
    assert tool.remaining_usd == 0


@patch(f"{MOD}._sign", side_effect=RuntimeError("no scheme for eip155:8453"))
@patch(f"{MOD}.requests.get")
def test_check_signing_failure_releases_reservation(get, sign):
    get.return_value = _quote()
    tool = BotsRatedCheckTool(budget_usd="0.001", x402_client=MagicMock())
    assert json.loads(tool.run(host="blockrun.ai"))["error"] == "payment_not_created"
    assert tool.remaining_usd == pytest.approx(0.001)
    assert get.call_count == 1


FREE = {
    "host": "blockrun.ai",
    "charged": False,
    "free_quota": {"route": "check", "used": 1, "limit": 100, "remaining": 99},
}


@patch(f"{MOD}._sign")
@patch(f"{MOD}.requests.get")
def test_check_free_quota_answer_is_not_charged(get, sign):
    get.return_value = _resp(200, FREE)
    tool = BotsRatedCheckTool(budget_usd="0.001", x402_client=MagicMock())
    out = json.loads(tool.run(host="blockrun.ai"))
    assert out["charged"] is False and out["remaining_usd"] == pytest.approx(0.001)
    sign.assert_not_called()


def test_sign_with_real_x402_client():
    pytest.importorskip("x402.mechanisms.evm")
    from eth_account import Account
    from x402 import x402ClientSync
    from x402.http.utils import decode_payment_signature_header
    from x402.mechanisms.evm import EthAccountSigner
    from x402.mechanisms.evm.exact import ExactEvmScheme

    from crewai_tools.tools.botsrated_tool.botsrated_tool import _sign

    client = x402ClientSync()
    client.register(
        "eip155:8453", ExactEvmScheme(signer=EthAccountSigner(Account.create()))
    )
    terms = {
        **TERMS,
        "resource": {"url": "https://botsrated.com/v1/check?host=blockrun.ai"},
        "accepts": [
            {**TERMS["accepts"][0], "maxTimeoutSeconds": 300,
             "extra": {"name": "USD Coin", "version": "2"}}
        ],
    }
    payload = decode_payment_signature_header(_sign(client, terms))
    assert payload.accepted.pay_to.lower() == BOTSRATED_PAY_TO.lower()
    assert payload.accepted.amount == "1000"


def test_check_requires_positive_budget():
    with pytest.raises(ValueError):
        BotsRatedCheckTool(budget_usd="0", x402_client=MagicMock())
