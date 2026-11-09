import base64
import copy
import json
import os
from unittest.mock import patch

from crewai_tools.tools.stallrow_tools.stallrow_page_metadata_tool import (
    StallRowPageMetadataTool,
)
import pytest


# Hardhat's well-known development account #0 — a public test vector,
# used so the paid-flow tests can really sign EIP-712 payloads.
TEST_PRIVATE_KEY = (
    "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
)
TEST_ADDRESS = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
PAY_TO = "0x097f9685D1e0752bF8B02d61E5Ce9be7eA0FBA7F"
USDC_BASE = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"

# The x402 v2 challenge shape StallRow's /v1/page-metadata answers with,
# captured from the live endpoint (bazaar extension trimmed).
CHALLENGE = {
    "x402Version": 2,
    "error": "Payment required",
    "resource": {
        "url": "https://stall-row-production.up.railway.app/v1/page-metadata",
        "description": "One web page's declared metadata in one call.",
        "mimeType": "application/json",
        "serviceName": "stall-row",
        "tags": ["fetch", "metadata", "web"],
    },
    "accepts": [
        {
            "scheme": "exact",
            "network": "eip155:8453",
            "asset": USDC_BASE,
            "amount": "5000",
            "payTo": PAY_TO,
            "maxTimeoutSeconds": 300,
            "extra": {"name": "USD Coin", "version": "2"},
        }
    ],
    "extensions": {"bazaar": {"info": {"input": {"type": "http"}}}},
}

METADATA = {
    "url": "https://example.com/",
    "final_url": "https://example.com/",
    "title": "Example Domain",
    "description": None,
    "canonical": None,
    "lang": "en",
    "favicon": "https://example.com/favicon.ico",
    "og": {},
    "twitter": {},
    "headings": {"h1": ["Example Domain"], "h2": []},
    "word_count": 29,
}

POST_PATH = "crewai_tools.tools.stallrow_tools.x402_client.requests.post"


class _FakeResponse:
    def __init__(self, status_code, json_body=None, headers=None, text=""):
        self.status_code = status_code
        self._json_body = json_body
        self.headers = headers or {}
        self.text = text

    def json(self):
        if self._json_body is None:
            raise ValueError("response has no JSON body")
        return self._json_body


def _challenge_response(
    amount="5000", *, via_header=True, asset=USDC_BASE, max_timeout_seconds=300
):
    challenge = copy.deepcopy(CHALLENGE)
    challenge["accepts"][0]["amount"] = amount
    challenge["accepts"][0]["asset"] = asset
    challenge["accepts"][0]["maxTimeoutSeconds"] = max_timeout_seconds
    headers = {}
    if via_header:
        headers["payment-required"] = base64.b64encode(
            json.dumps(challenge).encode()
        ).decode()
    return _FakeResponse(
        402, json_body=challenge, headers=headers, text=json.dumps(challenge)
    )


def _ok_response():
    return _FakeResponse(200, json_body=copy.deepcopy(METADATA))


@pytest.fixture(autouse=True)
def _clean_env():
    with patch.dict(os.environ, {}, clear=True):
        yield


@pytest.fixture
def _wallet_env():
    with patch.dict(
        os.environ, {"STALLROW_WALLET_PRIVATE_KEY": TEST_PRIVATE_KEY}
    ):
        yield


def test_tool_constructs_without_wallet_configured():
    tool = StallRowPageMetadataTool()
    assert tool.name == "StallRow Page Metadata"
    assert tool.base_url == "https://stall-row-production.up.railway.app"


@patch(POST_PATH)
def test_no_wallet_fails_closed_and_pays_nothing(mock_post):
    mock_post.return_value = _challenge_response()
    tool = StallRowPageMetadataTool()

    result = tool._run("https://example.com/")

    assert result.startswith("Error fetching page metadata:")
    assert "STALLROW_WALLET_PRIVATE_KEY" in result
    # Exactly one (unpaid) request, and it carried no payment header.
    assert mock_post.call_count == 1
    assert "PAYMENT-SIGNATURE" not in (mock_post.call_args.kwargs.get("headers") or {})


@patch(POST_PATH)
def test_paid_flow_signs_and_retries(mock_post, _wallet_env):
    mock_post.side_effect = [_challenge_response(), _ok_response()]
    tool = StallRowPageMetadataTool()

    result = json.loads(tool._run("https://example.com/"))

    assert result == METADATA
    assert mock_post.call_count == 2
    header = mock_post.call_args_list[1].kwargs["headers"]["PAYMENT-SIGNATURE"]
    payload = json.loads(base64.b64decode(header))
    assert payload["x402Version"] == 2
    assert payload["accepted"]["amount"] == "5000"
    assert payload["accepted"]["payTo"] == PAY_TO
    # The bazaar extension must be echoed, or catalogues never index the call.
    assert payload["extensions"] == CHALLENGE["extensions"]
    authorization = payload["payload"]["authorization"]
    assert authorization["from"] == TEST_ADDRESS
    assert authorization["to"] == PAY_TO
    assert authorization["value"] == "5000"
    signature = payload["payload"]["signature"]
    assert signature.startswith("0x") and len(signature) == 132


@patch(POST_PATH)
def test_challenge_accepted_from_body_when_header_missing(mock_post, _wallet_env):
    mock_post.side_effect = [
        _challenge_response(via_header=False),
        _ok_response(),
    ]
    tool = StallRowPageMetadataTool()

    result = json.loads(tool._run("https://example.com/"))

    assert result == METADATA
    assert mock_post.call_count == 2


@patch(POST_PATH)
def test_price_above_cap_is_refused_before_signing(mock_post, _wallet_env):
    mock_post.return_value = _challenge_response(amount="6000")
    tool = StallRowPageMetadataTool()

    result = tool._run("https://example.com/")

    assert result.startswith("Error fetching page metadata:")
    assert "cap" in result
    # Refused after the unpaid probe — the paid retry never happened.
    assert mock_post.call_count == 1


@patch(POST_PATH)
def test_non_usdc_asset_is_refused_before_signing(mock_post, _wallet_env):
    # WETH on Base, priced under the cap: the amount is fine, but this
    # client settles USDC only, so the challenge must be refused
    # before anything is signed.
    mock_post.return_value = _challenge_response(
        amount="4000", asset="0x4200000000000000000000000000000000000006"
    )
    tool = StallRowPageMetadataTool()

    result = tool._run("https://example.com/")

    assert result.startswith("Error fetching page metadata:")
    assert "USDC" in result
    assert "0x4200000000000000000000000000000000000006" in result
    # Refused after the unpaid probe — the paid retry never happened.
    assert mock_post.call_count == 1


@patch(POST_PATH)
def test_excessive_timeout_is_refused_before_signing(mock_post, _wallet_env):
    # A multi-year validity window priced under the cap in the right
    # asset: the amount and asset are fine, but a signed authorization
    # must never stay settleable that long at the endpoint's say-so.
    mock_post.return_value = _challenge_response(max_timeout_seconds=63_072_000)
    tool = StallRowPageMetadataTool()

    result = tool._run("https://example.com/")

    assert result.startswith("Error fetching page metadata:")
    assert "validity window" in result
    # Refused after the unpaid probe — the paid retry never happened.
    assert mock_post.call_count == 1


@patch(POST_PATH)
def test_cap_override_via_env_allows_higher_price(mock_post):
    with patch.dict(
        os.environ,
        {
            "STALLROW_WALLET_PRIVATE_KEY": TEST_PRIVATE_KEY,
            "STALLROW_MAX_PRICE_USDC": "0.01",
        },
    ):
        mock_post.side_effect = [
            _challenge_response(amount="6000"),
            _ok_response(),
        ]
        tool = StallRowPageMetadataTool()

        result = json.loads(tool._run("https://example.com/"))

    assert result == METADATA
    assert mock_post.call_count == 2


@patch(POST_PATH)
def test_cap_override_via_tool_parameter(mock_post, _wallet_env):
    mock_post.return_value = _challenge_response(amount="5000")
    tool = StallRowPageMetadataTool(max_price_usdc="0.004")

    result = tool._run("https://example.com/")

    assert "cap" in result
    assert mock_post.call_count == 1


@patch(POST_PATH)
def test_upstream_error_after_payment_is_surfaced(mock_post, _wallet_env):
    mock_post.side_effect = [
        _challenge_response(),
        _FakeResponse(503, text="upstream fetch failed"),
    ]
    tool = StallRowPageMetadataTool()

    result = tool._run("https://example.com/")

    assert result.startswith("Error fetching page metadata:")
    assert "503" in result
    assert mock_post.call_count == 2


@patch(POST_PATH)
def test_second_402_after_payment_is_an_error(mock_post, _wallet_env):
    mock_post.side_effect = [_challenge_response(), _challenge_response()]
    tool = StallRowPageMetadataTool()

    result = tool._run("https://example.com/")

    assert result.startswith("Error fetching page metadata:")
    assert "402" in result
    assert mock_post.call_count == 2


@patch(POST_PATH)
def test_unpaid_200_response_is_returned_without_a_wallet(mock_post):
    mock_post.return_value = _ok_response()
    tool = StallRowPageMetadataTool()

    result = json.loads(tool._run("https://example.com/"))

    assert result == METADATA
    assert mock_post.call_count == 1
