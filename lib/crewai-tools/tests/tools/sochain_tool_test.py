import json
import os
from unittest.mock import MagicMock, patch

from crewai_tools.tools.sochain_tool import (
    SoChainAddressBalanceTool,
    SoChainAddressSummaryTool,
    SoChainAddressTransactionsTool,
    SoChainBlockTool,
    SoChainNetworkInfoTool,
    SoChainTransactionTool,
)
from pydantic import ValidationError
import pytest
import requests


DOGE_ADDRESS = "DGUzATWpGdehtujWNyM464LB9Zx7Ch1Z67"
TX_HASH = "484c94145e97c0b2dd96912c1f056c9e05c8d358ab96ea429c51039d75d0d3c4"
API = "https://sochain.com/api/v3"


def _response(
    status_code: int = 200,
    json_data: object | None = None,
    headers: dict[str, str] | None = None,
) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.text = json.dumps(json_data) if json_data is not None else ""
    if json_data is None:
        resp.json.side_effect = ValueError("no JSON")
    else:
        resp.json.return_value = json_data
    return resp


def _success(data: object) -> MagicMock:
    return _response(json_data={"status": "success", "data": data})


@pytest.fixture(autouse=True)
def _no_key_in_env():
    env = {k: v for k, v in os.environ.items() if k != "SOCHAIN_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        yield


@pytest.fixture
def mock_get():
    with patch("crewai_tools.tools.sochain_tool.base.requests.get") as get:
        yield get


def test_balance_keyless(mock_get):
    mock_get.return_value = _success(
        {"confirmed": "0.00059184", "unconfirmed": "0.00000000"}
    )

    result = SoChainAddressBalanceTool().run(network="BTC", address="bc1qtest0")

    assert json.loads(result) == {
        "confirmed": "0.00059184",
        "unconfirmed": "0.00000000",
    }
    url = mock_get.call_args.args[0]
    assert url == f"{API}/balance/BTC/bc1qtest0"
    assert "API-KEY" not in mock_get.call_args.kwargs["headers"]


def test_api_key_from_env(mock_get):
    mock_get.return_value = _success({"confirmed": "1.0", "unconfirmed": "0"})

    with patch.dict(os.environ, {"SOCHAIN_API_KEY": "env-key"}):
        tool = SoChainAddressBalanceTool()
    tool.run(network="LTC", address="ltc1qtest0")

    assert mock_get.call_args.kwargs["headers"]["API-KEY"] == "env-key"


def test_api_key_argument_wins(mock_get):
    mock_get.return_value = _success({"confirmed": "1.0", "unconfirmed": "0"})

    with patch.dict(os.environ, {"SOCHAIN_API_KEY": "env-key"}):
        tool = SoChainAddressBalanceTool(api_key="arg-key")
    tool.run(network="LTC", address="ltc1qtest0")

    assert mock_get.call_args.kwargs["headers"]["API-KEY"] == "arg-key"


def test_address_summary(mock_get):
    data = {
        "txs_sent": 389,
        "txs_received": 512,
        "confirmed_balance": "2744364.03886668",
        "unconfirmed_balance": "0.00000000",
    }
    mock_get.return_value = _success(data)

    result = SoChainAddressSummaryTool().run(network="DOGE", address=DOGE_ADDRESS)

    assert json.loads(result) == data
    assert mock_get.call_args.args[0] == f"{API}/address_summary/DOGE/{DOGE_ADDRESS}"


def test_address_transactions_page(mock_get):
    data = {"transactions": [{"hash": TX_HASH, "balance_change": "9.51727163"}]}
    mock_get.return_value = _success(data)

    result = SoChainAddressTransactionsTool().run(
        network="DOGE", address=DOGE_ADDRESS, page=2
    )

    assert json.loads(result) == data
    assert mock_get.call_args.args[0] == f"{API}/transactions/DOGE/{DOGE_ADDRESS}/2"


def test_address_transactions_default_page(mock_get):
    mock_get.return_value = _success({"transactions": []})

    SoChainAddressTransactionsTool().run(network="DOGE", address=DOGE_ADDRESS)

    assert mock_get.call_args.args[0] == f"{API}/transactions/DOGE/{DOGE_ADDRESS}/1"


def test_transaction(mock_get):
    data = {"hash": TX_HASH, "fee": "0.45600000", "inputs": [], "outputs": []}
    mock_get.return_value = _success(data)

    result = SoChainTransactionTool().run(network="DOGE", tx_hash=TX_HASH)

    assert json.loads(result)["fee"] == "0.45600000"
    assert mock_get.call_args.args[0] == f"{API}/transaction/DOGE/{TX_HASH}"


def test_block_omits_transaction_ids_by_default(mock_get):
    mock_get.return_value = _success(
        {"height": 5000000, "num_txs": 2, "txs": ["a" * 64, "b" * 64]}
    )

    result = json.loads(SoChainBlockTool().run(network="DOGE", block="5000000"))

    assert result == {"height": 5000000, "num_txs": 2}
    assert mock_get.call_args.args[0] == f"{API}/block/DOGE/5000000"


def test_block_with_transaction_ids(mock_get):
    mock_get.return_value = _success({"height": 1, "txs": ["a" * 64]})

    result = json.loads(
        SoChainBlockTool().run(
            network="BTC", block="1", include_transactions=True
        )
    )

    assert result["txs"] == ["a" * 64]


def test_network_info_testnet(mock_get):
    data = {"block_count": 100, "mempool": {"blocks": [{"median_fee_rate": 2}]}}
    mock_get.return_value = _success(data)

    result = SoChainNetworkInfoTool().run(network="BTCTEST")

    assert json.loads(result) == data
    assert mock_get.call_args.args[0] == f"{API}/network_info/BTCTEST"


def test_fail_status_returns_error_message(mock_get):
    mock_get.return_value = _response(
        404,
        {
            "status": "fail",
            "data": {"error_message": "Unable to fetch address balance. Invalid address?"},
        },
    )

    result = SoChainAddressBalanceTool().run(network="BTC", address="nope")

    assert result == (
        "SoChain API error (HTTP 404): "
        "Unable to fetch address balance. Invalid address?"
    )


def test_bad_key(mock_get):
    mock_get.return_value = _response(
        401, {"status": "fail", "data": {"error_message": "bad key"}}
    )

    result = SoChainNetworkInfoTool(api_key="wrong").run(network="BTC")

    assert "HTTP 401" in result
    assert "https://sochain.com/dashboard" in result


@patch("crewai_tools.tools.sochain_tool.base.time.sleep")
def test_rate_limit_short_retry_after_retries_once(mock_sleep, mock_get):
    mock_get.side_effect = [
        _response(429, {"status": "fail", "data": {}}, {"Retry-After": "2"}),
        _success({"block_count": 1}),
    ]

    result = SoChainNetworkInfoTool().run(network="LTC")

    assert json.loads(result) == {"block_count": 1}
    mock_sleep.assert_called_once_with(2.0)
    assert mock_get.call_count == 2


@patch("crewai_tools.tools.sochain_tool.base.time.sleep")
def test_rate_limit_long_retry_after_is_reported(mock_sleep, mock_get):
    mock_get.return_value = _response(
        429, {"status": "fail", "data": {}}, {"Retry-After": "3600"}
    )

    result = SoChainNetworkInfoTool().run(network="LTC")

    assert "HTTP 429" in result
    assert "Retry after 3600 seconds" in result
    mock_sleep.assert_not_called()
    assert mock_get.call_count == 1


def test_timeout(mock_get):
    mock_get.side_effect = requests.Timeout()

    result = SoChainNetworkInfoTool().run(network="ZEC")

    assert result == "SoChain API error: the request timed out."


def test_non_json_response(mock_get):
    resp = _response(502)
    resp.text = "Bad gateway"
    mock_get.return_value = resp

    result = SoChainNetworkInfoTool().run(network="ZEC")

    assert result == "SoChain API error (HTTP 502): Bad gateway"


@pytest.mark.parametrize(
    "tool, kwargs",
    [
        (SoChainNetworkInfoTool, {"network": "ETH"}),
        (SoChainAddressBalanceTool, {"network": "BTC", "address": "../account_info"}),
        (SoChainTransactionTool, {"network": "BTC", "tx_hash": "abc"}),
        (SoChainBlockTool, {"network": "BTC", "block": "latest"}),
    ],
)
def test_invalid_arguments_make_no_request(tool, kwargs, mock_get):
    with pytest.raises(ValidationError):
        tool().args_schema(**kwargs)
    mock_get.assert_not_called()


def test_env_var_is_optional():
    (env_var,) = SoChainNetworkInfoTool().env_vars

    assert env_var.name == "SOCHAIN_API_KEY"
    assert env_var.required is False


def test_address_transactions_null_page_means_first(mock_get):
    mock_get.return_value = _success({"transactions": []})

    SoChainAddressTransactionsTool().run(
        network="DOGE", address=DOGE_ADDRESS, page=None
    )

    assert mock_get.call_args.args[0] == f"{API}/transactions/DOGE/{DOGE_ADDRESS}/1"
