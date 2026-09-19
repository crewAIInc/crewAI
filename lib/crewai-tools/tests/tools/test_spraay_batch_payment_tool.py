import json
from unittest.mock import MagicMock, patch

from crewai_tools.tools.spraay_tool.spraay_batch_payment_tool import (
    MAX_RECIPIENTS,
    SpraayBatchPaymentTool,
)
import pytest


MODULE = "crewai_tools.tools.spraay_tool.spraay_batch_payment_tool"
SENDER = "0x1111111111111111111111111111111111111111"
TOKEN = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"


def _recipients(count: int, amount: str = "1.0") -> list[dict[str, str]]:
    return [
        {"address": f"0x{i:040x}", "amount": amount} for i in range(1, count + 1)
    ]


@pytest.mark.parametrize("action", ["validate", "estimate", "execute"])
@patch(f"{MODULE}.post_with_x402")
@patch(f"{MODULE}.token_decimals")
@patch(f"{MODULE}.requests.get")
@patch(f"{MODULE}.requests.post")
def test_rejects_more_than_max_recipients_without_http(
    mock_post, mock_get, mock_decimals, mock_x402, action
):
    result = SpraayBatchPaymentTool()._run(
        action=action,
        token_address=TOKEN,
        recipients=_recipients(MAX_RECIPIENTS + 1),
        sender_address=SENDER,
    )

    assert result == "Error: 'recipients' list must contain at most 200 entries."
    mock_post.assert_not_called()
    mock_get.assert_not_called()
    mock_decimals.assert_not_called()
    mock_x402.assert_not_called()


@patch(f"{MODULE}.token_decimals")
@patch(f"{MODULE}.requests.get")
def test_estimate_skips_amount_conversion(mock_get, mock_decimals):
    response = MagicMock()
    response.json.return_value = {"gas": "123"}
    mock_get.return_value = response

    result = SpraayBatchPaymentTool()._run(
        action="estimate",
        token_address=TOKEN,
        recipients=_recipients(3, amount="not-a-number"),
        sender_address=SENDER,
    )

    mock_decimals.assert_not_called()
    mock_get.assert_called_once()
    assert mock_get.call_args.kwargs["params"] == {"recipients": 3, "chain": "base"}
    assert json.loads(result) == {
        "status": "estimated",
        "recipientCount": 3,
        "estimate": {"gas": "123"},
    }
