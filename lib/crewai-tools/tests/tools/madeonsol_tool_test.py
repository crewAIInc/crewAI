import json
import sys
from unittest.mock import MagicMock, patch

import pytest

# madeonsol-x402 is an optional dependency — mock the module so the tools are
# importable and testable without it installed (same pattern as couchbase).
mock_madeonsol = MagicMock()
sys.modules.setdefault("madeonsol_x402", mock_madeonsol)

from crewai_tools.tools.madeonsol_tool.madeonsol_tool import (  # noqa: E402
    MadeOnSolDeployerAlertsTool,
    MadeOnSolKolCoordinationTool,
    MadeOnSolKolFeedTool,
    MadeOnSolKolLeaderboardTool,
    MadeOnSolTokenRiskTool,
)


@pytest.fixture
def mock_client():
    client = MagicMock()
    with patch(
        "crewai_tools.tools.madeonsol_tool.madeonsol_tool._make_client",
        return_value=client,
    ):
        yield client


def test_kol_feed_tool(mock_client):
    mock_client.kol_feed.return_value = {"trades": [{"mint": "abc", "action": "buy"}]}
    result = MadeOnSolKolFeedTool().run(limit=5, action="buy")
    mock_client.kol_feed.assert_called_once_with(limit=5, action="buy")
    assert json.loads(result) == {"trades": [{"mint": "abc", "action": "buy"}]}


def test_kol_leaderboard_tool(mock_client):
    mock_client.kol_leaderboard.return_value = {"leaderboard": []}
    result = MadeOnSolKolLeaderboardTool().run(period="30d", limit=20)
    mock_client.kol_leaderboard.assert_called_once_with(period="30d", limit=20)
    assert json.loads(result) == {"leaderboard": []}


def test_kol_coordination_tool(mock_client):
    mock_client.kol_coordination.return_value = {"signals": []}
    result = MadeOnSolKolCoordinationTool().run(period="6h", min_kols=4)
    mock_client.kol_coordination.assert_called_once_with(period="6h", min_kols=4)
    assert json.loads(result) == {"signals": []}


def test_deployer_alerts_tool(mock_client):
    mock_client.deployer_alerts.return_value = {"alerts": []}
    result = MadeOnSolDeployerAlertsTool().run(limit=10, tier="elite")
    mock_client.deployer_alerts.assert_called_once_with(limit=10, tier="elite")
    assert json.loads(result) == {"alerts": []}


def test_token_risk_tool(mock_client):
    mint = "So11111111111111111111111111111111111111112"
    mock_client.token_risk.return_value = {"risk_score": 12}
    result = MadeOnSolTokenRiskTool().run(mint=mint)
    mock_client.token_risk.assert_called_once_with(mint)
    assert json.loads(result) == {"risk_score": 12}


def test_missing_credentials_raises(monkeypatch):
    from crewai_tools.tools.madeonsol_tool import madeonsol_tool as module

    monkeypatch.delenv("MADEONSOL_API_KEY", raising=False)
    monkeypatch.delenv("SVM_PRIVATE_KEY", raising=False)
    with pytest.raises(ValueError, match="MADEONSOL_API_KEY"):
        module._make_client()
