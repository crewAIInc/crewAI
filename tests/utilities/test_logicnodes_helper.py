"""Unit tests for logicnodes_helper — no network required (web3 is mocked).

These tests exercise the real-ABI helper against a mocked deployed
LogicNodesRegistry contract (getNodeCount / nodes), covering the
node-id and liveness paths, strict mode, chain-ID enforcement, and
network / import error handling.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_w3(
    chain_id: int = 8453,
    node_count: int = 1,
    active: bool = True,
) -> MagicMock:
    """Build a minimal web3 mock matching the deployed registry ABI."""
    w3 = MagicMock()
    w3.eth.chain_id = chain_id
    # Web3.to_checksum_address is a staticmethod on the class; passthrough.
    w3.to_checksum_address = staticmethod(lambda a: a)

    contract = MagicMock()
    contract.functions.getNodeCount.return_value.call.return_value = node_count
    # nodes(bytes32) -> (name, endpoint, metadata, owner, active)
    contract.functions.nodes.return_value.call.return_value = (
        "demo-node",
        "https://logicnodes.io/call/demo",
        "{}",
        "0x" + "b" * 40,
        active,
    )
    w3.eth.contract.return_value = contract
    return w3


NODE_ID = "0x" + "a" * 64  # bytes32-looking node id


# ── liveness path (no node_id) ────────────────────────────────────────────────

def test_liveness_true_when_nodes_present():
    with patch("web3.Web3.HTTPProvider"), \
         patch("web3.Web3", return_value=_make_w3(node_count=1909)):
        from crewai.utilities.logicnodes_helper import requireLogicNodes
        assert requireLogicNodes() is True


def test_liveness_false_when_empty():
    with patch("web3.Web3.HTTPProvider"), \
         patch("web3.Web3", return_value=_make_w3(node_count=0)):
        from crewai.utilities.logicnodes_helper import requireLogicNodes
        assert requireLogicNodes() is False


def test_liveness_strict_raises_when_empty():
    with patch("web3.Web3.HTTPProvider"), \
         patch("web3.Web3", return_value=_make_w3(node_count=0)):
        from crewai.utilities.logicnodes_helper import requireLogicNodes
        with pytest.raises(RuntimeError, match="no"):
            requireLogicNodes(strict=True)


# ── node-id path ──────────────────────────────────────────────────────────────

def test_active_node_returns_true():
    with patch("web3.Web3.HTTPProvider"), \
         patch("web3.Web3", return_value=_make_w3(active=True)):
        from crewai.utilities.logicnodes_helper import requireLogicNodes
        assert requireLogicNodes(node_id=NODE_ID) is True


def test_inactive_node_returns_false():
    with patch("web3.Web3.HTTPProvider"), \
         patch("web3.Web3", return_value=_make_w3(active=False)):
        from crewai.utilities.logicnodes_helper import requireLogicNodes
        assert requireLogicNodes(node_id=NODE_ID) is False


def test_inactive_node_strict_raises():
    with patch("web3.Web3.HTTPProvider"), \
         patch("web3.Web3", return_value=_make_w3(active=False)):
        from crewai.utilities.logicnodes_helper import requireLogicNodes
        with pytest.raises(RuntimeError, match="not active"):
            requireLogicNodes(node_id=NODE_ID, strict=True)


# ── chain-id enforcement ──────────────────────────────────────────────────────

def test_wrong_chain_id_returns_false():
    with patch("web3.Web3.HTTPProvider"), \
         patch("web3.Web3", return_value=_make_w3(chain_id=1)):  # Ethereum mainnet
        from crewai.utilities.logicnodes_helper import requireLogicNodes
        assert requireLogicNodes() is False


def test_wrong_chain_id_strict_raises():
    with patch("web3.Web3.HTTPProvider"), \
         patch("web3.Web3", return_value=_make_w3(chain_id=1)):
        from crewai.utilities.logicnodes_helper import requireLogicNodes
        with pytest.raises(RuntimeError, match="chain ID"):
            requireLogicNodes(strict=True)


# ── error handling ────────────────────────────────────────────────────────────

def test_network_error_returns_false():
    w3 = MagicMock()
    w3.eth.chain_id = 8453
    w3.to_checksum_address = staticmethod(lambda a: a)
    w3.eth.contract.side_effect = ConnectionError("RPC unreachable")
    with patch("web3.Web3.HTTPProvider"), \
         patch("web3.Web3", return_value=w3):
        from crewai.utilities.logicnodes_helper import requireLogicNodes
        assert requireLogicNodes() is False


def test_network_error_strict_raises():
    w3 = MagicMock()
    w3.eth.chain_id = 8453
    w3.to_checksum_address = staticmethod(lambda a: a)
    w3.eth.contract.side_effect = ConnectionError("RPC unreachable")
    with patch("web3.Web3.HTTPProvider"), \
         patch("web3.Web3", return_value=w3):
        from crewai.utilities.logicnodes_helper import requireLogicNodes
        with pytest.raises(RuntimeError, match="LogicNodes check failed"):
            requireLogicNodes(strict=True)


def test_web3_not_installed_returns_false():
    real_import = __import__

    def mock_import(name, *args, **kwargs):
        if name == "web3":
            raise ImportError("No module named 'web3'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        from crewai.utilities.logicnodes_helper import requireLogicNodes
        assert requireLogicNodes() is False
