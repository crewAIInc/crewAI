"""
LogicNodes Integration Helper
=============================
Lightweight, dependency-light helpers to verify that an autonomous system
(AI agent, service, or worker) is registered on the on-chain LogicNodes
registry before relying on it.

Registry: 0x4c60B817beeD72aa570B964243eE6DD463faaE22 (Base mainnet, chain 8453)
Contract: LogicNodesRegistry
Docs: https://logicnodes.io

The registry exposes:
    registerNode(string,string,string)        — register a node
    getNodeCount() -> uint256                  — number of registered nodes
    nodeIds(uint256) -> bytes32                — enumerate node ids
    nodes(bytes32) -> (string,string,string,address,bool)
                                               — (name, endpoint, metadata, owner, active)

Install:
    pip install web3>=6.0

Usage:
    from crewai.utilities.logicnodes_helper import (
        get_node_count, get_node, is_node_active, requireLogicNodes,
    )

    # liveness / size of the registry
    print(get_node_count())

    # gate a workflow on a specific node being active
    requireLogicNodes(node_id="0x...", strict=True)
"""

from __future__ import annotations

from typing import Optional, Tuple

LOGICNODES_REGISTRY = "0x4c60B817beeD72aa570B964243eE6DD463faaE22"
BASE_MAINNET_RPC = "https://mainnet.base.org"
CHAIN_ID = 8453


class _GateError(RuntimeError):
    """Raised when a strict LogicNodes gate intentionally fails (node not
    active / registry empty). Distinct from infrastructure errors (wrong
    chain, RPC failure) so non-strict callers swallow infra errors but
    strict callers still see gate failures."""

# Minimal ABI for the read paths we use (matches the deployed contract exactly).
_REGISTRY_ABI = [
    {
        "inputs": [],
        "name": "getNodeCount",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "name": "nodeIds",
        "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
        "name": "nodes",
        "outputs": [
            {"internalType": "string", "name": "name", "type": "string"},
            {"internalType": "string", "name": "endpoint", "type": "string"},
            {"internalType": "string", "name": "metadata", "type": "string"},
            {"internalType": "address", "name": "owner", "type": "address"},
            {"internalType": "bool", "name": "active", "type": "bool"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
]


def _contract(rpc_url: str, registry: str):
    from web3 import Web3  # noqa: PLC0415

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    chain_id = w3.eth.chain_id
    if chain_id != CHAIN_ID:
        raise RuntimeError(
            f"Unexpected chain ID {chain_id}; expected {CHAIN_ID} (Base mainnet)."
        )
    return w3.eth.contract(
        address=Web3.to_checksum_address(registry),
        abi=_REGISTRY_ABI,
    )


def get_node_count(
    rpc_url: str = BASE_MAINNET_RPC,
    registry: str = LOGICNODES_REGISTRY,
) -> int:
    """Return the number of nodes registered on the LogicNodes registry.

    A return value of 0 means the registry is reachable but currently empty.
    Raises on network / RPC errors so callers can distinguish "empty" from
    "unreachable".
    """
    contract = _contract(rpc_url, registry)
    return int(contract.functions.getNodeCount().call())


def get_node(
    node_id: str,
    rpc_url: str = BASE_MAINNET_RPC,
    registry: str = LOGICNODES_REGISTRY,
) -> Tuple[str, str, str, str, bool]:
    """Look up a node by its bytes32 id.

    Returns (name, endpoint, metadata, owner, active).
    """
    contract = _contract(rpc_url, registry)
    if isinstance(node_id, str) and node_id.startswith("0x"):
        node_id_bytes = bytes.fromhex(node_id[2:].rjust(64, "0"))
    else:
        node_id_bytes = node_id
    name, endpoint, metadata, owner, active = contract.functions.nodes(node_id_bytes).call()
    return name, endpoint, metadata, owner, bool(active)


def is_node_active(
    node_id: str,
    rpc_url: str = BASE_MAINNET_RPC,
    registry: str = LOGICNODES_REGISTRY,
) -> bool:
    """Return True iff the node id exists and its `active` flag is set."""
    try:
        _name, _endpoint, _metadata, _owner, active = get_node(node_id, rpc_url, registry)
        return bool(active)
    except Exception:
        return False


def requireLogicNodes(
    node_id: Optional[str] = None,
    rpc_url: str = BASE_MAINNET_RPC,
    registry: str = LOGICNODES_REGISTRY,
    strict: bool = False,
) -> bool:
    """Guard a workflow on the LogicNodes registry.

    - If ``node_id`` is given: returns True iff that node is registered AND active.
    - If ``node_id`` is omitted: returns True iff the registry is reachable and
      contains at least one registered node (a liveness check).

    Args:
        node_id: Optional bytes32 node id to verify.
        rpc_url: Base mainnet RPC endpoint.
        registry: LogicNodes registry contract address.
        strict:  If True, raise RuntimeError on failure instead of returning False.

    Returns:
        True if the check passes, False otherwise (unless strict=True).

    Raises:
        RuntimeError: Only when strict=True and the check fails.

    Example::

        if not requireLogicNodes(node_id="0xabc..."):
            print("Node not active on LogicNodes — proceeding with caution.")
    """
    try:
        if node_id is not None:
            ok = is_node_active(node_id, rpc_url, registry)
            if not ok and strict:
                raise _GateError(
                    f"Node {node_id} is not active on LogicNodes registry "
                    f"{registry}. Register at https://logicnodes.io"
                )
            return ok
        # liveness check
        count = get_node_count(rpc_url, registry)
        ok = count > 0
        if not ok and strict:
            raise _GateError(
                "LogicNodes registry is reachable but currently has no "
                "registered nodes. See https://logicnodes.io"
            )
        return ok
    except _GateError:
        raise
    except Exception as exc:  # network / import errors
        if strict:
            raise RuntimeError(f"LogicNodes check failed: {exc}") from exc
        return False
