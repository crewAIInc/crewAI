"""CrewAI tools for Vaultfire Protocol on-chain trust verification.

Vaultfire publishes verifiable identity, partnership bonds, and reputation for
AI agents on four EVM mainnets (Base, Avalanche, Arbitrum, Polygon). These
tools wrap the upstream :pypi:`vaultfire-crewai` package so any CrewAI agent
can verify another agent's trust profile before delegating tasks.

Public package: https://pypi.org/project/vaultfire-crewai/
Source:        https://github.com/Ghostkey316/vaultfire-crewai
Standard:      ERC-8004 (AI agent identity)

All tools are read-only and require no API keys or wallet — they make direct
RPC calls against public mainnet endpoints.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field


SupportedChain = Literal["base", "avalanche", "arbitrum", "polygon"]


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


class _AddressOnChain(BaseModel):
    address: str = Field(
        ..., description="EVM address of the agent (0x-prefixed, 42 chars)."
    )
    chain: SupportedChain = Field(
        "base",
        description="Chain to query: 'base', 'avalanche', 'arbitrum', or 'polygon'.",
    )


class _DiscoverParams(BaseModel):
    capability: str = Field(
        ...,
        description=(
            "Capability to search for. Either a human-readable name (e.g. "
            "'image-generation', 'csv-analysis') or a 0x-prefixed bytes32 keccak256 hash."
        ),
    )
    chain: SupportedChain = Field("base", description="Chain to query.")


# --------------------------------------------------------------------------
# Helper
# --------------------------------------------------------------------------


_PACKAGE = "vaultfire-crewai"


def _get_client(chain: SupportedChain) -> Any:
    """Lazily construct a VaultfireClient, with a clear error if the dep is missing."""
    try:
        from vaultfire_crewai.client import VaultfireClient
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'vaultfire-crewai' package is required to use the Vaultfire tools. "
            "Install it with `uv add vaultfire-crewai` or `pip install vaultfire-crewai`."
        ) from exc
    return VaultfireClient(chain=chain)


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


class VaultfireTrustTool(BaseTool):
    """Verify an agent's full on-chain trust profile via Vaultfire.

    Returns a JSON document with on-chain registration status, agent name,
    Street Cred score and tier, active bond count, and reputation summary.
    Read-only — no API keys or wallet required.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "Vaultfire Trust Verifier"
    description: str = (
        "Verify an AI agent's on-chain trust profile (identity, bonds, reputation, "
        "Street Cred) on Vaultfire Protocol. Read-only. Pass an EVM address and "
        "optionally a chain ('base', 'avalanche', 'arbitrum', 'polygon')."
    )
    args_schema: type[BaseModel] = _AddressOnChain
    package_dependencies: list[str] = Field(default_factory=lambda: [_PACKAGE])
    env_vars: list[EnvVar] = Field(default_factory=list)

    def _run(self, address: str, chain: SupportedChain = "base", **_: Any) -> str:
        client = _get_client(chain)
        return json.dumps(client.verify_trust(address))


class VaultfireDiscoverTool(BaseTool):
    """Discover registered agents matching a capability on the on-chain registry.

    Useful before delegating a task: "find me agents on Base that registered the
    'image-generation' capability". Read-only.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "Vaultfire Trusted Agent Discovery"
    description: str = (
        "Discover AI agents registered on Vaultfire Protocol that match a given "
        "capability. Pass a human-readable capability name or a 0x-prefixed keccak256 "
        "bytes32 hash. Read-only."
    )
    args_schema: type[BaseModel] = _DiscoverParams
    package_dependencies: list[str] = Field(default_factory=lambda: [_PACKAGE])
    env_vars: list[EnvVar] = Field(default_factory=list)

    def _run(
        self,
        capability: str,
        chain: SupportedChain = "base",
        **_: Any,
    ) -> str:
        try:
            from web3 import Web3
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'web3' package is required to compute capability hashes. It is "
                "installed transitively with 'vaultfire-crewai'."
            ) from exc

        if capability.startswith("0x") and len(capability) == 66:
            cap_bytes = bytes.fromhex(capability[2:])
        else:
            cap_bytes = Web3.keccak(text=capability)

        client = _get_client(chain)
        agents = client.discover_agents_by_capability(cap_bytes)
        return json.dumps(
            {
                "capability": capability,
                "capability_hash": "0x" + cap_bytes.hex(),
                "chain": chain,
                "count": len(agents),
                "agents": agents,
            }
        )


class VaultfireBondsTool(BaseTool):
    """Read the partnership bonds (mutual economic stakes) an agent has on-chain."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "Vaultfire Bonds Reader"
    description: str = (
        "Read all partnership bonds (mutual economic stakes) for an AI agent on "
        "Vaultfire Protocol. Read-only."
    )
    args_schema: type[BaseModel] = _AddressOnChain
    package_dependencies: list[str] = Field(default_factory=lambda: [_PACKAGE])
    env_vars: list[EnvVar] = Field(default_factory=list)

    def _run(self, address: str, chain: SupportedChain = "base", **_: Any) -> str:
        client = _get_client(chain)
        return json.dumps(client.get_bonds_by_participant(address))


class VaultfireReputationTool(BaseTool):
    """Read on-chain reputation (verified ratings + average score) for an agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "Vaultfire Reputation Reader"
    description: str = (
        "Read on-chain verified reputation (ratings and average score) for an AI "
        "agent on Vaultfire Protocol. Read-only."
    )
    args_schema: type[BaseModel] = _AddressOnChain
    package_dependencies: list[str] = Field(default_factory=lambda: [_PACKAGE])
    env_vars: list[EnvVar] = Field(default_factory=list)

    def _run(self, address: str, chain: SupportedChain = "base", **_: Any) -> str:
        client = _get_client(chain)
        return json.dumps(client.get_reputation(address))
