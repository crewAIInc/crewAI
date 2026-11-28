from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from crewai_tools.tools.sochain_tool.base import (
    NETWORK_DESCRIPTION,
    Network,
    SoChainBaseTool,
)


ADDRESS_PATTERN = r"^[A-Za-z0-9]{1,1000}$"
HASH_PATTERN = r"^[0-9a-fA-F]{64}$"
BLOCK_PATTERN = r"^([0-9]{1,8}|[0-9a-fA-F]{64})$"


class SoChainAddressSchema(BaseModel):
    network: Network = Field(..., description=NETWORK_DESCRIPTION)
    address: str = Field(
        ..., description="The address to look up.", pattern=ADDRESS_PATTERN
    )


class SoChainAddressTransactionsSchema(SoChainAddressSchema):
    page: int | None = Field(
        default=1,
        ge=1,
        le=999999,
        description="Page number from 1; ten transactions per page, newest first.",
    )


class SoChainTransactionSchema(BaseModel):
    network: Network = Field(..., description=NETWORK_DESCRIPTION)
    tx_hash: str = Field(
        ..., description="Transaction ID (64 hex characters).", pattern=HASH_PATTERN
    )


class SoChainBlockSchema(BaseModel):
    network: Network = Field(..., description=NETWORK_DESCRIPTION)
    block: str = Field(
        ...,
        description="Block height or block hash (64 hex characters).",
        pattern=BLOCK_PATTERN,
    )
    include_transactions: bool = Field(
        default=False,
        description="Also return every transaction ID in the block (can be long).",
    )


class SoChainNetworkSchema(BaseModel):
    network: Network = Field(..., description=NETWORK_DESCRIPTION)


class SoChainAddressBalanceTool(SoChainBaseTool):
    """Confirmed and unconfirmed balance of an address."""

    name: str = "SoChain Address Balance"
    description: str = (
        "Get the confirmed balance and the pending (unconfirmed) change of a "
        "Bitcoin, Litecoin, Dogecoin or Zcash address. Amounts are decimal "
        "strings in coin units."
    )
    args_schema: type[BaseModel] = SoChainAddressSchema

    def _run(self, network: str, address: str, **_: Any) -> str:
        return self._get(f"balance/{network.upper()}/{address}")


class SoChainAddressSummaryTool(SoChainBaseTool):
    """Balances, total received and transaction counts of an address."""

    name: str = "SoChain Address Summary"
    description: str = (
        "Summarize a Bitcoin, Litecoin, Dogecoin or Zcash address: confirmed "
        "and unconfirmed balance, total received, transactions sent and "
        "received, and when it was first and last active."
    )
    args_schema: type[BaseModel] = SoChainAddressSchema

    def _run(self, network: str, address: str, **_: Any) -> str:
        return self._get(f"address_summary/{network.upper()}/{address}")


class SoChainAddressTransactionsTool(SoChainBaseTool):
    """An address's transactions, ten per page."""

    name: str = "SoChain Address Transactions"
    description: str = (
        "List the transactions of a Bitcoin, Litecoin, Dogecoin or Zcash "
        "address, ten per page, unconfirmed first and then newest first, with "
        "the amount sent, received and the balance change of each."
    )
    args_schema: type[BaseModel] = SoChainAddressTransactionsSchema

    def _run(self, network: str, address: str, page: int | None = 1, **_: Any) -> str:
        return self._get(f"transactions/{network.upper()}/{address}/{page or 1}")


class SoChainTransactionTool(SoChainBaseTool):
    """One transaction with its inputs and outputs."""

    name: str = "SoChain Transaction"
    description: str = (
        "Get a Bitcoin, Litecoin, Dogecoin or Zcash transaction by its ID: "
        "confirmations, block, fee, size, and every input and output."
    )
    args_schema: type[BaseModel] = SoChainTransactionSchema

    def _run(self, network: str, tx_hash: str, **_: Any) -> str:
        return self._get(f"transaction/{network.upper()}/{tx_hash}")


class SoChainBlockTool(SoChainBaseTool):
    """One block by height or hash."""

    name: str = "SoChain Block"
    description: str = (
        "Get a Bitcoin, Litecoin, Dogecoin or Zcash block by height or hash: "
        "time, confirmations, transaction count, reward, fees, fee rates and "
        "value out."
    )
    args_schema: type[BaseModel] = SoChainBlockSchema

    def _run(
        self,
        network: str,
        block: str,
        include_transactions: bool | None = False,
        **_: Any,
    ) -> str:
        data, error = self._get_data(f"block/{network.upper()}/{block}")
        if error is not None:
            return error
        if not include_transactions and isinstance(data, dict):
            data.pop("txs", None)
        return json.dumps(data, ensure_ascii=False)


class SoChainNetworkInfoTool(SoChainBaseTool):
    """Network state, mempool and fee rates."""

    name: str = "SoChain Network Info"
    description: str = (
        "Get the state of the Bitcoin, Litecoin, Dogecoin or Zcash network: "
        "newest block, hashrate, mempool size and the next projected blocks "
        "with their fee rates, to estimate the fee a transaction needs."
    )
    args_schema: type[BaseModel] = SoChainNetworkSchema

    def _run(self, network: str, **_: Any) -> str:
        return self._get(f"network_info/{network.upper()}")
