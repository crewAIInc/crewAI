"""USDCtoFiat BaseTools for CrewAI.

USDCtoFiat by Galleon Labs. Built on the public Peer/ZKP2P protocol.
Docs: https://usdctofiat.xyz/developers

Wraps ``usdctofiat.cashout(mode="fast"|"best")``, ``watch``,
``withdraw``/``close``, ``deposits``, and ``estimate``. Mode is required
on every priced or mutating call. There is no constructor default to
Fast or Best.

These tools never accept a wallet private key. Inject a signer callback,
or call cashout without one to receive unsigned ``{to, data, value,
chainId}`` txs.

Install: ``pip install 'crewai-tools[usdctofiat]'`` (depends on
``usdctofiat>=0.1.0``).
"""

from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


_BANNED_KEY_KWARGS = (
    "private_key",
    "privateKey",
    "key",
    "secret",
    "mnemonic",
    "wallet_key",
    "evm_private_key",
    "EVM_PRIVATE_KEY",
)

_OFFRAMP_KWARGS = (
    "curator_url",
    "indexer_url",
    "curator",
    "indexer",
    "referrer",
    "referrers",
    "extra_referrers",
    "referral_code",
)


def _reject_keys_and_mode(kwargs: dict[str, Any], *, mode: str | None) -> None:
    for banned in _BANNED_KEY_KWARGS:
        if banned in kwargs:
            raise TypeError(
                "This tool does not accept a private key. "
                "Inject a signer callback or call cashout without a signer "
                "to receive unsigned txs."
            )
    if mode is not None:
        raise TypeError(
            "This tool does not default mode. "
            'Pass mode="fast" (0% / TOFIAT) or mode="best" (Delegate, 10 bps) '
            "on each cashout/estimate call."
        )


def _create_offramp(**kwargs: Any) -> Any:
    try:
        from usdctofiat import create_offramp
    except ImportError as exc:
        raise ImportError(
            "The 'usdctofiat' package is required for USDCtoFiat tools. "
            "Install it with: pip install 'crewai-tools[usdctofiat]'"
        ) from exc
    return create_offramp(**kwargs)


class UsdctoFiatCashoutSchema(BaseModel):
    """Input schema for UsdctoFiatCashoutTool. mode is required."""

    mode: str = Field(
        ...,
        description='Required. "fast" (0% / TOFIAT) or "best" (Delegate, 10 bps).',
    )
    amount: str = Field(
        ..., description="Human USDC amount. An int is six-decimal units."
    )
    currency: str = Field(..., description="Fiat ISO code, e.g. EUR, USD, GBP.")
    platform: str = Field(..., description="Payment rail, e.g. revolut, venmo, monzo.")
    payee: str = Field(..., description="Handle on that platform.")


class UsdctoFiatEstimateSchema(BaseModel):
    """Input schema for UsdctoFiatEstimateTool. mode is required."""

    mode: str = Field(..., description='Required. "fast" (0 bps) or "best" (10 bps).')
    amount: str = Field(..., description="Human USDC amount.")
    currency: str = Field(..., description="Fiat ISO code.")


class UsdctoFiatDepositSchema(BaseModel):
    """Input schema for watch / withdraw tools."""

    deposit_id: str = Field(
        ..., description="Fast composite resume key or Best numeric EscrowV2 id."
    )


class UsdctoFiatOwnerSchema(BaseModel):
    """Input schema for UsdctoFiatDepositsTool."""

    owner: str = Field(..., description="0x depositor on Base.")


class _UsdctoFiatBase(BaseTool):
    """Shared construction for USDCtoFiat tools."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    signer: Callable[[Any], Any] | None = None
    package_dependencies: list[str] = Field(default_factory=lambda: ["usdctofiat"])
    _offramp: Any = PrivateAttr(default=None)
    _offramp_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)

    def __init__(
        self,
        signer: Callable[[Any], Any] | None = None,
        mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        _reject_keys_and_mode(kwargs, mode=mode)
        offramp_kwargs = {
            key: kwargs.pop(key) for key in _OFFRAMP_KWARGS if key in kwargs
        }
        super().__init__(signer=signer, **kwargs)
        self._offramp_kwargs = offramp_kwargs
        self._offramp = None

    def _get_offramp(self) -> Any:
        if self._offramp is None:
            self._offramp = _create_offramp(**self._offramp_kwargs)
        return self._offramp


class UsdctoFiatCashoutTool(_UsdctoFiatBase):
    """Cash out Base USDC to fiat via USDCtoFiat by Galleon Labs.

    mode is required. There is no default.
    - fast: Live market pricing with 0% spread / 0 bps.
    - best: Delegate, 10 bps.

    If a signer was injected, unsigned txs are submitted and the deposit
    id / tx hash are returned. Otherwise this returns unsigned
    ``{to, data, value, chainId}`` txs for the host to sign. Never pass a
    wallet private key to this tool.
    """

    name: str = "usdctofiat_cashout"
    description: str = (
        "Cash out Base USDC to fiat via USDCtoFiat by Galleon Labs. "
        "Built on the public Peer/ZKP2P protocol. "
        'mode is required: "fast" (0% / TOFIAT) or "best" (Delegate, 10 bps). '
        "Never pass a wallet private key. https://usdctofiat.xyz/developers"
    )
    args_schema: type[BaseModel] = UsdctoFiatCashoutSchema

    def _run(
        self, mode: str, amount: str, currency: str, platform: str, payee: str
    ) -> str:
        try:
            offramp = self._get_offramp()
            if self.signer is None:
                prepared = offramp.prepare(
                    mode=mode,
                    amount=amount,
                    currency=currency,
                    platform=platform,
                    payee=payee,
                )
                return _dumps({"prepared": _as_dict(prepared), "signed": False})
            result = offramp.cashout(
                mode=mode,
                amount=amount,
                currency=currency,
                platform=platform,
                payee=payee,
                signer=self.signer,
            )
            return _dumps({"result": _as_dict(result), "signed": True})
        except Exception as exc:
            return _error(exc)


class UsdctoFiatEstimateTool(_UsdctoFiatBase):
    """Estimate a USDCtoFiat cash-out. Not a locked quote. mode is required."""

    name: str = "usdctofiat_estimate"
    description: str = (
        "Estimate a USDCtoFiat cash-out. Not a locked quote. "
        'mode is required: "fast" (0 bps) or "best" (10 bps). '
        "Docs: https://usdctofiat.xyz/developers"
    )
    args_schema: type[BaseModel] = UsdctoFiatEstimateSchema

    def _run(self, mode: str, amount: str, currency: str) -> str:
        try:
            return _dumps(
                _as_dict(
                    self._get_offramp().estimate(
                        mode=mode, amount=amount, currency=currency
                    )
                )
            )
        except Exception as exc:
            return _error(exc)


class UsdctoFiatWatchTool(_UsdctoFiatBase):
    """Watch a USDCtoFiat deposit by id (indexer snapshot)."""

    name: str = "usdctofiat_watch"
    description: str = (
        "Watch a USDCtoFiat deposit by id. Docs: https://usdctofiat.xyz/developers"
    )
    args_schema: type[BaseModel] = UsdctoFiatDepositSchema

    def _run(self, deposit_id: str) -> str:
        try:
            rows = list(self._get_offramp().watch(deposit_id))
            return _dumps({"deposit_id": deposit_id, "snapshots": rows})
        except Exception as exc:
            return _error(exc)


class UsdctoFiatWithdrawTool(_UsdctoFiatBase):
    """Withdraw / close a USDCtoFiat deposit. Alias: close."""

    name: str = "usdctofiat_withdraw"
    description: str = (
        "Withdraw or close a USDCtoFiat deposit. "
        "Docs: https://usdctofiat.xyz/developers"
    )
    args_schema: type[BaseModel] = UsdctoFiatDepositSchema

    def _run(self, deposit_id: str) -> str:
        try:
            result = self._get_offramp().withdraw(deposit_id, signer=self.signer)
            return _dumps(_as_dict(result))
        except Exception as exc:
            return _error(exc)

    def close(self, deposit_id: str) -> str:
        """Alias for withdraw."""
        return self._run(deposit_id)


class UsdctoFiatDepositsTool(_UsdctoFiatBase):
    """List USDCtoFiat deposits for an owner address."""

    name: str = "usdctofiat_deposits"
    description: str = (
        "List USDCtoFiat deposits for an owner on Base. "
        "Docs: https://usdctofiat.xyz/developers"
    )
    args_schema: type[BaseModel] = UsdctoFiatOwnerSchema

    def _run(self, owner: str) -> str:
        try:
            return _dumps(
                {"owner": owner, "deposits": self._get_offramp().deposits(owner)}
            )
        except Exception as exc:
            return _error(exc)


def _as_dict(value: Any) -> Any:
    if hasattr(value, "as_dict"):
        return value.as_dict()
    return value


def _dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, default=str)


def _error(exc: Exception) -> str:
    payload: dict[str, Any] = {
        "error": str(exc),
        "code": getattr(exc, "code", type(exc).__name__),
    }
    details = getattr(exc, "details", None)
    if details is not None:
        payload["details"] = details
    return _dumps(payload)
