"""Payload transformation for the Spraay gateway request boundary.

The tools accept user-friendly input (chain IDs, token addresses, decimal
token amounts) while the gateway expects chain slugs and integer base-unit
amount strings (verified against https://gateway.spraay.app/openapi.json
and the live /free/validate-batch validator). This module converts between
the two at the request boundary only — the tools' public input schemas are
unchanged.

Token decimals are resolved per (chain, token) — never guessed: a wrong
decimals value silently mis-scales every amount by a power of ten (e.g.
USDC on Polygon uses 6 decimals, not the ERC-20-common 18). Resolution
order: native token -> static known-token table -> cache -> gateway token
directory (Base) -> the token contract's decimals() via public RPC. If all
of these fail, a ValueError is raised instead of falling back to a default.
"""

from decimal import Decimal, InvalidOperation
import re

import requests


NATIVE_ADDRESS = "0x0000000000000000000000000000000000000000"

# The native coin on every supported chain (ETH, BNB, POL, AVAX, XPL) uses
# 18 decimals.
NATIVE_DECIMALS = 18

# EVM chain IDs mapped to the gateway's chain slugs (the slug set the
# /free/validate-batch endpoint reports as supported).
CHAIN_SLUGS: dict[int, str] = {
    1: "ethereum",
    56: "bnb",
    130: "unichain",
    137: "polygon",
    8453: "base",
    9745: "plasma",
    42161: "arbitrum",
    43114: "avalanche",
    60808: "bob",
}

BASE_CHAIN_ID = 8453

# The gateway's free token directory (Base tokens with addresses, symbols,
# and decimals). The gateway publishes no per-chain decimals endpoint for
# the other chains, so those resolve on-chain via RPC below.
GATEWAY_TOKENS_URL = "https://gateway.spraay.app/api/v1/tokens"

# Public JSON-RPC endpoints used to resolve an ERC-20 token's decimals()
# on-chain when the token is not in the static table or gateway directory.
RPC_URLS: dict[int, str] = {
    1: "https://ethereum-rpc.publicnode.com",
    56: "https://bsc-rpc.publicnode.com",
    130: "https://unichain-rpc.publicnode.com",
    137: "https://polygon-bor-rpc.publicnode.com",
    8453: "https://base-rpc.publicnode.com",
    9745: "https://rpc.plasma.to",
    42161: "https://arbitrum-one-rpc.publicnode.com",
    43114: "https://avalanche-c-chain-rpc.publicnode.com",
    60808: "https://rpc.gobob.xyz",
}

# keccak256("decimals()")[:4]
_DECIMALS_SELECTOR = "0x313ce567"

# Decimals for well-known tokens, keyed by (chain_id, lowercase contract
# address). The Base entries mirror the gateway token directory.
KNOWN_TOKEN_DECIMALS: dict[tuple[int, str], int] = {
    # Base
    (8453, "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"): 6,  # USDC
    (8453, "0xfde4c96c8593536e31f229ea8f37b2ada2699bb2"): 6,  # USDT
    (8453, "0x50c5725949a6f0c72e6c4a641f24049a917db0cb"): 18,  # DAI
    (8453, "0x60a3e35cc302bfa44cb288bc5a4f316fdb1adb42"): 6,  # EURC
    (8453, "0x4200000000000000000000000000000000000006"): 18,  # WETH
    # Ethereum
    (1, "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"): 6,  # USDC
    (1, "0xdac17f958d2ee523a2206206994597c13d831ec7"): 6,  # USDT
    (1, "0x6b175474e89094c44da98b954eedeac495271d0f"): 18,  # DAI
}

# Symbol shortcuts from the gateway token directory — valid on Base only.
# The same symbol can use different decimals elsewhere (e.g. Binance-peg
# USDT on BNB Chain uses 18, not 6), so symbols never resolve cross-chain.
BASE_SYMBOL_DECIMALS: dict[str, int] = {
    "ETH": 18,
    "WETH": 18,
    "DAI": 18,
    "USDC": 6,
    "USDT": 6,
    "EURC": 6,
}

_ADDRESS_RE = re.compile(r"0x[0-9a-fA-F]{40}\Z")

_decimals_cache: dict[tuple[int, str], int] = {}
_gateway_directory_cache: dict[str, int] | None = None


def chain_slug(chain_id: int) -> str:
    """Map an EVM chain ID to the gateway's chain slug.

    Raises:
        ValueError: If the chain ID is not a supported gateway chain.
    """
    slug = CHAIN_SLUGS.get(chain_id)
    if slug is None:
        supported = ", ".join(
            f"{cid} ({name})" for cid, name in sorted(CHAIN_SLUGS.items())
        )
        raise ValueError(
            f"Unsupported chain_id {chain_id}. Supported chain IDs: {supported}."
        )
    return slug


def _gateway_token_directory() -> dict[str, int]:
    """Fetch the gateway's Base token directory, mapping lowercase contract
    addresses to decimals. Cached after the first successful fetch; a failed
    fetch returns an empty mapping without poisoning the cache.
    """
    global _gateway_directory_cache
    if _gateway_directory_cache is None:
        directory: dict[str, int] = {}
        try:
            response = requests.get(GATEWAY_TOKENS_URL, timeout=10)
            response.raise_for_status()
            tokens = response.json().get("popularTokens", {})
            for info in tokens.values():
                address = info.get("address")
                decimals = info.get("decimals")
                if isinstance(address, str) and isinstance(decimals, int):
                    directory[address.lower()] = decimals
        except (requests.RequestException, ValueError, AttributeError):
            return {}
        _gateway_directory_cache = directory
    return _gateway_directory_cache


def _rpc_decimals(chain_id: int, token: str) -> int | None:
    """Read decimals() from the token contract via the chain's public RPC.

    Returns None when the value cannot be read (unreachable RPC, address is
    not a contract, contract without decimals(), or an out-of-range result).
    """
    url = RPC_URLS.get(chain_id)
    if url is None:
        return None
    try:
        response = requests.post(
            url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_call",
                "params": [{"to": token, "data": _DECIMALS_SELECTOR}, "latest"],
            },
            timeout=10,
        )
        response.raise_for_status()
        result = response.json().get("result")
    except (requests.RequestException, ValueError, AttributeError):
        return None
    if not isinstance(result, str) or not result.startswith("0x") or len(result) <= 2:
        return None
    try:
        decimals = int(result, 16)
    except ValueError:
        return None
    if not 0 <= decimals <= 255:
        return None
    return decimals


def token_decimals(token: str, chain_id: int) -> int:
    """Resolve the decimals for a token on a specific chain.

    Raises:
        ValueError: If the token's decimals cannot be resolved. Decimals
            are never defaulted for unknown tokens — a wrong value would
            silently mis-scale every amount by a power of ten.
    """
    if token == NATIVE_ADDRESS or token.upper() == "ETH":
        return NATIVE_DECIMALS

    if not _ADDRESS_RE.fullmatch(token):
        symbol = token.upper()
        if chain_id == BASE_CHAIN_ID and symbol in BASE_SYMBOL_DECIMALS:
            return BASE_SYMBOL_DECIMALS[symbol]
        raise ValueError(
            f"Unknown token symbol {token!r} on chain {chain_id}. Symbol "
            "shortcuts are only supported on Base (8453); pass the token's "
            "contract address (0x...) instead."
        )

    key = (chain_id, token.lower())
    known = KNOWN_TOKEN_DECIMALS.get(key)
    if known is not None:
        return known
    cached = _decimals_cache.get(key)
    if cached is not None:
        return cached

    decimals = None
    if chain_id == BASE_CHAIN_ID:
        decimals = _gateway_token_directory().get(key[1])
    if decimals is None:
        decimals = _rpc_decimals(chain_id, token)
    if decimals is None:
        raise ValueError(
            f"Could not resolve decimals for token {token} on chain "
            f"{chain_id}: the gateway token directory and the on-chain "
            "decimals() lookup both failed. Refusing to guess, since a "
            "wrong decimals value would mis-scale every amount. Verify the "
            "token contract address, or retry when the chain's RPC "
            "endpoint is reachable."
        )
    _decimals_cache[key] = decimals
    return decimals


def to_base_units(amount: str, decimals: int) -> str:
    """Convert a decimal token amount string to an integer base-unit string.

    Raises:
        ValueError: If the amount is not a finite number, is negative, or
            has more fractional digits than the token's decimals.
    """
    try:
        value = Decimal(str(amount))
    except InvalidOperation as e:
        raise ValueError(f"Invalid amount {amount!r}: not a decimal number.") from e
    if not value.is_finite():
        raise ValueError(f"Invalid amount {amount!r}: must be a finite number.")
    try:
        scaled = value.scaleb(decimals)
    except InvalidOperation as e:
        raise ValueError(f"Invalid amount {amount!r}: out of range.") from e
    if scaled < 0:
        raise ValueError(f"Invalid amount {amount!r}: must not be negative.")
    if scaled != scaled.to_integral_value():
        raise ValueError(
            f"Invalid amount {amount!r}: has more than {decimals} decimal "
            "places for this token."
        )
    return str(int(scaled))
