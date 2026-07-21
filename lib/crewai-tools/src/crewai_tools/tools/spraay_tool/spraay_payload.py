"""Payload transformation for the Spraay gateway request boundary.

The tools accept user-friendly input (chain IDs, token addresses, decimal
token amounts) while the gateway expects chain slugs and integer base-unit
amount strings (verified against https://gateway.spraay.app/openapi.json
and the live /free/validate-batch validator). This module converts between
the two at the request boundary only — the tools' public input schemas are
unchanged.
"""

from decimal import Decimal, InvalidOperation


NATIVE_ADDRESS = "0x0000000000000000000000000000000000000000"

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

# Decimals for well-known tokens, keyed by lowercase contract address or
# uppercase native symbol. Amounts sent to the gateway must be integer
# strings in the token's base units (e.g. 0.01 USDC with 6 decimals ->
# "10000"). Unknown ERC-20 tokens fall back to the common default of 18.
DEFAULT_DECIMALS = 18
TOKEN_DECIMALS: dict[str, int] = {
    NATIVE_ADDRESS: 18,
    "ETH": 18,
    "WETH": 18,
    "DAI": 18,
    "USDC": 6,
    "USDT": 6,
    "EURC": 6,
    # USDC (Base / Ethereum)
    "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913": 6,
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": 6,
    # USDT (Base / Ethereum)
    "0xfde4c96c8593536e31f229ea8f37b2ada2699bb2": 6,
    "0xdac17f958d2ee523a2206206994597c13d831ec7": 6,
    # EURC (Base)
    "0x60a3e35cc302bfa44cb288bc5a4f316fdb1adb42": 6,
    # DAI (Ethereum), WETH (Base)
    "0x6b175474e89094c44da98b954eedeac495271d0f": 18,
    "0x4200000000000000000000000000000000000006": 18,
}


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


def token_decimals(token: str) -> int:
    """Return the decimals for a token address or native symbol."""
    return TOKEN_DECIMALS.get(
        token.lower(), TOKEN_DECIMALS.get(token.upper(), DEFAULT_DECIMALS)
    )


def to_base_units(amount: str, decimals: int) -> str:
    """Convert a decimal token amount string to an integer base-unit string.

    Raises:
        ValueError: If the amount is not a valid number, is negative, or has
            more fractional digits than the token's decimals.
    """
    try:
        scaled = Decimal(str(amount)).scaleb(decimals)
    except InvalidOperation as e:
        raise ValueError(f"Invalid amount {amount!r}: not a decimal number.") from e
    if scaled < 0:
        raise ValueError(f"Invalid amount {amount!r}: must not be negative.")
    if scaled != scaled.to_integral_value():
        raise ValueError(
            f"Invalid amount {amount!r}: has more than {decimals} decimal "
            "places for this token."
        )
    return str(int(scaled))
