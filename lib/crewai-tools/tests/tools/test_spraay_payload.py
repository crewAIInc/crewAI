from decimal import Overflow

from crewai_tools.tools.spraay_tool.spraay_payload import (
    BASE_CHAIN_ID,
    NATIVE_ADDRESS,
    NATIVE_DECIMALS,
    to_base_units,
    token_decimals,
)
import pytest


NON_BASE_CHAIN_IDS = [1, 56, 137, 43114]


@pytest.mark.parametrize("symbol", ["ETH", "eth"])
def test_eth_symbol_resolves_on_base(symbol):
    assert token_decimals(symbol, BASE_CHAIN_ID) == 18


@pytest.mark.parametrize("chain_id", NON_BASE_CHAIN_IDS)
@pytest.mark.parametrize("symbol", ["ETH", "eth"])
def test_eth_symbol_rejected_on_non_base_chains(symbol, chain_id):
    with pytest.raises(ValueError, match="only supported on Base"):
        token_decimals(symbol, chain_id)


@pytest.mark.parametrize("chain_id", [BASE_CHAIN_ID, *NON_BASE_CHAIN_IDS])
def test_native_address_resolves_on_every_chain(chain_id):
    assert token_decimals(NATIVE_ADDRESS, chain_id) == NATIVE_DECIMALS


def test_to_base_units_scales_amount():
    assert to_base_units("1.5", 6) == "1500000"


@pytest.mark.parametrize("amount", ["1E999999", "9.99E999990"])
def test_to_base_units_maps_decimal_overflow_to_value_error(amount):
    with pytest.raises(ValueError, match="out of range") as exc_info:
        to_base_units(amount, 18)
    assert isinstance(exc_info.value.__cause__, Overflow)


@pytest.mark.parametrize("amount", ["inf", "-Infinity", "NaN"])
def test_to_base_units_rejects_non_finite(amount):
    with pytest.raises(ValueError, match="finite"):
        to_base_units(amount, 18)
