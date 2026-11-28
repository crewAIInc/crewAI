from crewai.utilities.string_utils import sanitize_tool_name
import pytest


@pytest.mark.parametrize("limit", range(1, 10))
def test_sanitize_tool_name_respects_short_limits(limit: int) -> None:
    result = sanitize_tool_name("fetch_customer_accounts", max_length=limit)

    assert 0 < len(result) <= limit
    assert result == sanitize_tool_name("fetch_customer_accounts", max_length=limit)


@pytest.mark.parametrize("limit", [0, -1, -8])
def test_sanitize_tool_name_rejects_non_positive_limits(limit: int) -> None:
    with pytest.raises(ValueError, match="max_length must be positive"):
        sanitize_tool_name("fetch", max_length=limit)


@pytest.mark.parametrize("limit", [9, 10, 64])
def test_sanitize_tool_name_preserves_existing_hash_suffix(limit: int) -> None:
    result = sanitize_tool_name("fetch_customer_accounts", max_length=limit)

    if limit == 64:
        assert result == "fetch_customer_accounts"
    elif limit == 10:
        assert result == "f_0453184f"
    else:
        assert result == "_0453184f"


def test_sanitize_tool_name_keeps_a_name_that_fits_short_limit() -> None:
    assert sanitize_tool_name("Get", max_length=3) == "get"
