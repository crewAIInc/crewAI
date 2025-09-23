"""Tests for ChromaDB utility functions."""

from crewai.rag.chromadb.utils import (
    MAX_COLLECTION_LENGTH,
    MIN_COLLECTION_LENGTH,
    _is_ipv4_pattern,
    _sanitize_collection_name,
)


class TestChromaDBUtils:
    """Test suite for ChromaDB utility functions."""

    def test_sanitize_collection_name_long_name(self) -> None:
        """Test sanitizing a very long collection name."""
        long_name = "This is an extremely long role name that will definitely exceed the ChromaDB collection name limit of 63 characters and cause an error when used as a collection name"
        sanitized = _sanitize_collection_name(long_name)
        assert len(sanitized) <= MAX_COLLECTION_LENGTH
        assert sanitized[0].isalnum()
        assert sanitized[-1].isalnum()
        assert all(c.isalnum() or c in ["_", "-"] for c in sanitized)

    def test_sanitize_collection_name_special_chars(self) -> None:
        """Test sanitizing a name with special characters."""
        special_chars = "Agent@123!#$%^&*()"
        sanitized = _sanitize_collection_name(special_chars)
        assert sanitized[0].isalnum()
        assert sanitized[-1].isalnum()
        assert all(c.isalnum() or c in ["_", "-"] for c in sanitized)

    def test_sanitize_collection_name_short_name(self) -> None:
        """Test sanitizing a very short name."""
        short_name = "A"
        sanitized = _sanitize_collection_name(short_name)
        assert len(sanitized) >= MIN_COLLECTION_LENGTH
        assert sanitized[0].isalnum()
        assert sanitized[-1].isalnum()

    def test_sanitize_collection_name_bad_ends(self) -> None:
        """Test sanitizing a name with non-alphanumeric start/end."""
        bad_ends = "_Agent_"
        sanitized = _sanitize_collection_name(bad_ends)
        assert sanitized[0].isalnum()
        assert sanitized[-1].isalnum()

    def test_sanitize_collection_name_none(self) -> None:
        """Test sanitizing a None value."""
        sanitized = _sanitize_collection_name(None)
        assert sanitized == "default_collection"

    def test_sanitize_collection_name_ipv4_pattern(self) -> None:
        """Test sanitizing an IPv4 address."""
        ipv4 = "192.168.1.1"
        sanitized = _sanitize_collection_name(ipv4)
        assert sanitized.startswith("ip_")
        assert sanitized[0].isalnum()
        assert sanitized[-1].isalnum()
        assert all(c.isalnum() or c in ["_", "-"] for c in sanitized)

    def test_is_ipv4_pattern(self) -> None:
        """Test IPv4 pattern detection."""
        assert _is_ipv4_pattern("192.168.1.1") is True
        assert _is_ipv4_pattern("not.an.ip.address") is False

    def test_sanitize_collection_name_properties(self) -> None:
        """Test that sanitized collection names always meet ChromaDB requirements."""
        test_cases: list[str] = [
            "A" * 100,  # Very long name
            "_start_with_underscore",
            "end_with_underscore_",
            "contains@special#characters",
            "192.168.1.1",  # IPv4 address
            "a" * 2,  # Too short
        ]
        for test_case in test_cases:
            sanitized = _sanitize_collection_name(test_case)
            assert len(sanitized) >= MIN_COLLECTION_LENGTH
            assert len(sanitized) <= MAX_COLLECTION_LENGTH
            assert sanitized[0].isalnum()
            assert sanitized[-1].isalnum()

    def test_sanitize_collection_name_empty_string(self) -> None:
        """Test sanitizing an empty string."""
        sanitized = _sanitize_collection_name("")
        assert sanitized == "default_collection"

    def test_sanitize_collection_name_whitespace_only(self) -> None:
        """Test sanitizing a string with only whitespace."""
        sanitized = _sanitize_collection_name("   ")
        assert (
            sanitized == "a__z"
        )  # Spaces become underscores, padded to meet requirements
        assert len(sanitized) >= MIN_COLLECTION_LENGTH
        assert sanitized[0].isalnum()
        assert sanitized[-1].isalnum()
