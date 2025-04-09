import unittest
from typing import Any, Dict, List, Union

import pytest

from crewai.utilities.chromadb import (
    MAX_COLLECTION_LENGTH,
    MIN_COLLECTION_LENGTH,
    is_ipv4_pattern,
    sanitize_collection_name,
)


class TestChromadbUtils(unittest.TestCase):
    def test_sanitize_collection_name_long_name(self):
        """Test sanitizing a very long collection name."""
        long_name = "This is an extremely long role name that will definitely exceed the ChromaDB collection name limit of 63 characters and cause an error when used as a collection name"
        sanitized = sanitize_collection_name(long_name)
        self.assertLessEqual(len(sanitized), MAX_COLLECTION_LENGTH)
        self.assertTrue(sanitized[0].isalnum())
        self.assertTrue(sanitized[-1].isalnum())
        self.assertTrue(all(c.isalnum() or c in ["_", "-"] for c in sanitized))

    def test_sanitize_collection_name_special_chars(self):
        """Test sanitizing a name with special characters."""
        special_chars = "Agent@123!#$%^&*()"
        sanitized = sanitize_collection_name(special_chars)
        self.assertTrue(sanitized[0].isalnum())
        self.assertTrue(sanitized[-1].isalnum())
        self.assertTrue(all(c.isalnum() or c in ["_", "-"] for c in sanitized))

    def test_sanitize_collection_name_short_name(self):
        """Test sanitizing a very short name."""
        short_name = "A"
        sanitized = sanitize_collection_name(short_name)
        self.assertGreaterEqual(len(sanitized), MIN_COLLECTION_LENGTH)
        self.assertTrue(sanitized[0].isalnum())
        self.assertTrue(sanitized[-1].isalnum())

    def test_sanitize_collection_name_bad_ends(self):
        """Test sanitizing a name with non-alphanumeric start/end."""
        bad_ends = "_Agent_"
        sanitized = sanitize_collection_name(bad_ends)
        self.assertTrue(sanitized[0].isalnum())
        self.assertTrue(sanitized[-1].isalnum())

    def test_sanitize_collection_name_none(self):
        """Test sanitizing a None value."""
        sanitized = sanitize_collection_name(None)
        self.assertEqual(sanitized, "default_collection")

    def test_sanitize_collection_name_ipv4_pattern(self):
        """Test sanitizing an IPv4 address."""
        ipv4 = "192.168.1.1"
        sanitized = sanitize_collection_name(ipv4)
        self.assertTrue(sanitized.startswith("ip_"))
        self.assertTrue(sanitized[0].isalnum())
        self.assertTrue(sanitized[-1].isalnum())
        self.assertTrue(all(c.isalnum() or c in ["_", "-"] for c in sanitized))

    def test_is_ipv4_pattern(self):
        """Test IPv4 pattern detection."""
        self.assertTrue(is_ipv4_pattern("192.168.1.1"))
        self.assertFalse(is_ipv4_pattern("not.an.ip.address"))

    def test_sanitize_collection_name_properties(self):
        """Test that sanitized collection names always meet ChromaDB requirements."""
        test_cases = [
            "A" * 100,  # Very long name
            "_start_with_underscore",
            "end_with_underscore_",
            "contains@special#characters",
            "192.168.1.1",  # IPv4 address
            "a" * 2,  # Too short
        ]
        for test_case in test_cases:
            sanitized = sanitize_collection_name(test_case)
            self.assertGreaterEqual(len(sanitized), MIN_COLLECTION_LENGTH)
            self.assertLessEqual(len(sanitized), MAX_COLLECTION_LENGTH)
            self.assertTrue(sanitized[0].isalnum())
            self.assertTrue(sanitized[-1].isalnum())
