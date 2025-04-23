"""Tests for conda compatibility."""
import sys
import unittest


class TestCondaCompatibility(unittest.TestCase):
    """Test conda compatibility."""

    def test_python_version_compatibility(self):
        """Test Python version compatibility."""
        version = sys.version_info
        self.assertGreaterEqual(version.major, 3)
        self.assertGreaterEqual(version.minor, 10)
        self.assertLess(version.minor, 13)
        
    def test_typing_self_import(self):
        """Test that Self can be imported from typing."""
        try:
            from typing import Self
            self.assertTrue(True)
        except ImportError:
            if sys.version_info.minor == 10:
                # In Python 3.10, Self might not be available directly from typing
                try:
                    from typing_extensions import Self
                    self.assertTrue(True)
                except ImportError:
                    self.fail("Self not available from typing or typing_extensions in Python 3.10")
            else:
                self.fail("Self not available from typing")
                
    def test_tokenizers_import(self):
        """Test tokenizers import if it's installed."""
        try:
            import tokenizers
            # Only test if tokenizers is installed
            if sys.version_info.minor == 12:
                self.assertTrue(True, "tokenizers successfully imported in Python 3.12")
        except ImportError:
            # Skip test if tokenizers is not installed
            self.skipTest("tokenizers package not installed")


if __name__ == "__main__":
    unittest.main()
