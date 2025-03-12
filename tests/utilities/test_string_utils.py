import unittest

from crewai.utilities import sanitize_collection_name


class TestStringUtils(unittest.TestCase):
    def test_sanitize_collection_name_long_name(self):
        """Test sanitizing a very long collection name."""
        long_name = "This is an extremely long role name that will definitely exceed the ChromaDB collection name limit of 63 characters and cause an error when used as a collection name"
        sanitized = sanitize_collection_name(long_name)
        self.assertLessEqual(len(sanitized), 63)
        self.assertTrue(sanitized[0].isalnum())
        self.assertTrue(sanitized[-1].isalnum())
        self.assertTrue(all(c.isalnum() or c in ['_', '-'] for c in sanitized))
    
    def test_sanitize_collection_name_special_chars(self):
        """Test sanitizing a name with special characters."""
        special_chars = "Agent@123!#$%^&*()"
        sanitized = sanitize_collection_name(special_chars)
        self.assertTrue(sanitized[0].isalnum())
        self.assertTrue(sanitized[-1].isalnum())
        self.assertTrue(all(c.isalnum() or c in ['_', '-'] for c in sanitized))
    
    def test_sanitize_collection_name_short_name(self):
        """Test sanitizing a very short name."""
        short_name = "A"
        sanitized = sanitize_collection_name(short_name)
        self.assertGreaterEqual(len(sanitized), 3)
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

if __name__ == '__main__':
    unittest.main()
