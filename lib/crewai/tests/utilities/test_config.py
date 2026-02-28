"""Tests for the config utility functions."""

import unittest
from unittest.mock import Mock
from typing import Any

from crewai.utilities.config import process_config


class MockBaseModel:
    """Mock BaseModel for testing purposes."""
    model_fields = {"test_field": "field_info"}


class TestProcessConfig(unittest.TestCase):
    """Test cases for the process_config function."""

    def test_process_config_with_dict_input(self):
        """Test that process_config works correctly with dict input (normal case)."""
        values = {
            "test_field": "original_value",
            "config": {"other_field": "config_value"}
        }
        
        result = process_config(values, MockBaseModel)
        
        # Should process normally and return a dict
        self.assertIsInstance(result, dict)
        self.assertIn("test_field", result)
        self.assertEqual(result["test_field"], "original_value")
        # Config should be removed after processing
        self.assertNotIn("config", result)

    def test_process_config_with_empty_dict(self):
        """Test that process_config works with empty dict."""
        values = {}
        
        result = process_config(values, MockBaseModel)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result, {})

    def test_process_config_with_string_input(self):
        """Test that process_config handles string input gracefully (fixes #4419)."""
        # This was the original bug - passing a string would cause AttributeError
        string_input = "not_an_agent"
        
        result = process_config(string_input, MockBaseModel)
        
        # Should return the input unchanged when it's not a dict
        self.assertEqual(result, "not_an_agent")
        self.assertIsInstance(result, str)

    def test_process_config_with_none_input(self):
        """Test that process_config handles None input gracefully."""
        result = process_config(None, MockBaseModel)
        
        # Should return None unchanged
        self.assertIsNone(result)

    def test_process_config_with_list_input(self):
        """Test that process_config handles list input gracefully."""
        list_input = ["item1", "item2"]
        
        result = process_config(list_input, MockBaseModel)
        
        # Should return the list unchanged
        self.assertEqual(result, list_input)
        self.assertIsInstance(result, list)

    def test_process_config_with_int_input(self):
        """Test that process_config handles integer input gracefully."""
        int_input = 42
        
        result = process_config(int_input, MockBaseModel)
        
        # Should return the integer unchanged
        self.assertEqual(result, 42)
        self.assertIsInstance(result, int)

    def test_process_config_with_custom_object(self):
        """Test that process_config handles custom object input gracefully."""
        custom_obj = Mock()
        
        result = process_config(custom_obj, MockBaseModel)
        
        # Should return the object unchanged
        self.assertEqual(result, custom_obj)

    def test_original_bug_reproduction(self):
        """Test that the original bug from #4419 is fixed.
        
        The original bug was:
        AttributeError: 'str' object has no attribute 'get'
        
        This happened when Task(agent="string") was passed because
        process_config was called with the string value and tried
        to call .get() on it.
        """
        # This would have crashed before the fix
        problematic_inputs = [
            "not_an_agent",
            42,
            [],
            None,
            Mock(),
        ]
        
        for problematic_input in problematic_inputs:
            with self.subTest(input_type=type(problematic_input).__name__):
                # Should not raise AttributeError about .get()
                try:
                    result = process_config(problematic_input, MockBaseModel)
                    # Should return the input unchanged for non-dict types
                    self.assertEqual(result, problematic_input)
                except AttributeError as e:
                    if "has no attribute 'get'" in str(e):
                        self.fail(f"Original bug still present for {type(problematic_input).__name__}: {e}")
                    else:
                        # Other AttributeErrors might be legitimate
                        raise


if __name__ == "__main__":
    unittest.main()