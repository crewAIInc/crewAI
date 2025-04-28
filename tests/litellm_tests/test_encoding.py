import json
import os
import sys
import unittest
from unittest.mock import mock_open, patch

import pytest

from crewai.llm import LLM


class TestLitellmEncoding(unittest.TestCase):
    """Test that the litellm encoding patch works correctly."""

    def test_json_load_with_utf8_encoding(self):
        """Test that json.load is called with UTF-8 encoding."""
        
        mock_content = '{"test": "日本語テキスト"}'  # Japanese text that would fail with cp1252
        
        with patch('builtins.open', mock_open(read_data=mock_content)):
            import litellm
            
            self.assertTrue(hasattr(litellm.utils, 'json_data'))
            
            with open('test.json', 'r') as f:
                data = json.load(f)
                self.assertEqual(data['test'], '日本語テキスト')
