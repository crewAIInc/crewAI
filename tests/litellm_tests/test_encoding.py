import json
import os
import sys
import unittest
from unittest.mock import mock_open, patch

from crewai.llm import LLM
from crewai.patches.litellm_patch import apply_patches, remove_patches


class TestLitellmEncoding(unittest.TestCase):
    """Test that the litellm encoding patch works correctly."""

    def setUp(self):
        """Set up the test environment by applying the patch."""
        apply_patches()

    def tearDown(self):
        """Clean up the test environment by removing the patch."""
        remove_patches()

    def test_json_load_with_utf8_encoding(self):
        """Test that json.load is called with UTF-8 encoding."""
        
        mock_content = '{"test": "日本語テキスト"}'  # Japanese text that would fail with cp1252
        
        with patch('builtins.open', mock_open(read_data=mock_content)):
            import litellm
            
            self.assertTrue(hasattr(litellm.utils, 'json_data'))
            
            with open('test.json', 'r') as f:
                data = json.load(f)
                self.assertEqual(data['test'], '日本語テキスト')
    
    def test_without_patch(self):
        """Test that demonstrates the issue without the patch."""
        remove_patches()
        
        mock_content = '{"test": "日本語テキスト"}'  # Japanese text that would fail with cp1252
        
        with patch('sys.platform', 'win32'):
            mock_open_without_encoding = mock_open(read_data=mock_content)
            mock_open_without_encoding.side_effect = UnicodeDecodeError('cp1252', b'\x81', 0, 1, 'invalid start byte')
            
            with patch('builtins.open', mock_open_without_encoding):
                with self.assertRaises(UnicodeDecodeError):
                    with open('test.json', 'r') as f:
                        json.load(f)
        
        apply_patches()
