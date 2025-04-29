import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from crewai.cli.utils import copy_template, load_env_vars, write_env_file, tree_find_and_replace
from crewai.cli.provider import read_cache_file, fetch_provider_data


class TestEncoding(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
        self.unicode_content = "Hello Unicode: 你好, こんにちは, Привет, مرحبا, 안녕하세요 🚀"
        self.src_file = self.test_dir / "src_file.txt"
        self.dst_file = self.test_dir / "dst_file.txt"
        
        with open(self.src_file, "w", encoding="utf-8") as f:
            f.write(self.unicode_content)
            
    def tearDown(self):
        self.temp_dir.cleanup()
        
    def test_copy_template_handles_unicode(self):
        """Test that copy_template handles Unicode characters properly."""
        copy_template(
            self.src_file, 
            self.dst_file, 
            "test_name", 
            "TestClass", 
            "test_folder"
        )
        
        with open(self.dst_file, "r", encoding="utf-8") as f:
            content = f.read()
            
        self.assertIn("你好", content)
        self.assertIn("こんにちは", content)
        self.assertIn("🚀", content)
        
    def test_env_vars_handle_unicode(self):
        """Test that environment variable functions handle Unicode characters properly."""
        test_env_path = self.test_dir / ".env"
        test_env_vars = {
            "KEY1": "Value with Unicode: 你好",
            "KEY2": "More Unicode: こんにちは 🚀"
        }
        
        write_env_file(self.test_dir, test_env_vars)
        
        loaded_vars = load_env_vars(self.test_dir)
        
        self.assertEqual(loaded_vars["KEY1"], "Value with Unicode: 你好")
        self.assertEqual(loaded_vars["KEY2"], "More Unicode: こんにちは 🚀")
        
    def test_tree_find_and_replace_handles_unicode(self):
        """Test that tree_find_and_replace handles Unicode characters properly."""
        test_file = self.test_dir / "replace_test.txt"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("Replace this: PLACEHOLDER with Unicode: 你好")
            
        tree_find_and_replace(self.test_dir, "PLACEHOLDER", "🚀")
        
        with open(test_file, "r", encoding="utf-8") as f:
            content = f.read()
            
        self.assertIn("Replace this: 🚀 with Unicode: 你好", content)
        
    @patch("crewai.cli.provider.requests.get")
    def test_provider_functions_handle_unicode(self, mock_get):
        """Test that provider data functions handle Unicode properly."""
        mock_response = unittest.mock.Mock()
        mock_response.iter_content.return_value = [self.unicode_content.encode("utf-8")]
        mock_response.headers.get.return_value = str(len(self.unicode_content))
        mock_get.return_value = mock_response
        
        cache_file = self.test_dir / "cache.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write('{"model": "Unicode test: 你好 🚀"}')
            
        cache_data = read_cache_file(cache_file)
        self.assertEqual(cache_data["model"], "Unicode test: 你好 🚀")
