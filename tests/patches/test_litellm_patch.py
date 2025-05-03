"""
Test for the litellm patch that fixes the IndexError in ollama_pt function.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys

import litellm
from litellm.litellm_core_utils.prompt_templates.factory import ollama_pt

from crewai.patches.litellm_patch import patch_litellm_ollama_pt


class TestLitellmPatch(unittest.TestCase):
    def test_ollama_pt_patch_fixes_index_error(self):
        """Test that the patch fixes the IndexError in ollama_pt."""
        # Create a message list where the assistant message is the last one
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        
        # Store the original function to restore it after the test
        original_ollama_pt = litellm.litellm_core_utils.prompt_templates.factory.ollama_pt
        
        try:
            # Apply the patch
            patch_litellm_ollama_pt()
            
            # The patched function should not raise an IndexError
            result = ollama_pt("qwen3:4b", messages)
            
            # Verify the result is as expected
            self.assertIn("prompt", result)
            self.assertIn("images", result)
            self.assertIn("### User:\nHello", result["prompt"])
            self.assertIn("### Assistant:\nHi there", result["prompt"])
        finally:
            # Restore the original function to avoid affecting other tests
            litellm.litellm_core_utils.prompt_templates.factory.ollama_pt = original_ollama_pt


if __name__ == "__main__":
    unittest.main()
