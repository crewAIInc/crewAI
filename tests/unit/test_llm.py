import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from crewai.llm import LLM


class TestLLM(unittest.TestCase):
    @patch("crewai.llm.litellm.completion")
    @patch("crewai.llm.LLM.supports_stop_words")
    def test_call_with_supported_stop_words(self, mock_supports_stop_words, mock_completion):
        mock_supports_stop_words.return_value = True
        
        message = SimpleNamespace(content="Hello, World!")
        choice = SimpleNamespace(message=message)
        response = SimpleNamespace(choices=[choice])
        mock_completion.return_value = response
        
        llm = LLM(model="gpt-4", stop=["STOP"])
        
        messages = [{"role": "user", "content": "Say Hello"}]
        result = llm.call(messages)
        
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args[1]
        self.assertIn("stop", call_args)
        self.assertEqual(call_args["stop"], ["STOP"])
        self.assertEqual(result, "Hello, World!")
    
    @patch("crewai.llm.litellm.completion")
    @patch("crewai.llm.LLM.supports_stop_words")
    def test_call_with_unsupported_stop_words(self, mock_supports_stop_words, mock_completion):
        mock_supports_stop_words.return_value = False
        
        message = SimpleNamespace(content="Hello, World!")
        choice = SimpleNamespace(message=message)
        response = SimpleNamespace(choices=[choice])
        mock_completion.return_value = response
        
        llm = LLM(model="o3", stop=["STOP"])
        
        messages = [{"role": "user", "content": "Say Hello"}]
        result = llm.call(messages)
        
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args[1]
        self.assertNotIn("stop", call_args)
        self.assertEqual(result, "Hello, World!")


if __name__ == "__main__":
    unittest.main()
