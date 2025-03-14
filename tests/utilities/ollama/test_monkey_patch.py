"""
Tests for the Ollama monkey patch utility.
"""

import unittest
from unittest.mock import patch, MagicMock, call
import json
from types import SimpleNamespace
import pytest

from crewai.utilities.ollama.monkey_patch import (
    apply_monkey_patch,
    query_ollama,
    extract_prompt_from_messages
)


class TestOllamaMonkeyPatch(unittest.TestCase):
    """Test cases for the Ollama monkey patch utility."""

    def test_extract_prompt_from_messages(self):
        """Test extracting a prompt from a list of messages."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
            {"role": "user", "content": "Tell me about CrewAI."}
        ]
        
        prompt = extract_prompt_from_messages(messages)
        
        self.assertIn("System:", prompt)
        self.assertIn("You are a helpful assistant.", prompt)
        self.assertIn("User:", prompt)
        self.assertIn("Hello, how are you?", prompt)
        self.assertIn("Assistant:", prompt)
        self.assertIn("I'm doing well, thank you!", prompt)
        self.assertIn("Tell me about CrewAI.", prompt)

    @patch('requests.post')
    def test_query_ollama_non_streaming(self, mock_post):
        """Test querying Ollama API in non-streaming mode."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "This is a test response."}
        mock_post.return_value = mock_response
        
        # Call the function
        result = query_ollama(
            prompt="Test prompt",
            model="llama3",
            base_url="http://localhost:11434",
            stream=False,
            temperature=0.5
        )
        
        # Verify the result
        self.assertEqual(result, "This is a test response.")
        
        # Verify the API call
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "http://localhost:11434/api/generate")
        self.assertEqual(kwargs["json"]["model"], "llama3")
        self.assertEqual(kwargs["json"]["prompt"], "Test prompt")
        self.assertEqual(kwargs["json"]["options"]["temperature"], 0.5)
        self.assertEqual(kwargs["json"]["options"]["stream"], False)

    @patch('requests.post')
    def test_query_ollama_streaming(self, mock_post):
        """Test querying Ollama API in streaming mode."""
        # Mock the response for streaming
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            json.dumps({"response": "This"}).encode(),
            json.dumps({"response": " is"}).encode(),
            json.dumps({"response": " a"}).encode(),
            json.dumps({"response": " test"}).encode(),
            json.dumps({"response": " response.", "done": True}).encode()
        ]
        mock_post.return_value = mock_response
        
        # Call the function
        result = query_ollama(
            prompt="Test prompt",
            model="llama3",
            base_url="http://localhost:11434",
            stream=True,
            temperature=0.5
        )
        
        # Verify the result is a generator
        self.assertTrue(hasattr(result, '__next__'))
        
        # Consume the generator and verify the results
        chunks = list(result)
        self.assertEqual(chunks, ["This", " is", " a", " test", " response."])
        
        # Verify the API call
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "http://localhost:11434/api/generate")
        self.assertEqual(kwargs["json"]["model"], "llama3")
        self.assertEqual(kwargs["json"]["prompt"], "Test prompt")
        self.assertEqual(kwargs["json"]["options"]["temperature"], 0.5)
        self.assertEqual(kwargs["json"]["options"]["stream"], True)

    @patch('requests.post')
    def test_query_ollama_with_stop_sequences(self, mock_post):
        """Test querying Ollama API with stop sequences."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "This is a test response."}
        mock_post.return_value = mock_response
        
        # Call the function with stop sequences
        result = query_ollama(
            prompt="Test prompt",
            model="llama3",
            stop=["END", "STOP"]
        )
        
        # Verify the API call includes stop sequences
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs["json"]["options"]["stop"], ["END", "STOP"])

    @patch('requests.post')
    def test_query_ollama_error_handling(self, mock_post):
        """Test error handling in query_ollama."""
        # Mock the response to raise an exception
        mock_post.side_effect = Exception("Test error")
        
        # Call the function
        result = query_ollama(prompt="Test prompt")
        
        # Verify the result contains the error message
        self.assertIn("Error:", result)
        self.assertIn("Test error", result)

    @patch('litellm.completion')
    def test_apply_monkey_patch(self, mock_completion):
        """Test applying the monkey patch."""
        # Apply the monkey patch
        result = apply_monkey_patch()
        
        # Verify the result
        self.assertTrue(result)
        
        # Verify that litellm.completion has been replaced
        import litellm
        self.assertNotEqual(litellm.completion, mock_completion)

    @patch('crewai.utilities.ollama.monkey_patch.query_ollama')
    @patch('litellm.completion')
    def test_custom_completion_non_ollama_model(self, mock_original_completion, mock_query_ollama):
        """Test that non-Ollama models are passed to the original completion function."""
        # Apply the monkey patch
        apply_monkey_patch()
        
        # Import litellm to get the patched completion function
        import litellm
        
        # Call the patched completion function with a non-Ollama model
        litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        # Verify that the original completion function was called
        mock_original_completion.assert_called_once()
        
        # Verify that query_ollama was not called
        mock_query_ollama.assert_not_called()

    @patch('crewai.utilities.ollama.monkey_patch.query_ollama')
    @patch('litellm.completion')
    def test_custom_completion_ollama_model_non_streaming(self, mock_original_completion, mock_query_ollama):
        """Test the custom completion function with an Ollama model in non-streaming mode."""
        # Set up the mock
        mock_query_ollama.return_value = "This is a test response."
        
        # Apply the monkey patch
        apply_monkey_patch()
        
        # Import litellm to get the patched completion function
        import litellm
        
        # Call the patched completion function with an Ollama model
        result = litellm.completion(
            model="ollama/llama3",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.5
        )
        
        # Verify that the original completion function was not called
        mock_original_completion.assert_not_called()
        
        # Verify that query_ollama was called
        mock_query_ollama.assert_called_once()
        
        # Verify the result structure
        self.assertEqual(result.choices[0].message.content, "This is a test response.")
        self.assertEqual(result.choices[0].finish_reason, "stop")
        self.assertEqual(result.model, "ollama/llama3")
        self.assertIsNotNone(result.usage)

    @patch('crewai.utilities.ollama.monkey_patch.query_ollama')
    @patch('litellm.completion')
    def test_custom_completion_ollama_model_streaming(self, mock_original_completion, mock_query_ollama):
        """Test the custom completion function with an Ollama model in streaming mode."""
        # Set up the mock to return a generator
        mock_query_ollama.return_value = (chunk for chunk in ["This", " is", " a", " test", " response."])
        
        # Apply the monkey patch
        apply_monkey_patch()
        
        # Import litellm to get the patched completion function
        import litellm
        
        # Call the patched completion function with an Ollama model in streaming mode
        result = litellm.completion(
            model="ollama/llama3",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.5,
            stream=True
        )
        
        # Verify that the original completion function was not called
        mock_original_completion.assert_not_called()
        
        # Verify that query_ollama was called
        mock_query_ollama.assert_called_once()
        
        # Verify the result is a generator
        self.assertTrue(hasattr(result, '__next__'))
        
        # Consume the generator and verify the structure of each chunk
        chunks = list(result)
        
        # Verify we have the expected number of chunks (5 content chunks + 1 final chunk)
        self.assertEqual(len(chunks), 6)
        
        # Check the content of the first 5 chunks
        for i, expected_content in enumerate(["This", " is", " a", " test", " response."]):
            self.assertEqual(chunks[i].choices[0].delta.content, expected_content)
            self.assertEqual(chunks[i].choices[0].delta.role, "assistant")
            self.assertIsNone(chunks[i].choices[0].finish_reason)
        
        # Check the final chunk
        self.assertEqual(chunks[5].choices[0].delta.content, "")
        self.assertEqual(chunks[5].choices[0].finish_reason, "stop")
        self.assertIsNotNone(chunks[5].usage)
