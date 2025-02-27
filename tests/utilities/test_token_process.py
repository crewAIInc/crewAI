import unittest

from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess


class TestTokenProcess(unittest.TestCase):
    """Test suite for TokenProcess class token counting functionality."""

    def setUp(self):
        """Initialize a fresh TokenProcess instance before each test."""
        self.token_process = TokenProcess()

    def test_sum_cached_prompt_tokens_with_none(self):
        """Test that passing None to sum_cached_prompt_tokens doesn't modify the counter."""
        initial_tokens = self.token_process.cached_prompt_tokens
        self.token_process.sum_cached_prompt_tokens(None)
        self.assertEqual(self.token_process.cached_prompt_tokens, initial_tokens)

    def test_sum_cached_prompt_tokens_with_int(self):
        """Test that passing an integer correctly increments the counter."""
        initial_tokens = self.token_process.cached_prompt_tokens
        self.token_process.sum_cached_prompt_tokens(5)
        self.assertEqual(self.token_process.cached_prompt_tokens, initial_tokens + 5)

    def test_sum_cached_prompt_tokens_with_zero(self):
        """Test that passing zero doesn't modify the counter."""
        initial_tokens = self.token_process.cached_prompt_tokens
        self.token_process.sum_cached_prompt_tokens(0)
        self.assertEqual(self.token_process.cached_prompt_tokens, initial_tokens)

    def test_sum_cached_prompt_tokens_with_large_number(self):
        """Test that the counter works with large numbers."""
        initial_tokens = self.token_process.cached_prompt_tokens
        self.token_process.sum_cached_prompt_tokens(1000000)
        self.assertEqual(self.token_process.cached_prompt_tokens, initial_tokens + 1000000)

    def test_sum_cached_prompt_tokens_multiple_calls(self):
        """Test that multiple calls accumulate correctly, ignoring None values."""
        initial_tokens = self.token_process.cached_prompt_tokens
        self.token_process.sum_cached_prompt_tokens(5)
        self.token_process.sum_cached_prompt_tokens(None)
        self.token_process.sum_cached_prompt_tokens(3)
        self.assertEqual(self.token_process.cached_prompt_tokens, initial_tokens + 8)

    def test_sum_cached_prompt_tokens_with_negative(self):
        """Test that negative values raise ValueError."""
        with self.assertRaises(ValueError) as context:
            self.token_process.sum_cached_prompt_tokens(-1)
        self.assertEqual(str(context.exception), "Token count cannot be negative")
