from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess
import unittest


class TestTokenProcess(unittest.TestCase):
    def setUp(self):
        self.token_process = TokenProcess()

    def test_sum_cached_prompt_tokens_with_none(self):
        initial_tokens = self.token_process.cached_prompt_tokens
        self.token_process.sum_cached_prompt_tokens(None)
        self.assertEqual(self.token_process.cached_prompt_tokens, initial_tokens)

    def test_sum_cached_prompt_tokens_with_int(self):
        initial_tokens = self.token_process.cached_prompt_tokens
        self.token_process.sum_cached_prompt_tokens(5)
        self.assertEqual(self.token_process.cached_prompt_tokens, initial_tokens + 5)
