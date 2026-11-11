"""Regression test: the internal cache_breakpoint flag must not reach providers.

Groq (served here through OpenRouter) rejects unknown message keys, so a leaked
``cache_breakpoint`` fails the whole request (OSS-272).
"""

from unittest.mock import patch

import litellm
import pytest

from crewai import LLM, Agent


@pytest.mark.vcr
def test_agent_kickoff_strips_cache_breakpoint_on_litellm_path():
    llm = LLM(
        model="openrouter/meta-llama/llama-3.3-70b-instruct",
        is_litellm=True,
        # Pin Groq as the upstream: it is the provider that rejects the flag.
        extra_body={"provider": {"order": ["Groq"], "allow_fallbacks": False}},
    )
    agent = Agent(role="a", goal="g", backstory="b", llm=llm, verbose=False)

    sent: list[list[dict]] = []
    real_completion = litellm.completion

    def spy(**kwargs):
        sent.append(kwargs["messages"])
        return real_completion(**kwargs)

    with patch.object(litellm, "completion", spy):
        result = agent.kickoff("Say hi in one word.")

    assert result.raw
    assert sent, "expected the LiteLLM completion to be called"
    assert all("cache_breakpoint" not in m for call in sent for m in call)
