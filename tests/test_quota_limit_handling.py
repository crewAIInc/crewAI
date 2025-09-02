import pytest
from unittest.mock import Mock, patch
from litellm.exceptions import RateLimitError

from crewai.llm import LLM
from crewai.agent import Agent
from crewai.task import Task
from crewai.crew import Crew
from crewai.utilities.exceptions import LLMQuotaLimitExceededException


class TestQuotaLimitHandling:
    """Test suite for quota limit handling in CrewAI."""

    def test_llm_non_streaming_quota_limit_exception(self):
        """Test that LLM raises LLMQuotaLimitExceededException for rate limit errors in non-streaming mode."""
        llm = LLM(model="gpt-3.5-turbo", stream=False)
        
        with patch("litellm.completion") as mock_completion:
            mock_completion.side_effect = RateLimitError("Rate limit exceeded")
            
            with pytest.raises(LLMQuotaLimitExceededException) as exc_info:
                llm.call(messages=[{"role": "user", "content": "Hello"}])
            
            assert "quota limit exceeded" in str(exc_info.value).lower()
            assert "Rate limit exceeded" in str(exc_info.value)

    def test_llm_streaming_quota_limit_exception(self):
        """Test that LLM raises LLMQuotaLimitExceededException for rate limit errors in streaming mode."""
        llm = LLM(model="gpt-3.5-turbo", stream=True)
        
        with patch("litellm.completion") as mock_completion:
            mock_completion.side_effect = RateLimitError("API quota exceeded")
            
            with pytest.raises(LLMQuotaLimitExceededException) as exc_info:
                llm.call(messages=[{"role": "user", "content": "Hello"}])
            
            assert "quota limit exceeded" in str(exc_info.value).lower()
            assert "API quota exceeded" in str(exc_info.value)

    def test_agent_handles_quota_limit_gracefully(self):
        """Test that Agent handles quota limit exceptions gracefully."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            llm=LLM(model="gpt-3.5-turbo")
        )
        
        with patch.object(agent.llm, "call") as mock_call:
            mock_call.side_effect = LLMQuotaLimitExceededException("Quota exceeded")
            
            with pytest.raises(LLMQuotaLimitExceededException):
                agent.execute_task(
                    task=Task(description="Test task", agent=agent),
                    context="Test context"
                )

    def test_crew_handles_quota_limit_in_task_execution(self):
        """Test that Crew handles quota limit exceptions during task execution."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            llm=LLM(model="gpt-3.5-turbo")
        )
        
        task = Task(
            description="Test task",
            agent=agent
        )
        
        crew = Crew(
            agents=[agent],
            tasks=[task]
        )
        
        with patch.object(agent.llm, "call") as mock_call:
            mock_call.side_effect = LLMQuotaLimitExceededException("Monthly quota exceeded")
            
            with pytest.raises(LLMQuotaLimitExceededException):
                crew.kickoff()

    def test_quota_limit_exception_error_message_format(self):
        """Test that LLMQuotaLimitExceededException formats error messages correctly."""
        original_error = "Resource exhausted: Quota exceeded for requests per day"
        exception = LLMQuotaLimitExceededException(original_error)
        
        error_message = str(exception)
        assert "LLM quota limit exceeded" in error_message
        assert original_error in error_message
        assert "API quota or rate limit has been reached" in error_message
        assert "upgrade your plan" in error_message

    def test_quota_limit_exception_preserves_original_error(self):
        """Test that LLMQuotaLimitExceededException preserves the original error message."""
        original_error = "429 Too Many Requests: Rate limit exceeded"
        exception = LLMQuotaLimitExceededException(original_error)
        
        assert exception.original_error_message == original_error

    @pytest.mark.parametrize("error_message,should_match", [
        ("quota exceeded", True),
        ("rate limit exceeded", True),
        ("resource exhausted", True),
        ("too many requests", True),
        ("quota limit reached", True),
        ("api quota exceeded", True),
        ("usage limit exceeded", True),
        ("billing quota exceeded", True),
        ("request limit exceeded", True),
        ("daily quota exceeded", True),
        ("monthly quota exceeded", True),
        ("QUOTA EXCEEDED", True),  # Case insensitive
        ("Rate Limit Exceeded", True),  # Case insensitive
        ("some other error", False),
        ("network timeout", False),
    ])
    def test_quota_limit_error_detection(self, error_message, should_match):
        """Test that quota limit error detection works for various error messages."""
        exception = LLMQuotaLimitExceededException(error_message)
        assert exception._is_quota_limit_error(error_message) == should_match

    def test_different_provider_quota_errors(self):
        """Test quota limit handling for different LLM providers."""
        test_cases = [
            "Rate limit reached for requests",
            "rate_limit_error: Number of requests per minute exceeded",
            "RESOURCE_EXHAUSTED: Quota exceeded",
            "429 Too Many Requests",
        ]
        
        llm = LLM(model="gpt-3.5-turbo")
        
        for error_message in test_cases:
            with patch("litellm.completion") as mock_completion:
                mock_completion.side_effect = RateLimitError(error_message)
                
                with pytest.raises(LLMQuotaLimitExceededException) as exc_info:
                    llm.call(messages=[{"role": "user", "content": "Hello"}])
                
                assert error_message in str(exc_info.value)

    def test_quota_limit_vs_context_window_exceptions(self):
        """Test that quota limit and context window exceptions are handled separately."""
        from litellm.exceptions import ContextWindowExceededError
        from crewai.utilities.exceptions import LLMContextLengthExceededException
        
        llm = LLM(model="gpt-3.5-turbo")
        
        with patch("litellm.completion") as mock_completion:
            mock_completion.side_effect = RateLimitError("Quota exceeded")
            
            with pytest.raises(LLMQuotaLimitExceededException):
                llm.call(messages=[{"role": "user", "content": "Hello"}])
        
        with patch("litellm.completion") as mock_completion:
            mock_completion.side_effect = ContextWindowExceededError("Context length exceeded")
            
            with pytest.raises(LLMContextLengthExceededException):
                llm.call(messages=[{"role": "user", "content": "Hello"}])

    def test_quota_limit_exception_in_crew_agent_executor(self):
        """Test that CrewAgentExecutor handles quota limit exceptions properly."""
        from crewai.agents.crew_agent_executor import CrewAgentExecutor
        
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            llm=LLM(model="gpt-3.5-turbo")
        )
        
        executor = CrewAgentExecutor(agent=agent)
        
        with patch.object(agent.llm, "call") as mock_call:
            mock_call.side_effect = LLMQuotaLimitExceededException("Daily quota exceeded")
            
            with pytest.raises(LLMQuotaLimitExceededException):
                executor.invoke({
                    "input": "Test input",
                    "chat_history": [],
                    "agent_scratchpad": ""
                })
