from typing import Any, Dict, List, Optional, Union

import pytest

from crewai import Agent, Crew, Process, Task
from crewai.llms.base_llm import BaseLLM
from crewai.utilities.llm_utils import create_llm


class CustomLLM(BaseLLM):
    """Custom LLM implementation for testing.

    This is a simple implementation of the BaseLLM abstract base class
    that returns a predefined response for testing purposes.
    """

    def __init__(self, response="Default response", model="test-model"):
        """Initialize the CustomLLM with a predefined response.

        Args:
            response: The predefined response to return from call().
        """
        super().__init__(model=model)
        self.response = response
        self.call_count = 0

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
    ):
        """
        Mock LLM call that returns a predefined response.
        Properly formats messages to match OpenAI's expected structure.
        """
        self.call_count += 1

        # If input is a string, convert to proper message format
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # Ensure each message has properly formatted content
        for message in messages:
            if isinstance(message["content"], str):
                message["content"] = [{"type": "text", "text": message["content"]}]

        # Return predefined response in expected format
        if "Thought:" in str(messages):
            return f"Thought: I will say hi\nFinal Answer: {self.response}"
        return self.response

    def supports_function_calling(self) -> bool:
        """Return False to indicate that function calling is not supported.

        Returns:
            False, indicating that this LLM does not support function calling.
        """
        return False

    def supports_stop_words(self) -> bool:
        """Return False to indicate that stop words are not supported.

        Returns:
            False, indicating that this LLM does not support stop words.
        """
        return False

    def get_context_window_size(self) -> int:
        """Return a default context window size.

        Returns:
            4096, a typical context window size for modern LLMs.
        """
        return 4096


@pytest.mark.vcr(filter_headers=["authorization"])
def test_custom_llm_implementation():
    """Test that a custom LLM implementation works with create_llm."""
    custom_llm = CustomLLM(response="The answer is 42")

    # Test that create_llm returns the custom LLM instance directly
    result_llm = create_llm(custom_llm)

    assert result_llm is custom_llm

    # Test calling the custom LLM
    response = result_llm.call(
        "What is the answer to life, the universe, and everything?"
    )

    # Verify that the response from the custom LLM was used
    assert "42" in response


@pytest.mark.vcr(filter_headers=["authorization"])
def test_custom_llm_within_crew():
    """Test that a custom LLM implementation works with create_llm."""
    custom_llm = CustomLLM(response="Hello! Nice to meet you!", model="test-model")

    agent = Agent(
        role="Say Hi",
        goal="Say hi to the user",
        backstory="""You just say hi to the user""",
        llm=custom_llm,
    )

    task = Task(
        description="Say hi to the user",
        expected_output="A greeting to the user",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
    )

    result = crew.kickoff()

    # Assert the LLM was called
    assert custom_llm.call_count > 0
    # Assert we got a response
    assert "Hello!" in result.raw


def test_custom_llm_message_formatting():
    """Test that the custom LLM properly formats messages"""
    custom_llm = CustomLLM(response="Test response", model="test-model")

    # Test with string input
    result = custom_llm.call("Test message")
    assert result == "Test response"

    # Test with message list
    messages = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "User message"},
    ]
    result = custom_llm.call(messages)
    assert result == "Test response"


class JWTAuthLLM(BaseLLM):
    """Custom LLM implementation with JWT authentication."""

    def __init__(self, jwt_token: str):
        super().__init__(model="test-model")
        if not jwt_token or not isinstance(jwt_token, str):
            raise ValueError("Invalid JWT token")
        self.jwt_token = jwt_token
        self.calls = []
        self.stop = []

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Any]:
        """Record the call and return a predefined response."""
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "callbacks": callbacks,
                "available_functions": available_functions,
            }
        )
        # In a real implementation, this would use the JWT token to authenticate
        # with an external service
        return "Response from JWT-authenticated LLM"

    def supports_function_calling(self) -> bool:
        """Return True to indicate that function calling is supported."""
        return True

    def supports_stop_words(self) -> bool:
        """Return True to indicate that stop words are supported."""
        return True

    def get_context_window_size(self) -> int:
        """Return a default context window size."""
        return 8192


def test_custom_llm_with_jwt_auth():
    """Test a custom LLM implementation with JWT authentication."""
    jwt_llm = JWTAuthLLM(jwt_token="example.jwt.token")

    # Test that create_llm returns the JWT-authenticated LLM instance directly
    result_llm = create_llm(jwt_llm)

    assert result_llm is jwt_llm

    # Test calling the JWT-authenticated LLM
    response = result_llm.call("Test message")

    # Verify that the JWT-authenticated LLM was called
    assert len(jwt_llm.calls) > 0
    # Verify that the response from the JWT-authenticated LLM was used
    assert response == "Response from JWT-authenticated LLM"


def test_jwt_auth_llm_validation():
    """Test that JWT token validation works correctly."""
    # Test with invalid JWT token (empty string)
    with pytest.raises(ValueError, match="Invalid JWT token"):
        JWTAuthLLM(jwt_token="")

    # Test with invalid JWT token (non-string)
    with pytest.raises(ValueError, match="Invalid JWT token"):
        JWTAuthLLM(jwt_token=None)


class TimeoutHandlingLLM(BaseLLM):
    """Custom LLM implementation with timeout handling and retry logic."""

    def __init__(self, max_retries: int = 3, timeout: int = 30):
        """Initialize the TimeoutHandlingLLM with retry and timeout settings.

        Args:
            max_retries: Maximum number of retry attempts.
            timeout: Timeout in seconds for each API call.
        """
        super().__init__(model="test-model")
        self.max_retries = max_retries
        self.timeout = timeout
        self.calls = []
        self.stop = []
        self.fail_count = 0  # Number of times to simulate failure

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Any]:
        """Simulate API calls with timeout handling and retry logic.

        Args:
            messages: Input messages for the LLM.
            tools: Optional list of tool schemas for function calling.
            callbacks: Optional list of callback functions.
            available_functions: Optional dict mapping function names to callables.

        Returns:
            A response string based on whether this is the first attempt or a retry.

        Raises:
            TimeoutError: If all retry attempts fail.
        """
        # Record the initial call
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "callbacks": callbacks,
                "available_functions": available_functions,
                "attempt": 0,
            }
        )

        # Simulate retry logic
        for attempt in range(self.max_retries):
            # Skip the first attempt recording since we already did that above
            if attempt == 0:
                # Simulate a failure if fail_count > 0
                if self.fail_count > 0:
                    self.fail_count -= 1
                    # If we've used all retries, raise an error
                    if attempt == self.max_retries - 1:
                        raise TimeoutError(
                            f"LLM request failed after {self.max_retries} attempts"
                        )
                    # Otherwise, continue to the next attempt (simulating backoff)
                    continue
                else:
                    # Success on first attempt
                    return "First attempt response"
            else:
                # This is a retry attempt (attempt > 0)
                # Always record retry attempts
                self.calls.append(
                    {
                        "retry_attempt": attempt,
                        "messages": messages,
                        "tools": tools,
                        "callbacks": callbacks,
                        "available_functions": available_functions,
                    }
                )

                # Simulate a failure if fail_count > 0
                if self.fail_count > 0:
                    self.fail_count -= 1
                    # If we've used all retries, raise an error
                    if attempt == self.max_retries - 1:
                        raise TimeoutError(
                            f"LLM request failed after {self.max_retries} attempts"
                        )
                    # Otherwise, continue to the next attempt (simulating backoff)
                    continue
                else:
                    # Success on retry
                    return "Response after retry"

    def supports_function_calling(self) -> bool:
        """Return True to indicate that function calling is supported.

        Returns:
            True, indicating that this LLM supports function calling.
        """
        return True

    def supports_stop_words(self) -> bool:
        """Return True to indicate that stop words are supported.

        Returns:
            True, indicating that this LLM supports stop words.
        """
        return True

    def get_context_window_size(self) -> int:
        """Return a default context window size.

        Returns:
            8192, a typical context window size for modern LLMs.
        """
        return 8192


def test_timeout_handling_llm():
    """Test a custom LLM implementation with timeout handling and retry logic."""
    # Test successful first attempt
    llm = TimeoutHandlingLLM()
    response = llm.call("Test message")
    assert response == "First attempt response"
    assert len(llm.calls) == 1

    # Test successful retry
    llm = TimeoutHandlingLLM()
    llm.fail_count = 1  # Fail once, then succeed
    response = llm.call("Test message")
    assert response == "Response after retry"
    assert len(llm.calls) == 2  # Initial call + successful retry call

    # Test failure after all retries
    llm = TimeoutHandlingLLM(max_retries=2)
    llm.fail_count = 2  # Fail twice, which is all retries
    with pytest.raises(TimeoutError, match="LLM request failed after 2 attempts"):
        llm.call("Test message")
    assert len(llm.calls) == 2  # Initial call + failed retry attempt
