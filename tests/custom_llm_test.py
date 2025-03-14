from typing import Any, Dict, List, Optional, Union

import pytest

from crewai.llm import LLM
from crewai.utilities.llm_utils import create_llm


class CustomLLM(LLM):
    """Custom LLM implementation for testing.
    
    This is a simple implementation of the LLM abstract base class
    that returns a predefined response for testing purposes.
    """
    
    def __init__(self, response: str = "Custom LLM response"):
        """Initialize the CustomLLM with a predefined response.
        
        Args:
            response: The predefined response to return from call().
        """
        super().__init__()
        self.response = response
        self.calls = []
        self.stop = []
        
    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Any]:
        """Record the call and return the predefined response.
        
        Args:
            messages: Input messages for the LLM.
            tools: Optional list of tool schemas for function calling.
            callbacks: Optional list of callback functions.
            available_functions: Optional dict mapping function names to callables.
            
        Returns:
            The predefined response string.
        """
        self.calls.append({
            "messages": messages, 
            "tools": tools,
            "callbacks": callbacks,
            "available_functions": available_functions
        })
        return self.response
        
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


def test_custom_llm_implementation():
    """Test that a custom LLM implementation works with create_llm."""
    custom_llm = CustomLLM(response="The answer is 42")
    
    # Test that create_llm returns the custom LLM instance directly
    result_llm = create_llm(custom_llm)
    
    assert result_llm is custom_llm
    
    # Test calling the custom LLM
    response = result_llm.call("What is the answer to life, the universe, and everything?")
    
    # Verify that the custom LLM was called
    assert len(custom_llm.calls) > 0
    # Verify that the response from the custom LLM was used
    assert response == "The answer is 42"


class JWTAuthLLM(LLM):
    """Custom LLM implementation with JWT authentication."""
    
    def __init__(self, jwt_token: str):
        super().__init__()
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
        self.calls.append({
            "messages": messages, 
            "tools": tools,
            "callbacks": callbacks,
            "available_functions": available_functions
        })
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


class TimeoutHandlingLLM(LLM):
    """Custom LLM implementation with timeout handling and retry logic."""
    
    def __init__(self, max_retries: int = 3, timeout: int = 30):
        """Initialize the TimeoutHandlingLLM with retry and timeout settings.
        
        Args:
            max_retries: Maximum number of retry attempts.
            timeout: Timeout in seconds for each API call.
        """
        super().__init__()
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
        self.calls.append({
            "messages": messages, 
            "tools": tools,
            "callbacks": callbacks,
            "available_functions": available_functions,
            "attempt": 0
        })
        
        # Simulate retry logic
        for attempt in range(self.max_retries):
            # Skip the first attempt recording since we already did that above
            if attempt == 0:
                # Simulate a failure if fail_count > 0
                if self.fail_count > 0:
                    self.fail_count -= 1
                    # If we've used all retries, raise an error
                    if attempt == self.max_retries - 1:
                        raise TimeoutError(f"LLM request failed after {self.max_retries} attempts")
                    # Otherwise, continue to the next attempt (simulating backoff)
                    continue
                else:
                    # Success on first attempt
                    return "First attempt response"
            else:
                # This is a retry attempt (attempt > 0)
                # Always record retry attempts
                self.calls.append({
                    "retry_attempt": attempt,
                    "messages": messages,
                    "tools": tools,
                    "callbacks": callbacks,
                    "available_functions": available_functions
                })
                
                # Simulate a failure if fail_count > 0
                if self.fail_count > 0:
                    self.fail_count -= 1
                    # If we've used all retries, raise an error
                    if attempt == self.max_retries - 1:
                        raise TimeoutError(f"LLM request failed after {self.max_retries} attempts")
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
