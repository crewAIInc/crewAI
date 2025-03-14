from collections import deque
from typing import Any, Dict, List, Optional, Union
import time

import jwt
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
    """Custom LLM implementation with JWT authentication.
    
    This class demonstrates how to implement a custom LLM that uses JWT
    authentication instead of API key-based authentication. It validates
    the JWT token before each call and checks for token expiration.
    """
    
    def __init__(self, jwt_token: str, expiration_buffer: int = 60):
        """Initialize the JWTAuthLLM with a JWT token.
        
        Args:
            jwt_token: The JWT token to use for authentication.
            expiration_buffer: Buffer time in seconds to warn about token expiration.
                               Default is 60 seconds.
                               
        Raises:
            ValueError: If the JWT token is invalid or missing.
        """
        super().__init__()
        if not jwt_token or not isinstance(jwt_token, str):
            raise ValueError("Invalid JWT token")
        
        self.jwt_token = jwt_token
        self.expiration_buffer = expiration_buffer
        self.calls = []
        self.stop = []
        
        # Validate the token immediately
        self._validate_token()
    
    def _validate_token(self) -> None:
        """Validate the JWT token.
        
        Checks if the token is valid and not expired. Also warns if the token
        is about to expire within the expiration_buffer time.
        
        Raises:
            ValueError: If the token is invalid, expired, or malformed.
        """
        try:
            # Decode without verification to check expiration
            # In a real implementation, you would verify the signature
            decoded = jwt.decode(self.jwt_token, options={"verify_signature": False})
            
            # Check if token is expired or about to expire
            if 'exp' in decoded:
                expiration_time = decoded['exp']
                current_time = time.time()
                
                if expiration_time < current_time:
                    raise ValueError("JWT token has expired")
                
                if expiration_time < current_time + self.expiration_buffer:
                    # Token will expire soon, log a warning
                    import logging
                    logging.warning(f"JWT token will expire in {expiration_time - current_time} seconds")
        except jwt.PyJWTError as e:
            raise ValueError(f"Invalid JWT token format: {str(e)}")
        
    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Any]:
        """Call the LLM with JWT authentication.
        
        Validates the JWT token before making the call to ensure it's still valid.
        
        Args:
            messages: Input messages for the LLM.
            tools: Optional list of tool schemas for function calling.
            callbacks: Optional list of callback functions.
            available_functions: Optional dict mapping function names to callables.
            
        Returns:
            The LLM response.
            
        Raises:
            ValueError: If the JWT token is invalid or expired.
            TimeoutError: If the request times out.
            ConnectionError: If there's a connection issue.
        """
        # Validate token before making the call
        self._validate_token()
        
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
    # Create a valid JWT token that expires 1 hour from now
    valid_token = jwt.encode(
        {"exp": int(time.time()) + 3600},
        "secret",
        algorithm="HS256"
    )
    
    jwt_llm = JWTAuthLLM(jwt_token=valid_token)
    
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
        
    # Test with expired token
    # Create a token that expired 1 hour ago
    expired_token = jwt.encode(
        {"exp": int(time.time()) - 3600},
        "secret",
        algorithm="HS256"
    )
    with pytest.raises(ValueError, match="JWT token has expired"):
        JWTAuthLLM(jwt_token=expired_token)
        
    # Test with malformed token
    with pytest.raises(ValueError, match="Invalid JWT token format"):
        JWTAuthLLM(jwt_token="not.a.valid.jwt.token")
        
    # Test with valid token
    # Create a token that expires 1 hour from now
    valid_token = jwt.encode(
        {"exp": int(time.time()) + 3600},
        "secret",
        algorithm="HS256"
    )
    # This should not raise an exception
    jwt_llm = JWTAuthLLM(jwt_token=valid_token)
    assert jwt_llm.jwt_token == valid_token


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


def test_rate_limited_llm():
    """Test that rate limiting works correctly."""
    # Create a rate limited LLM with a very low limit (2 requests per minute)
    llm = RateLimitedLLM(requests_per_minute=2)
    
    # First request should succeed
    response1 = llm.call("Test message 1")
    assert response1 == "Rate limited response"
    assert len(llm.calls) == 1
    
    # Second request should succeed
    response2 = llm.call("Test message 2")
    assert response2 == "Rate limited response"
    assert len(llm.calls) == 2
    
    # Third request should fail due to rate limiting
    with pytest.raises(ValueError, match="Rate limit exceeded"):
        llm.call("Test message 3")
    
    # Test with invalid requests_per_minute
    with pytest.raises(ValueError, match="requests_per_minute must be a positive integer"):
        RateLimitedLLM(requests_per_minute=0)
    
    with pytest.raises(ValueError, match="requests_per_minute must be a positive integer"):
        RateLimitedLLM(requests_per_minute=-1)


def test_rate_limit_reset():
    """Test that rate limits reset after the time window passes."""
    # Create a rate limited LLM with a very low limit (1 request per minute)
    # and a short time window for testing (1 second instead of 60 seconds)
    time_window = 1  # 1 second instead of 60 seconds
    llm = RateLimitedLLM(requests_per_minute=1, time_window=time_window)
    
    # First request should succeed
    response1 = llm.call("Test message 1")
    assert response1 == "Rate limited response"
    
    # Second request should fail due to rate limiting
    with pytest.raises(ValueError, match="Rate limit exceeded"):
        llm.call("Test message 2")
    
    # Wait for the rate limit to reset
    import time
    time.sleep(time_window + 0.1)  # Add a small buffer
    
    # After waiting, we should be able to make another request
    response3 = llm.call("Test message 3")
    assert response3 == "Rate limited response"
    assert len(llm.calls) == 2  # First and third requests


class RateLimitedLLM(LLM):
    """Custom LLM implementation with rate limiting.
    
    This class demonstrates how to implement a custom LLM with rate limiting
    capabilities. It uses a sliding window algorithm to ensure that no more
    than a specified number of requests are made within a given time period.
    """
    
    def __init__(self, requests_per_minute: int = 60, base_response: str = "Rate limited response", time_window: int = 60):
        """Initialize the RateLimitedLLM with rate limiting parameters.
        
        Args:
            requests_per_minute: Maximum number of requests allowed per minute.
            base_response: Default response to return.
            time_window: Time window in seconds for rate limiting (default: 60).
                         This is configurable for testing purposes.
            
        Raises:
            ValueError: If requests_per_minute is not a positive integer.
        """
        super().__init__()
        if not isinstance(requests_per_minute, int) or requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be a positive integer")
            
        self.requests_per_minute = requests_per_minute
        self.base_response = base_response
        self.time_window = time_window
        self.request_times = deque()
        self.calls = []
        self.stop = []
        
    def _check_rate_limit(self) -> None:
        """Check if the current request exceeds the rate limit.
        
        This method implements a sliding window rate limiting algorithm.
        It keeps track of request timestamps and ensures that no more than
        `requests_per_minute` requests are made within the configured time window.
        
        Raises:
            ValueError: If the rate limit is exceeded.
        """
        current_time = time.time()
        
        # Remove requests older than the time window
        while self.request_times and current_time - self.request_times[0] > self.time_window:
            self.request_times.popleft()
        
        # Check if we've exceeded the rate limit
        if len(self.request_times) >= self.requests_per_minute:
            wait_time = self.time_window - (current_time - self.request_times[0])
            raise ValueError(
                f"Rate limit exceeded. Maximum {self.requests_per_minute} "
                f"requests per {self.time_window} seconds. Try again in {wait_time:.2f} seconds."
            )
        
        # Record this request
        self.request_times.append(current_time)
        
    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Any]:
        """Call the LLM with rate limiting.
        
        Args:
            messages: Input messages for the LLM.
            tools: Optional list of tool schemas for function calling.
            callbacks: Optional list of callback functions.
            available_functions: Optional dict mapping function names to callables.
            
        Returns:
            The LLM response.
            
        Raises:
            ValueError: If the rate limit is exceeded.
        """
        # Check rate limit before making the call
        self._check_rate_limit()
        
        self.calls.append({
            "messages": messages, 
            "tools": tools,
            "callbacks": callbacks,
            "available_functions": available_functions
        })
        
        return self.base_response
        
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
