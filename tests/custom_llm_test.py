import pytest
from typing import Any, Dict, List, Optional, Union

from crewai.llm import BaseLLM
from crewai.utilities.llm_utils import create_llm


class CustomLLM(BaseLLM):
    """Custom LLM implementation for testing."""
    
    def __init__(self, response: str = "Custom LLM response"):
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
        """Record the call and return the predefined response."""
        self.calls.append({
            "messages": messages, 
            "tools": tools,
            "callbacks": callbacks,
            "available_functions": available_functions
        })
        return self.response
        
    def supports_function_calling(self) -> bool:
        """Return True to indicate that function calling is supported."""
        return True
        
    def supports_stop_words(self) -> bool:
        """Return True to indicate that stop words are supported."""
        return True
        
    def get_context_window_size(self) -> int:
        """Return a default context window size."""
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


class JWTAuthLLM(BaseLLM):
    def __init__(self, jwt_token: str):
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
        return True
        
    def supports_stop_words(self) -> bool:
        return True
        
    def get_context_window_size(self) -> int:
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
