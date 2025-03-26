# Custom LLM Implementations

CrewAI now supports custom LLM implementations through the `BaseLLM` abstract base class. This allows you to create your own LLM implementations that don't rely on litellm's authentication mechanism.

## Using Custom LLM Implementations

To create a custom LLM implementation, you need to:

1. Inherit from the `BaseLLM` abstract base class
2. Implement the required methods:
   - `call()`: The main method to call the LLM with messages
   - `supports_function_calling()`: Whether the LLM supports function calling
   - `supports_stop_words()`: Whether the LLM supports stop words
   - `get_context_window_size()`: The context window size of the LLM

## Example: Basic Custom LLM

```python
from crewai import BaseLLM
from typing import Any, Dict, List, Optional, Union

class CustomLLM(BaseLLM):
    def __init__(self, api_key: str, endpoint: str):
        super().__init__()  # Initialize the base class to set default attributes
        if not api_key or not isinstance(api_key, str):
            raise ValueError("Invalid API key: must be a non-empty string")
        if not endpoint or not isinstance(endpoint, str):
            raise ValueError("Invalid endpoint URL: must be a non-empty string")
        self.api_key = api_key
        self.endpoint = endpoint
        self.stop = []  # You can customize stop words if needed
        
    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Any]:
        """Call the LLM with the given messages.
        
        Args:
            messages: Input messages for the LLM.
            tools: Optional list of tool schemas for function calling.
            callbacks: Optional list of callback functions.
            available_functions: Optional dict mapping function names to callables.
            
        Returns:
            Either a text response from the LLM or the result of a tool function call.
            
        Raises:
            TimeoutError: If the LLM request times out.
            RuntimeError: If the LLM request fails for other reasons.
            ValueError: If the response format is invalid.
        """
        # Implement your own logic to call the LLM
        # For example, using requests:
        import requests
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Convert string message to proper format if needed
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            
            data = {
                "messages": messages,
                "tools": tools
            }
            
            response = requests.post(
                self.endpoint, 
                headers=headers, 
                json=data,
                timeout=30  # Set a reasonable timeout
            )
            response.raise_for_status()  # Raise an exception for HTTP errors
            return response.json()["choices"][0]["message"]["content"]
        except requests.Timeout:
            raise TimeoutError("LLM request timed out")
        except requests.RequestException as e:
            raise RuntimeError(f"LLM request failed: {str(e)}")
        except (KeyError, IndexError, ValueError) as e:
            raise ValueError(f"Invalid response format: {str(e)}")
        
    def supports_function_calling(self) -> bool:
        """Check if the LLM supports function calling.
        
        Returns:
            True if the LLM supports function calling, False otherwise.
        """
        # Return True if your LLM supports function calling
        return True
        
    def supports_stop_words(self) -> bool:
        """Check if the LLM supports stop words.
        
        Returns:
            True if the LLM supports stop words, False otherwise.
        """
        # Return True if your LLM supports stop words
        return True
        
    def get_context_window_size(self) -> int:
        """Get the context window size of the LLM.
        
        Returns:
            The context window size as an integer.
        """
        # Return the context window size of your LLM
        return 8192
```

## Error Handling Best Practices

When implementing custom LLMs, it's important to handle errors properly to ensure robustness and reliability. Here are some best practices:

### 1. Implement Try-Except Blocks for API Calls

Always wrap API calls in try-except blocks to handle different types of errors:

```python
def call(
    self,
    messages: Union[str, List[Dict[str, str]]],
    tools: Optional[List[dict]] = None,
    callbacks: Optional[List[Any]] = None,
    available_functions: Optional[Dict[str, Any]] = None,
) -> Union[str, Any]:
    try:
        # API call implementation
        response = requests.post(
            self.endpoint,
            headers=self.headers,
            json=self.prepare_payload(messages),
            timeout=30  # Set a reasonable timeout
        )
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()["choices"][0]["message"]["content"]
    except requests.Timeout:
        raise TimeoutError("LLM request timed out")
    except requests.RequestException as e:
        raise RuntimeError(f"LLM request failed: {str(e)}")
    except (KeyError, IndexError, ValueError) as e:
        raise ValueError(f"Invalid response format: {str(e)}")
```

### 2. Implement Retry Logic for Transient Failures

For transient failures like network issues or rate limiting, implement retry logic with exponential backoff:

```python
def call(
    self,
    messages: Union[str, List[Dict[str, str]]],
    tools: Optional[List[dict]] = None,
    callbacks: Optional[List[Any]] = None,
    available_functions: Optional[Dict[str, Any]] = None,
) -> Union[str, Any]:
    import time
    
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                self.endpoint,
                headers=self.headers,
                json=self.prepare_payload(messages),
                timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except (requests.Timeout, requests.ConnectionError) as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                continue
            raise TimeoutError(f"LLM request failed after {max_retries} attempts: {str(e)}")
        except requests.RequestException as e:
            raise RuntimeError(f"LLM request failed: {str(e)}")
```

### 3. Validate Input Parameters

Always validate input parameters to prevent runtime errors:

```python
def __init__(self, api_key: str, endpoint: str):
    super().__init__()
    if not api_key or not isinstance(api_key, str):
        raise ValueError("Invalid API key: must be a non-empty string")
    if not endpoint or not isinstance(endpoint, str):
        raise ValueError("Invalid endpoint URL: must be a non-empty string")
    self.api_key = api_key
    self.endpoint = endpoint
```

### 4. Handle Authentication Errors Gracefully

Provide clear error messages for authentication failures:

```python
def call(
    self,
    messages: Union[str, List[Dict[str, str]]],
    tools: Optional[List[dict]] = None,
    callbacks: Optional[List[Any]] = None,
    available_functions: Optional[Dict[str, Any]] = None,
) -> Union[str, Any]:
    try:
        response = requests.post(self.endpoint, headers=self.headers, json=data)
        if response.status_code == 401:
            raise ValueError("Authentication failed: Invalid API key or token")
        elif response.status_code == 403:
            raise ValueError("Authorization failed: Insufficient permissions")
        response.raise_for_status()
        # Process response
    except Exception as e:
        # Handle error
        raise
```

## Example: JWT-based Authentication

For services that use JWT-based authentication instead of API keys, you can implement a custom LLM like this:

```python
from crewai import BaseLLM, Agent, Task
from typing import Any, Dict, List, Optional, Union

class JWTAuthLLM(BaseLLM):
    def __init__(self, jwt_token: str, endpoint: str):
        super().__init__()  # Initialize the base class to set default attributes
        if not jwt_token or not isinstance(jwt_token, str):
            raise ValueError("Invalid JWT token: must be a non-empty string")
        if not endpoint or not isinstance(endpoint, str):
            raise ValueError("Invalid endpoint URL: must be a non-empty string")
        self.jwt_token = jwt_token
        self.endpoint = endpoint
        self.stop = []  # You can customize stop words if needed
        
    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Any]:
        """Call the LLM with JWT authentication.
        
        Args:
            messages: Input messages for the LLM.
            tools: Optional list of tool schemas for function calling.
            callbacks: Optional list of callback functions.
            available_functions: Optional dict mapping function names to callables.
            
        Returns:
            Either a text response from the LLM or the result of a tool function call.
            
        Raises:
            TimeoutError: If the LLM request times out.
            RuntimeError: If the LLM request fails for other reasons.
            ValueError: If the response format is invalid.
        """
        # Implement your own logic to call the LLM with JWT authentication
        import requests
        
        try:
            headers = {
                "Authorization": f"Bearer {self.jwt_token}",
                "Content-Type": "application/json"
            }
            
            # Convert string message to proper format if needed
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            
            data = {
                "messages": messages,
                "tools": tools
            }
            
            response = requests.post(
                self.endpoint,
                headers=headers,
                json=data,
                timeout=30  # Set a reasonable timeout
            )
            
            if response.status_code == 401:
                raise ValueError("Authentication failed: Invalid JWT token")
            elif response.status_code == 403:
                raise ValueError("Authorization failed: Insufficient permissions")
                
            response.raise_for_status()  # Raise an exception for HTTP errors
            return response.json()["choices"][0]["message"]["content"]
        except requests.Timeout:
            raise TimeoutError("LLM request timed out")
        except requests.RequestException as e:
            raise RuntimeError(f"LLM request failed: {str(e)}")
        except (KeyError, IndexError, ValueError) as e:
            raise ValueError(f"Invalid response format: {str(e)}")
        
    def supports_function_calling(self) -> bool:
        """Check if the LLM supports function calling.
        
        Returns:
            True if the LLM supports function calling, False otherwise.
        """
        return True
        
    def supports_stop_words(self) -> bool:
        """Check if the LLM supports stop words.
        
        Returns:
            True if the LLM supports stop words, False otherwise.
        """
        return True
        
    def get_context_window_size(self) -> int:
        """Get the context window size of the LLM.
        
        Returns:
            The context window size as an integer.
        """
        return 8192
```

## Troubleshooting

Here are some common issues you might encounter when implementing custom LLMs and how to resolve them:

### 1. Authentication Failures

**Symptoms**: 401 Unauthorized or 403 Forbidden errors

**Solutions**:
- Verify that your API key or JWT token is valid and not expired
- Check that you're using the correct authentication header format
- Ensure that your token has the necessary permissions

### 2. Timeout Issues

**Symptoms**: Requests taking too long or timing out

**Solutions**:
- Implement timeout handling as shown in the examples
- Use retry logic with exponential backoff
- Consider using a more reliable network connection

### 3. Response Parsing Errors

**Symptoms**: KeyError, IndexError, or ValueError when processing responses

**Solutions**:
- Validate the response format before accessing nested fields
- Implement proper error handling for malformed responses
- Check the API documentation for the expected response format

### 4. Rate Limiting

**Symptoms**: 429 Too Many Requests errors

**Solutions**:
- Implement rate limiting in your custom LLM
- Add exponential backoff for retries
- Consider using a token bucket algorithm for more precise rate control

## Advanced Features

### Logging

Adding logging to your custom LLM can help with debugging and monitoring:

```python
import logging
from typing import Any, Dict, List, Optional, Union

class LoggingLLM(BaseLLM):
    def __init__(self, api_key: str, endpoint: str):
        super().__init__()
        self.api_key = api_key
        self.endpoint = endpoint
        self.logger = logging.getLogger("crewai.llm.custom")
        
    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Any]:
        self.logger.info(f"Calling LLM with {len(messages) if isinstance(messages, list) else 1} messages")
        try:
            # API call implementation
            response = self._make_api_call(messages, tools)
            self.logger.debug(f"LLM response received: {response[:100]}...")
            return response
        except Exception as e:
            self.logger.error(f"LLM call failed: {str(e)}")
            raise
```

### Rate Limiting

Implementing rate limiting can help avoid overwhelming the LLM API:

```python
import time
from typing import Any, Dict, List, Optional, Union

class RateLimitedLLM(BaseLLM):
    def __init__(
        self, 
        api_key: str, 
        endpoint: str, 
        requests_per_minute: int = 60
    ):
        super().__init__()
        self.api_key = api_key
        self.endpoint = endpoint
        self.requests_per_minute = requests_per_minute
        self.request_times: List[float] = []
        
    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Any]:
        self._enforce_rate_limit()
        # Record this request time
        self.request_times.append(time.time())
        # Make the actual API call
        return self._make_api_call(messages, tools)
        
    def _enforce_rate_limit(self) -> None:
        """Enforce the rate limit by waiting if necessary."""
        now = time.time()
        # Remove request times older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        if len(self.request_times) >= self.requests_per_minute:
            # Calculate how long to wait
            oldest_request = min(self.request_times)
            wait_time = 60 - (now - oldest_request)
            if wait_time > 0:
                time.sleep(wait_time)
```

### Metrics Collection

Collecting metrics can help you monitor your LLM usage:

```python
import time
from typing import Any, Dict, List, Optional, Union

class MetricsCollectingLLM(BaseLLM):
    def __init__(self, api_key: str, endpoint: str):
        super().__init__()
        self.api_key = api_key
        self.endpoint = endpoint
        self.metrics: Dict[str, Any] = {
            "total_calls": 0,
            "total_tokens": 0,
            "errors": 0,
            "latency": []
        }
        
    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Any]:
        start_time = time.time()
        self.metrics["total_calls"] += 1
        
        try:
            response = self._make_api_call(messages, tools)
            # Estimate tokens (simplified)
            if isinstance(messages, str):
                token_estimate = len(messages) // 4
            else:
                token_estimate = sum(len(m.get("content", "")) // 4 for m in messages)
            self.metrics["total_tokens"] += token_estimate
            return response
        except Exception as e:
            self.metrics["errors"] += 1
            raise
        finally:
            latency = time.time() - start_time
            self.metrics["latency"].append(latency)
            
    def get_metrics(self) -> Dict[str, Any]:
        """Return the collected metrics."""
        avg_latency = sum(self.metrics["latency"]) / len(self.metrics["latency"]) if self.metrics["latency"] else 0
        return {
            **self.metrics,
            "avg_latency": avg_latency
        }
```

## Advanced Usage: Function Calling

If your LLM supports function calling, you can implement the function calling logic in your custom LLM:

```python
import json
from typing import Any, Dict, List, Optional, Union

def call(
    self,
    messages: Union[str, List[Dict[str, str]]],
    tools: Optional[List[dict]] = None,
    callbacks: Optional[List[Any]] = None,
    available_functions: Optional[Dict[str, Any]] = None,
) -> Union[str, Any]:
    import requests
    
    try:
        headers = {
            "Authorization": f"Bearer {self.jwt_token}",
            "Content-Type": "application/json"
        }
        
        # Convert string message to proper format if needed
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        data = {
            "messages": messages,
            "tools": tools
        }
        
        response = requests.post(
            self.endpoint,
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        response_data = response.json()
        
        # Check if the LLM wants to call a function
        if response_data["choices"][0]["message"].get("tool_calls"):
            tool_calls = response_data["choices"][0]["message"]["tool_calls"]
            
            # Process each tool call
            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                function_args = json.loads(tool_call["function"]["arguments"])
                
                if available_functions and function_name in available_functions:
                    function_to_call = available_functions[function_name]
                    function_response = function_to_call(**function_args)
                    
                    # Add the function response to the messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": function_name,
                        "content": str(function_response)
                    })
            
            # Call the LLM again with the updated messages
            return self.call(messages, tools, callbacks, available_functions)
        
        # Return the text response if no function call
        return response_data["choices"][0]["message"]["content"]
    except requests.Timeout:
        raise TimeoutError("LLM request timed out")
    except requests.RequestException as e:
        raise RuntimeError(f"LLM request failed: {str(e)}")
    except (KeyError, IndexError, ValueError) as e:
        raise ValueError(f"Invalid response format: {str(e)}")
```

## Using Your Custom LLM with CrewAI

Once you've implemented your custom LLM, you can use it with CrewAI agents and crews:

```python
from crewai import Agent, Task, Crew
from typing import Dict, Any

# Create your custom LLM instance
jwt_llm = JWTAuthLLM(
    jwt_token="your.jwt.token", 
    endpoint="https://your-llm-endpoint.com/v1/chat/completions"
)

# Use it with an agent
agent = Agent(
    role="Research Assistant",
    goal="Find information on a topic",
    backstory="You are a research assistant tasked with finding information.",
    llm=jwt_llm,
)

# Create a task for the agent
task = Task(
    description="Research the benefits of exercise",
    agent=agent,
    expected_output="A summary of the benefits of exercise",
)

# Execute the task
result = agent.execute_task(task)
print(result)

# Or use it with a crew
crew = Crew(
    agents=[agent],
    tasks=[task],
    manager_llm=jwt_llm,  # Use your custom LLM for the manager
)

# Run the crew
result = crew.kickoff()
print(result)
```

## Implementing Your Own Authentication Mechanism

The `BaseLLM` class allows you to implement any authentication mechanism you need, not just JWT or API keys. You can use:

- OAuth tokens
- Client certificates
- Custom headers
- Session-based authentication
- Any other authentication method required by your LLM provider

Simply implement the appropriate authentication logic in your custom LLM class.
