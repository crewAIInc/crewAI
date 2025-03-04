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
        # Implement your own logic to call the LLM
        # For example, using requests:
        import requests
        
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
        
        response = requests.post(self.endpoint, headers=headers, json=data)
        return response.json()["choices"][0]["message"]["content"]
        
    def supports_function_calling(self) -> bool:
        # Return True if your LLM supports function calling
        return True
        
    def supports_stop_words(self) -> bool:
        # Return True if your LLM supports stop words
        return True
        
    def get_context_window_size(self) -> int:
        # Return the context window size of your LLM
        return 8192
```

## Example: JWT-based Authentication

For services that use JWT-based authentication instead of API keys, you can implement a custom LLM like this:

```python
from crewai import BaseLLM, Agent, Task
from typing import Any, Dict, List, Optional, Union

class JWTAuthLLM(BaseLLM):
    def __init__(self, jwt_token: str, endpoint: str):
        super().__init__()  # Initialize the base class to set default attributes
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
        # Implement your own logic to call the LLM with JWT authentication
        import requests
        
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
        
        response = requests.post(self.endpoint, headers=headers, json=data)
        return response.json()["choices"][0]["message"]["content"]
        
    def supports_function_calling(self) -> bool:
        # Return True if your LLM supports function calling
        return True
        
    def supports_stop_words(self) -> bool:
        # Return True if your LLM supports stop words
        return True
        
    def get_context_window_size(self) -> int:
        # Return the context window size of your LLM
        return 8192
```

## Using Your Custom LLM with CrewAI

Once you've implemented your custom LLM, you can use it with CrewAI agents and crews:

```python
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
```

## Advanced Usage: Function Calling

If your LLM supports function calling, you can implement the function calling logic in your custom LLM:

```python
def call(
    self,
    messages: Union[str, List[Dict[str, str]]],
    tools: Optional[List[dict]] = None,
    callbacks: Optional[List[Any]] = None,
    available_functions: Optional[Dict[str, Any]] = None,
) -> Union[str, Any]:
    import requests
    
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
    
    response = requests.post(self.endpoint, headers=headers, json=data)
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
```

## Implementing Your Own Authentication Mechanism

The `BaseLLM` class allows you to implement any authentication mechanism you need, not just JWT or API keys. You can use:

- OAuth tokens
- Client certificates
- Custom headers
- Session-based authentication
- Any other authentication method required by your LLM provider

Simply implement the appropriate authentication logic in your custom LLM class.
