# OAuth2 LLM Providers

CrewAI supports OAuth2 authentication for custom LiteLLM providers through configuration files.

## Configuration

Create a `litellm_config.json` file in your project directory:

```json
{
  "oauth2_providers": {
    "my_custom_provider": {
      "client_id": "your_client_id",
      "client_secret": "your_client_secret", 
      "token_url": "https://your-provider.com/oauth/token",
      "scope": "llm.read llm.write"
    },
    "another_provider": {
      "client_id": "another_client_id",
      "client_secret": "another_client_secret",
      "token_url": "https://another-provider.com/token"
    }
  }
}
```

## Usage

```python
from crewai import LLM

# Initialize LLM with OAuth2 support
llm = LLM(
    model="my_custom_provider/my-model",
    oauth2_config_path="./litellm_config.json"  # Optional, defaults to ./litellm_config.json
)

# Use in CrewAI
from crewai import Agent, Task, Crew

agent = Agent(
    role="Data Analyst",
    goal="Analyze data trends",
    backstory="Expert in data analysis",
    llm=llm
)

task = Task(
    description="Analyze the latest sales data",
    agent=agent
)

crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()
```

## Environment Variables

You can also use environment variables in your configuration:

```json
{
  "oauth2_providers": {
    "my_provider": {
      "client_id": "os.environ/MY_CLIENT_ID",
      "client_secret": "os.environ/MY_CLIENT_SECRET",
      "token_url": "https://my-provider.com/token"
    }
  }
}
```

## Supported OAuth2 Flow

Currently supports the **Client Credentials** OAuth2 flow, which is suitable for server-to-server authentication.

## Token Management

- Tokens are automatically cached and refreshed when they expire
- A 60-second buffer is used before token expiration to ensure reliability
- Failed token acquisition will raise a `RuntimeError` with details

## Configuration Schema

The `litellm_config.json` file should follow this schema:

```json
{
  "oauth2_providers": {
    "<provider_name>": {
      "client_id": "string (required)",
      "client_secret": "string (required)",
      "token_url": "string (required)",
      "scope": "string (optional)",
      "refresh_token": "string (optional)"
    }
  }
}
```

## Error Handling

- If OAuth2 authentication fails, a `RuntimeError` will be raised with details
- Invalid configuration files will raise a `ValueError` with specifics
- Network errors during token acquisition are wrapped in `RuntimeError`

## Examples

### Basic OAuth2 Provider

```python
from crewai import LLM

llm = LLM(
    model="my_provider/gpt-4",
    oauth2_config_path="./config.json"
)

response = llm.call("Hello, world!")
print(response)
```

### Multiple Providers

```json
{
  "oauth2_providers": {
    "provider_a": {
      "client_id": "client_a",
      "client_secret": "secret_a",
      "token_url": "https://provider-a.com/token"
    },
    "provider_b": {
      "client_id": "client_b", 
      "client_secret": "secret_b",
      "token_url": "https://provider-b.com/oauth/token",
      "scope": "read write"
    }
  }
}
```

```python
# Use different providers
llm_a = LLM(model="provider_a/model-1", oauth2_config_path="./config.json")
llm_b = LLM(model="provider_b/model-2", oauth2_config_path="./config.json")
```
