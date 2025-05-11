# Multiple Model Configuration in CrewAI

CrewAI now supports configuring multiple language models with different API keys and configurations. This feature allows you to:

1. Load-balance across multiple model deployments
2. Set up fallback models in case of rate limits or errors
3. Configure different routing strategies for model selection
4. Maintain fine-grained control over model selection and usage

## Basic Usage

You can configure multiple models at the agent level:

```python
from crewai import Agent

# Define model configurations
model_list = [
    {
        "model_name": "gpt-4o-mini",
        "litellm_params": {
            "model": "gpt-4o-mini",  # Required: model name must be specified here
            "api_key": "your-openai-api-key-1"
        }
    },
    {
        "model_name": "gpt-3.5-turbo",
        "litellm_params": {
            "model": "gpt-3.5-turbo",  # Required: model name must be specified here
            "api_key": "your-openai-api-key-2"
        }
    },
    {
        "model_name": "claude-3-sonnet-20240229",
        "litellm_params": {
            "model": "claude-3-sonnet-20240229",  # Required: model name must be specified here
            "api_key": "your-anthropic-api-key"
        }
    }
]

# Create an agent with multiple model configurations
agent = Agent(
    role="Data Analyst",
    goal="Analyze the data and provide insights",
    backstory="You are an expert data analyst with years of experience.",
    model_list=model_list,
    routing_strategy="simple-shuffle"  # Optional routing strategy
)
```

## Routing Strategies

CrewAI supports the following routing strategies for precise control over model selection:

- `simple-shuffle`: Randomly selects a model from the list
- `least-busy`: Routes to the model with the least number of ongoing requests
- `usage-based`: Routes based on token usage across models
- `latency-based`: Routes to the model with the lowest latency
- `cost-based`: Routes to the model with the lowest cost

Example with latency-based routing:

```python
agent = Agent(
    role="Data Analyst",
    goal="Analyze the data and provide insights",
    backstory="You are an expert data analyst with years of experience.",
    model_list=model_list,
    routing_strategy="latency-based"
)
```

## Direct LLM Configuration

You can also configure multiple models directly with the LLM class for more flexibility:

```python
from crewai import LLM

llm = LLM(
    model="gpt-4o-mini",
    model_list=model_list,
    routing_strategy="simple-shuffle"
)
```

## Advanced Configuration

For more advanced configurations, you can specify additional parameters for each model to handle complex use cases:

```python
model_list = [
    {
        "model_name": "gpt-4o-mini",
        "litellm_params": {
            "model": "gpt-4o-mini",  # Required: model name must be specified here
            "api_key": "your-openai-api-key-1",
            "temperature": 0.7
        },
        "tpm": 100000,  # Tokens per minute limit
        "rpm": 1000     # Requests per minute limit
    },
    {
        "model_name": "gpt-3.5-turbo",
        "litellm_params": {
            "model": "gpt-3.5-turbo",  # Required: model name must be specified here
            "api_key": "your-openai-api-key-2",
            "temperature": 0.5
        }
    }
]
```

## Error Handling and Troubleshooting

When working with multiple model configurations, you may encounter various issues. Here are some common problems and their solutions:

### Missing Required Parameters

**Problem**: Router initialization fails with an error about missing parameters.

**Solution**: Ensure each model configuration in `model_list` includes both `model_name` and `litellm_params` with the required `model` parameter:

```python
# Correct configuration
model_config = {
    "model_name": "gpt-4o-mini",  # Required
    "litellm_params": {
        "model": "gpt-4o-mini",   # Required
        "api_key": "your-api-key"
    }
}
```

### Invalid Routing Strategy

**Problem**: Error when specifying an unsupported routing strategy.

**Solution**: Use only the supported routing strategies:

```python
# Valid routing strategies
valid_strategies = [
    "simple-shuffle", 
    "least-busy", 
    "usage-based", 
    "latency-based", 
    "cost-based"
]
```

### API Key Authentication Errors

**Problem**: Authentication errors when making API calls.

**Solution**: Verify that all API keys are valid and have the necessary permissions:

```python
# Check environment variables first
import os
os.environ.get("OPENAI_API_KEY")  # Should be set if using OpenAI models

# Or explicitly provide in the configuration
model_list = [{
    "model_name": "gpt-4o-mini",
    "litellm_params": {
        "model": "gpt-4o-mini",
        "api_key": "valid-api-key-here"  # Ensure this is correct
    }
}]
```

### Rate Limit Handling

**Problem**: Encountering rate limits with multiple models.

**Solution**: Configure rate limits and implement fallback mechanisms:

```python
model_list = [
    {
        "model_name": "primary-model",
        "litellm_params": {"model": "primary-model", "api_key": "key1"},
        "rpm": 100  # Requests per minute
    },
    {
        "model_name": "fallback-model",
        "litellm_params": {"model": "fallback-model", "api_key": "key2"}
    }
]

# Configure with fallback
llm = LLM(
    model="primary-model",
    model_list=model_list,
    routing_strategy="least-busy"  # Will route to fallback when primary is busy
)
```

### Debugging Router Issues

If you're experiencing issues with the router, you can enable verbose logging to get more information:

```python
import litellm
litellm.set_verbose = True

# Then initialize your LLM
llm = LLM(model="gpt-4o-mini", model_list=model_list)
```

This feature leverages litellm's Router functionality under the hood, providing robust load balancing and fallback capabilities for your CrewAI agents. The implementation ensures predictability and consistency in model selection while maintaining security through proper API key management.
