# AI/ML API Integration with CrewAI

CrewAI now supports AI/ML API as a provider, giving you access to 300+ AI models through their platform. AI/ML API provides a unified interface to models from various providers including Meta (Llama), Anthropic (Claude), Mistral, Qwen, and more.

## Setup

1. Get your API key from [AI/ML API](https://aimlapi.com)
2. Set your API key as an environment variable:

```bash
export AIML_API_KEY="your-api-key-here"
```

## Usage

AI/ML API models use the `openai/` prefix for compatibility with LiteLLM. Here are some examples:

### Basic Usage

```python
from crewai import Agent, LLM

# Use Llama 3.1 70B through AI/ML API
llm = LLM(
    model="openai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    api_key="your-aiml-api-key"  # or set AIML_API_KEY env var
)

agent = Agent(
    role="Research Assistant",
    goal="Help with research tasks",
    backstory="You are an expert researcher with access to advanced AI capabilities",
    llm=llm
)
```

### Available Models

Popular models available through AI/ML API:

#### Llama Models
- `openai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo` - Largest Llama model
- `openai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo` - High performance
- `openai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` - Fast and efficient
- `openai/meta-llama/Meta-Llama-3.2-90B-Vision-Instruct-Turbo` - Vision capabilities

#### Claude Models
- `openai/anthropic/claude-3-5-sonnet-20241022` - Latest Claude Sonnet
- `openai/anthropic/claude-3-5-haiku-20241022` - Fast Claude model
- `openai/anthropic/claude-3-opus-20240229` - Most capable Claude

#### Other Models
- `openai/mistralai/Mixtral-8x7B-Instruct-v0.1` - Mistral's mixture of experts
- `openai/Qwen/Qwen2.5-72B-Instruct-Turbo` - Qwen's large model
- `openai/deepseek-ai/DeepSeek-V2.5` - DeepSeek's latest model

### Complete Example

```python
from crewai import Agent, Task, Crew, LLM

# Configure AI/ML API LLM
llm = LLM(
    model="openai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    api_key="your-aiml-api-key"
)

# Create an agent with AI/ML API model
researcher = Agent(
    role="AI Research Specialist",
    goal="Analyze AI trends and provide insights",
    backstory="You are an expert in artificial intelligence with deep knowledge of current trends and developments",
    llm=llm
)

# Create a task
research_task = Task(
    description="Research the latest developments in large language models and summarize key findings",
    expected_output="A comprehensive summary of recent LLM developments with key insights",
    agent=researcher
)

# Create and run the crew
crew = Crew(
    agents=[researcher],
    tasks=[research_task]
)

result = crew.kickoff()
print(result)
```

### Environment Configuration

You can configure AI/ML API in several ways:

```python
# Method 1: Direct API key
llm = LLM(
    model="openai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    api_key="your-aiml-api-key"
)

# Method 2: Environment variable (recommended)
# Set AIML_API_KEY in your environment
llm = LLM(model="openai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")

# Method 3: Base URL configuration (if needed)
llm = LLM(
    model="openai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    base_url="https://api.aimlapi.com/v1",
    api_key="your-aiml-api-key"
)
```

## Features

AI/ML API models through CrewAI support:

- **Function Calling**: Most models support tool usage and function calling
- **Streaming**: Real-time response streaming for better user experience
- **Context Windows**: Optimized context window management for each model
- **Vision Models**: Some models support image understanding capabilities
- **Structured Output**: JSON and Pydantic model output formatting

## Model Selection Guide

Choose the right model for your use case:

- **For complex reasoning**: Use Llama 3.1 405B or Claude 3.5 Sonnet
- **For balanced performance**: Use Llama 3.1 70B or Claude 3.5 Haiku
- **For speed and efficiency**: Use Llama 3.1 8B or smaller models
- **For vision tasks**: Use Llama 3.2 Vision models
- **For coding**: Consider DeepSeek or specialized coding models

## Troubleshooting

### Common Issues

1. **Authentication Error**: Ensure your AIML_API_KEY is set correctly
2. **Model Not Found**: Verify the model name uses the correct `openai/` prefix
3. **Rate Limits**: AI/ML API has rate limits; implement appropriate retry logic
4. **Context Length**: Monitor context window usage for optimal performance

### Getting Help

- Check the [AI/ML API Documentation](https://docs.aimlapi.com)
- Review model-specific capabilities and limitations
- Monitor usage and costs through the AI/ML API dashboard

## Migration from Other Providers

If you're migrating from other providers:

```python
# From OpenAI
# OLD: llm = LLM(model="gpt-4")
# NEW: llm = LLM(model="openai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")

# From Anthropic
# OLD: llm = LLM(model="claude-3-sonnet")
# NEW: llm = LLM(model="openai/anthropic/claude-3-5-sonnet-20241022")
```

The integration maintains full compatibility with CrewAI's existing features while providing access to AI/ML API's extensive model catalog.
