---
title: Connect CrewAI to LLMs
description: Comprehensive guide on integrating CrewAI with various Large Language Models (LLMs) using LiteLLM, including supported providers and configuration options.
---

## Connect CrewAI to LLMs

CrewAI uses LiteLLM to connect to a wide variety of Language Models (LLMs). This integration provides extensive versatility, allowing you to use models from numerous providers with a simple, unified interface.

!!! note "Default LLM"
    By default, CrewAI uses the `gpt-4o-mini` model. This is determined by the `OPENAI_MODEL_NAME` environment variable, which defaults to "gpt-4o-mini" if not set. You can easily configure your agents to use a different model or provider as described in this guide.

## Supported Providers

LiteLLM supports a wide range of providers, including but not limited to:

- OpenAI
- Anthropic
- Google (Vertex AI, Gemini)
- Azure OpenAI
- AWS (Bedrock, SageMaker)
- Cohere
- Hugging Face
- Ollama
- Mistral AI
- Replicate
- Together AI
- AI21
- Cloudflare Workers AI
- DeepInfra
- Groq
- And many more!

For a complete and up-to-date list of supported providers, please refer to the [LiteLLM Providers documentation](https://docs.litellm.ai/docs/providers).

## Changing the LLM

To use a different LLM with your CrewAI agents, you have several options:

### 1. Using a String Identifier

Pass the model name as a string when initializing the agent:

```python
from crewai import Agent

# Using OpenAI's GPT-4
openai_agent = Agent(
    role='OpenAI Expert',
    goal='Provide insights using GPT-4',
    backstory="An AI assistant powered by OpenAI's latest model.",
    llm='gpt-4'
)

# Using Anthropic's Claude
claude_agent = Agent(
    role='Anthropic Expert',
    goal='Analyze data using Claude',
    backstory="An AI assistant leveraging Anthropic's language model.",
    llm='claude-2'
)
```

### 2. Using the LLM Class

For more detailed configuration, use the LLM class:

```python
from crewai import Agent, LLM

llm = LLM(
    model="gpt-4",
    temperature=0.7,
    base_url="https://api.openai.com/v1",
    api_key="your-api-key-here"
)

agent = Agent(
    role='Customized LLM Expert',
    goal='Provide tailored responses',
    backstory="An AI assistant with custom LLM settings.",
    llm=llm
)
```

## Configuration Options

When configuring an LLM for your agent, you have access to a wide range of parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | str | The name of the model to use (e.g., "gpt-4", "claude-2") |
| `temperature` | float | Controls randomness in output (0.0 to 1.0) |
| `max_tokens` | int | Maximum number of tokens to generate |
| `top_p` | float | Controls diversity of output (0.0 to 1.0) |
| `frequency_penalty` | float | Penalizes new tokens based on their frequency in the text so far |
| `presence_penalty` | float | Penalizes new tokens based on their presence in the text so far |
| `stop` | str, List[str] | Sequence(s) to stop generation |
| `base_url` | str | The base URL for the API endpoint |
| `api_key` | str | Your API key for authentication |

For a complete list of parameters and their descriptions, refer to the LLM class documentation.

## Connecting to OpenAI-Compatible LLMs

You can connect to OpenAI-compatible LLMs using either environment variables or by setting specific attributes on the LLM class:

### Using Environment Variables

```python
import os

os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["OPENAI_API_BASE"] = "https://api.your-provider.com/v1"
os.environ["OPENAI_MODEL_NAME"] = "your-model-name"
```

### Using LLM Class Attributes

```python
llm = LLM(
    model="custom-model-name",
    api_key="your-api-key",
    base_url="https://api.your-provider.com/v1"
)
agent = Agent(llm=llm, ...)
```

## Using Local Models with Ollama

For local models like those provided by Ollama:

1. [Download and install Ollama](https://ollama.com/download)
2. Pull the desired model (e.g., `ollama pull llama2`)
3. Configure your agent:

```python
agent = Agent(
    role='Local AI Expert',
    goal='Process information using a local model',
    backstory="An AI assistant running on local hardware.",
    llm=LLM(model="ollama/llama2", base_url="http://localhost:11434")
)
```

## Changing the Base API URL

You can change the base API URL for any LLM provider by setting the `base_url` parameter:

```python
llm = LLM(
    model="custom-model-name",
    base_url="https://api.your-provider.com/v1",
    api_key="your-api-key"
)
agent = Agent(llm=llm, ...)
```

This is particularly useful when working with OpenAI-compatible APIs or when you need to specify a different endpoint for your chosen provider.

## Conclusion

By leveraging LiteLLM, CrewAI offers seamless integration with a vast array of LLMs. This flexibility allows you to choose the most suitable model for your specific needs, whether you prioritize performance, cost-efficiency, or local deployment. Remember to consult the [LiteLLM documentation](https://docs.litellm.ai/docs/) for the most up-to-date information on supported models and configuration options.
