---
title: Connect CrewAI to LLMs
description: Comprehensive guide on integrating CrewAI with various Large Language Models (LLMs) using LiteLLM, including supported providers and configuration options.
---

## Connect CrewAI to LLMs

CrewAI now uses LiteLLM to connect to a wide variety of Language Models (LLMs). This integration provides extensive versatility, allowing you to use models from numerous providers with a simple, unified interface.

!!! note "Default LLM"
    By default, CrewAI uses OpenAI's GPT-4 model (specifically, the model specified by the OPENAI_MODEL_NAME environment variable, defaulting to "gpt-4") for language processing. You can easily configure your agents to use a different model or provider as described in this guide.

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

To use a different LLM with your CrewAI agents, you simply need to pass the model name as a string when initializing the agent. Here are some examples:

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

# Using Ollama's local Llama 2 model
ollama_agent = Agent(
    role='Local AI Expert',
    goal='Process information using a local model',
    backstory="An AI assistant running on local hardware.",
    llm='ollama/llama2'
)

# Using Google's Gemini model
gemini_agent = Agent(
    role='Google AI Expert',
    goal='Generate creative content with Gemini',
    backstory="An AI assistant powered by Google's advanced language model.",
    llm='gemini-pro'
)
```

## Configuration

For most providers, you'll need to set up your API keys as environment variables. Here's how you can do it for some common providers:

```python
import os

# OpenAI
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Anthropic
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"

# Google (Vertex AI)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/your/credentials.json"

# Azure OpenAI
os.environ["AZURE_API_KEY"] = "your-azure-api-key"
os.environ["AZURE_API_BASE"] = "your-azure-endpoint"

# AWS (Bedrock)
os.environ["AWS_ACCESS_KEY_ID"] = "your-aws-access-key-id"
os.environ["AWS_SECRET_ACCESS_KEY"] = "your-aws-secret-access-key"
```

For providers that require additional configuration or have specific setup requirements, please refer to the [LiteLLM documentation](https://docs.litellm.ai/docs/) for detailed instructions.

## Using Local Models

For local models like those provided by Ollama, ensure you have the necessary software installed and running. For example, to use Ollama:

1. [Download and install Ollama](https://ollama.com/download)
2. Pull the desired model (e.g., `ollama pull llama2`)
3. Use the model in your CrewAI agent by specifying `llm='ollama/llama2'`

## Conclusion

By leveraging LiteLLM, CrewAI now offers seamless integration with a vast array of LLMs. This flexibility allows you to choose the most suitable model for your specific needs, whether you prioritize performance, cost-efficiency, or local deployment. Remember to consult the [LiteLLM documentation](https://docs.litellm.ai/docs/) for the most up-to-date information on supported models and configuration options.