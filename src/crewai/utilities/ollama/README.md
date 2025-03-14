# Ollama Integration for CrewAI

This module provides integration between CrewAI and Ollama, allowing you to use local LLMs with CrewAI without requiring an OpenAI API key.

## Overview

The integration works by applying a monkey patch to `litellm.completion`, which is used by CrewAI to communicate with LLMs. The monkey patch intercepts calls to Ollama models and redirects them to the local Ollama API instead of going through LiteLLM's normal channels.

## Usage

To use this integration, you need to:

1. Install and run Ollama locally (see [ollama.ai](https://ollama.ai))
2. Pull the desired model (e.g., `ollama pull llama3`)
3. Apply the monkey patch at the beginning of your CrewAI application:

```python
from crewai import Agent, Crew, Task
from crewai.llm import LLM
from crewai import apply_monkey_patch

# Apply the monkey patch
apply_monkey_patch()

# Create an LLM instance with an Ollama model
llm = LLM(model="ollama/llama3", base_url="http://localhost:11434")

# Use the LLM instance with CrewAI
agent = Agent(
    role="Local AI Expert",
    goal="Process information using a local model",
    backstory="An AI assistant running on local hardware.",
    llm=llm
)

# Continue with your CrewAI application...
```

## Configuration

The Ollama integration supports the following configuration options:

- `model`: The name of the Ollama model to use, prefixed with "ollama/" (e.g., "ollama/llama3")
- `base_url`: The base URL for the Ollama API (default: "http://localhost:11434")
- `temperature`: The temperature parameter for generation (default: 0.7)
- `stream`: Whether to stream the response (default: False)

## Supported Models

Any model available in your local Ollama installation can be used with this integration. Just prefix the model name with "ollama/" when creating the LLM instance.

## Limitations

- Tool calling is not fully supported with local Ollama models
- Some advanced features like response formatting may not work as expected
- Token counting is estimated rather than exact

## Troubleshooting

If you encounter issues with the Ollama integration, check the following:

1. Make sure Ollama is running locally
2. Verify that you've pulled the model you're trying to use
3. Check that the base_url is correct
4. Look for error messages in the logs

For more information, see the [CrewAI documentation](https://docs.crewai.com/how-to/LLM-Connections/).
