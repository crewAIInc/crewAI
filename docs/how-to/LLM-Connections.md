---
title: Connect CrewAI to LLMs
description: Guide on integrating CrewAI with various Large Language Models (LLMs).
---

## Connect CrewAI to LLMs
!!! note "Default LLM"
    By default, CrewAI uses OpenAI's GPT-4 model for language processing. However, you can configure your agents to use a different model or API. This guide will show you how to connect your agents to different LLMs through environment variables and direct instantiation.

CrewAI offers flexibility in connecting to various LLMs, including local models via [Ollama](https://ollama.ai) and different APIs like Azure. It's compatible with all [LangChain LLM](https://python.langchain.com/docs/integrations/llms/) components, enabling diverse integrations for tailored AI solutions.

## CrewAI Agent Overview
The `Agent` class in CrewAI is central to implementing AI solutions. Here's a brief overview:

- **Attributes**:
    - `role`: Defines the agent's role within the solution.
    - `goal`: Specifies the agent's objective.
    - `backstory`: Provides a background story to the agent.
    - `llm`: Indicates the Large Language Model the agent uses.

### Example Changing OpenAI's GPT model
```python
# Required
os.environ["OPENAI_MODEL_NAME"]="gpt-4-0125-preview"

# Agent will automatically use the model defined in the environment variable
example_agent = Agent(
  role='Local Expert',
  goal='Provide insights about the city',
  backstory="A knowledgeable local guide.",
  verbose=True
)
```

## Ollama Integration
Ollama is preferred for local LLM integration, offering customization and privacy benefits. To integrate Ollama with CrewAI, set the appropriate environment variables as shown below. Note: Detailed Ollama setup is beyond this document's scope, but general guidance is provided.

### Setting Up Ollama
- **Environment Variables Configuration**: To integrate Ollama, set the following environment variables:
```sh
OPENAI_API_BASE='http://localhost:11434/v1'
OPENAI_MODEL_NAME='openhermes'  # Adjust based on available model
OPENAI_API_KEY=''
```

## OpenAI Compatible API Endpoints
Switch between APIs and models seamlessly using environment variables, supporting platforms like FastChat, LM Studio, and Mistral AI.

### Configuration Examples
#### FastChat
```sh
OPENAI_API_BASE="http://localhost:8001/v1"
OPENAI_MODEL_NAME='oh-2.5m7b-q51'
OPENAI_API_KEY=NA
```

#### LM Studio
```sh
OPENAI_API_BASE="http://localhost:8000/v1"
OPENAI_MODEL_NAME=NA
OPENAI_API_KEY=NA
```

#### Mistral API
```sh
OPENAI_API_KEY=your-mistral-api-key
OPENAI_API_BASE=https://api.mistral.ai/v1
OPENAI_MODEL_NAME="mistral-small"
```

### Azure Open AI Configuration
For Azure OpenAI API integration, set the following environment variables:
```sh
AZURE_OPENAI_VERSION="2022-12-01"
AZURE_OPENAI_DEPLOYMENT=""
AZURE_OPENAI_ENDPOINT=""
AZURE_OPENAI_KEY=""
```

### Example Agent with Azure LLM
```python
from dotenv import load_dotenv
from crewai import Agent
from langchain_openai import AzureChatOpenAI

load_dotenv()

azure_llm = AzureChatOpenAI(
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_KEY")
)

azure_agent = Agent(
  role='Example Agent',
  goal='Demonstrate custom LLM configuration',
  backstory='A diligent explorer of GitHub docs.',
  llm=azure_llm
)
```

## Conclusion
Integrating CrewAI with different LLMs expands the framework's versatility, allowing for customized, efficient AI solutions across various domains and platforms.
