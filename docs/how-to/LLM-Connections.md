---
title: Connect CrewAI to LLMs
description: Guide on integrating CrewAI with various Large Language Models (LLMs).
---

## Connect CrewAI to LLMs
!!! note "Default LLM"
    By default, crewAI uses OpenAI's GPT-4 model for language processing. However, you can configure your agents to use a different model or API. This guide will show you how to connect your agents to different LLMs.

CrewAI offers flexibility in connecting to various LLMs, including local models via [Ollama](https://ollama.ai) and different APIs like Azure. It's compatible with all [LangChain LLM](https://python.langchain.com/docs/integrations/llms/) components, enabling diverse integrations for tailored AI solutions.

## Ollama Integration
Ollama is preferred for local LLM integration, offering customization and privacy benefits. It requires installation and configuration, including model adjustments via a Modelfile to optimize performance.

### Setting Up Ollama
- **Installation**: Follow Ollama's guide for setup.
- **Configuration**: [Adjust your local model with a Modelfile](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md), considering adding `Observation` as a stop word and playing with parameters like `top_p` and `temperature`.

### Integrating Ollama with CrewAI
Instantiate Ollama and pass it to your agents within CrewAI, enhancing them with the local model's capabilities.

```python
from langchain_community.llms import Ollama

# Assuming you have Ollama installed and downloaded the openhermes model
ollama_openhermes = Ollama(model="openhermes")

local_expert = Agent(
  role='Local Expert',
  goal='Provide insights about the city',
  backstory="A knowledgeable local guide.",
  tools=[SearchTools.search_internet, BrowserTools.scrape_and_summarize_website],
  llm=ollama_openhermes,
  verbose=True
)
```

## OpenAI Compatible API Endpoints
You can use environment variables for easy switch between APIs and models, supporting diverse platforms like FastChat, LM Studio, and Mistral AI.

### Configuration Examples

### FastChat
```sh
# Required
OPENAI_API_BASE="http://localhost:8001/v1"
OPENAI_API_KEY=NA
MODEL_NAME='oh-2.5m7b-q51' # Depending on the model you have available
```

### LM Studio
```sh
# Required
OPENAI_API_BASE="http://localhost:8000/v1"
OPENAI_API_KEY=NA
MODEL_NAME=NA
```

### Mistral API
```sh
OPENAI_API_KEY=your-mistral-api-key
OPENAI_API_BASE=https://api.mistral.ai/v1
MODEL_NAME="mistral-small" # Check documentation for available models
```

### text-gen-web-ui
```sh
# Required
API_BASE_URL=http://localhost:5000
OPENAI_API_KEY=NA
MODEL_NAME=NA
```

### Azure Open AI
Azure's OpenAI API needs a distinct setup, utilizing the `langchain_openai` component for Azure-specific configurations.

Configuration settings:
```sh
AZURE_OPENAI_VERSION="2022-12-01"
AZURE_OPENAI_DEPLOYMENT=""
AZURE_OPENAI_ENDPOINT=""
AZURE_OPENAI_KEY=""
```

```python
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv()

default_llm = AzureChatOpenAI(
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_KEY")
)

example_agent = Agent(
  role='Example Agent',
  goal='Demonstrate custom LLM configuration',
  backstory='A diligent explorer of GitHub docs.',
  llm=default_llm
)
```

## Conclusion
Integrating CrewAI with different LLMs expands the framework's versatility, allowing for customized, efficient AI solutions across various domains and platforms.
