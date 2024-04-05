---
title: Connect CrewAI to LLMs
description: Comprehensive guide on integrating CrewAI with various Large Language Models (LLMs), including detailed class attributes and methods.
---

## Connect CrewAI to LLMs
!!! note "Default LLM"
    By default, CrewAI uses OpenAI's GPT-4 model for language processing. You can configure your agents to use a different model or API. This guide shows how to connect your agents to various LLMs through environment variables and direct instantiation.

CrewAI offers flexibility in connecting to various LLMs, including local models via [Ollama](https://ollama.ai) and different APIs like Azure. It's compatible with all [LangChain LLM](https://python.langchain.com/docs/integrations/llms/) components, enabling diverse integrations for tailored AI solutions.

## CrewAI Agent Overview
The `Agent` class is the cornerstone for implementing AI solutions in CrewAI. Here's an updated overview reflecting the latest codebase changes:

- **Attributes**:
    - `role`: Defines the agent's role within the solution.
    - `goal`: Specifies the agent's objective.
    - `backstory`: Provides a background story to the agent.
    - `llm`: The language model that will run the agent. By default, it uses the GPT-4 model defined in the environment variable "OPENAI_MODEL_NAME".
    - `function_calling_llm`: The language model that will handle the tool calling for this agent, overriding the crew function_calling_llm. Optional.
    - `max_iter`: Maximum number of iterations for an agent to execute a task, default is 15.
    - `memory`: Enables the agent to retain information during and a across executions. Default is `False`.
    - `max_rpm`: Maximum number of requests per minute the agent's execution should respect. Optional.
    - `verbose`: Enables detailed logging of the agent's execution. Default is `False`.
    - `allow_delegation`: Allows the agent to delegate tasks to other agents, default is `True`.
    - `tools`: Specifies the tools available to the agent for task execution. Optional.
    - `step_callback`: Provides a callback function to be executed after each step. Optional.
    - `cache`: Determines whether the agent should use a cache for tool usage. Default is `True`.

```python
# Required
os.environ["OPENAI_MODEL_NAME"]="gpt-4-0125-preview"

# Agent will automatically use the model defined in the environment variable
example_agent = Agent(
  role='Local Expert',
  goal='Provide insights about the city',
  backstory="A knowledgeable local guide.",
  verbose=True,
  memory=True
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

## HuggingFace Integration
There are a couple of different ways you can use HuggingFace to host your LLM.

### Your own HuggingFace endpoint
```python
from langchain_community.llms import HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    endpoint_url="<YOUR_ENDPOINT_URL_HERE>",
    huggingfacehub_api_token="<HF_TOKEN_HERE>",
    task="text-generation",
    max_new_tokens=512
)

agent = Agent(
    role="HuggingFace Agent",
    goal="Generate text using HuggingFace",
    backstory="A diligent explorer of GitHub docs.",
    llm=llm
)
```

### From HuggingFaceHub endpoint
```python
from langchain_community.llms import HuggingFaceHub

llm = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token="<HF_TOKEN_HERE>",
    task="text-generation",
)
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

### Solar
```sh
from langchain_community.chat_models.solar import SolarChat
# Initialize language model
os.environ["SOLAR_API_KEY"] = "your-solar-api-key"
llm = SolarChat(max_tokens=1024)

Free developer API key available here: https://console.upstage.ai/services/solar
Langchain Example: https://github.com/langchain-ai/langchain/pull/18556
```
### text-gen-web-ui
```sh
OPENAI_API_BASE=http://localhost:5000/v1
OPENAI_MODEL_NAME=NA
OPENAI_API_KEY=NA
```

### Cohere
```sh
from langchain_community.chat_models import ChatCohere
# Initialize language model
os.environ["COHERE_API_KEY"] = "your-cohere-api-key"
llm = ChatCohere()

Free developer API key available here: https://cohere.com/
Langchain Documentation: https://python.langchain.com/docs/integrations/chat/cohere
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
