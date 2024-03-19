---
title: Connect CrewAI to LLMs
description: Comprehensive guide on integrating CrewAI with various Large Language Models (LLMs), including detailed class attributes and methods.
---

## Connect CrewAI to LLMs
!!! note "Default LLM"
    By default, CrewAI uses OpenAI's GPT-4 model for language processing. However, you can configure your agents to use a different model or API. This guide will show you how to connect your agents to different LLMs through environment variables and direct instantiation.

CrewAI offers flexibility in connecting to various LLMs, including local models via [Ollama](https://ollama.ai) and different APIs like Azure. It's compatible with all [LangChain LLM](https://python.langchain.com/docs/integrations/llms/) components, enabling diverse integrations for tailored AI solutions.

## CrewAI Agent Overview
The `Agent` class is the cornerstone for implementing AI solutions in CrewAI. Here's an updated overview reflecting the latest codebase changes:

- **Attributes**:
    - `role`: Defines the agent's role within the solution.
    - `goal`: Specifies the agent's objective.
    - `backstory`: Provides a background story to the agent.
    - `llm`: Indicates the Large Language Model the agent uses.
    - `function_calling_llm` *Optinal*: Will turn the ReAct crewAI agent into a function calling agent.
    - `max_iter`: Maximum number of iterations for an agent to execute a task, default is 15.
    - `memory`: Enables the agent to retain information during the execution.
    - `max_rpm`: Sets the maximum number of requests per minute.
    - `verbose`: Enables detailed logging of the agent's execution.
    - `allow_delegation`: Allows the agent to delegate tasks to other agents, default is `True`.
    - `tools`: Specifies the tools available to the agent for task execution.
    - `step_callback`: Provides a callback function to be executed after each step.

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
Ollama is preferred for local LLM integration, offering customization and privacy benefits. To integrate Ollama with CrewAI, set the appropriate environment variables as shown below. 

### Setting Up Ollama
- **Environment Variables Configuration**: To integrate Ollama, set the following environment variables:
```sh
OPENAI_API_BASE='http://localhost:11434/v1'
OPENAI_MODEL_NAME='openhermes'  # Adjust based on available model
OPENAI_API_KEY=''
```

## Ollama Integration for using Llama 2 locally
1. [Download Ollama](https://ollama.com/download).   
2. After setting up the Ollama, Pull the Llama2 by typing following lines into the terminal ```ollama pull Llama2```.   
3. Create a ModelFile similar the one below in your project directory.
```
FROM llama2

# Set parameters

PARAMETER temperature 0.8
PARAMETER stop Result

# Sets a custom system message to specify the behavior of the chat assistant

# Leaving it blank for now.

SYSTEM """"""
```
4. Create a script to get the base model, which in our case is llama2, and create a model on top of that with ModelFile above. PS: this will be ".sh" file.     
```
#!/bin/zsh

# variables
model_name="llama2"
custom_model_name="crewai-llama2"

#get the base model
ollama pull $model_name

#create the model file
ollama create $custom_model_name -f ./Llama2ModelFile
```
5. Go into the directory where the script file and ModelFile is located and run the script.   
6. Enjoy your free Llama2 model that powered up by excellent agents from crewai.   
```
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import os
os.environ["OPENAI_API_KEY"] = "NA"

llm = ChatOpenAI(
    model = "crewai-llama2",
    base_url = "http://localhost:11434/v1")

general_agent = Agent(role = "Math Professor",
                      goal = """Provide the solution to the students that are asking mathematical questions and give them the answer.""",
                      backstory = """You are an excellent math professor that likes to solve math questions in a way that everyone can understand your solution""",
                      allow_delegation = False,
                      verbose = True,
                      llm = llm)
task = Task (description="""what is 3 + 5""",
             agent = general_agent)

crew = Crew(
            agents=[general_agent],
            tasks=[task],
            verbose=2
        )

result = crew.kickoff()

print(result)
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
