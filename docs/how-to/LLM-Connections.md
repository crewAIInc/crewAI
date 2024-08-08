---
title: Connect CrewAI to LLMs
description: Comprehensive guide on integrating CrewAI with various Large Language Models (LLMs), including detailed class attributes, methods, and configuration options.
---

## Connect CrewAI to LLMs

!!! note "Default LLM"
    By default, CrewAI uses OpenAI's GPT-4o model (specifically, the model specified by the OPENAI_MODEL_NAME environment variable, defaulting to "gpt-4o") for language processing. You can configure your agents to use a different model or API as described in this guide.
    By default, CrewAI uses OpenAI's GPT-4 model (specifically, the model specified by the OPENAI_MODEL_NAME environment variable, defaulting to "gpt-4") for language processing. You can configure your agents to use a different model or API as described in this guide.

CrewAI provides extensive versatility in integrating with various Language Models (LLMs), including local options through Ollama such as  Llama and Mixtral to cloud-based solutions like Azure. Its compatibility extends to all [LangChain LLM components](https://python.langchain.com/v0.2/docs/integrations/llms/), offering a wide range of integration possibilities for customized AI applications.

The platform supports connections to an array of Generative AI models, including:

 - OpenAI's suite of advanced language models
 - Anthropic's cutting-edge AI offerings
 - Ollama's diverse range of locally-hosted generative model & embeddings
 - LM Studio's diverse range of locally hosted generative models & embeddings
 - Groq's Super Fast LLM offerings
 - Azures' generative AI offerings
 - HuggingFace's generative AI offerings

This broad spectrum of LLM options enables users to select the most suitable model for their specific needs, whether prioritizing local deployment, specialized capabilities, or cloud-based scalability.

## Changing the default LLM
The default LLM is provided through the `langchain openai` package, which is installed by default when you install CrewAI. You can change this default LLM to a different model or API by setting the `OPENAI_MODEL_NAME` environment variable. This straightforward process allows you to harness the power of different OpenAI models, enhancing the flexibility and capabilities of your CrewAI implementation.
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
## Ollama Local Integration
Ollama is preferred for local LLM integration, offering customization and privacy benefits. To integrate Ollama with CrewAI, you will need the `langchain-ollama` package. You can then set the following environment variables to connect to your Ollama instance running locally on port 11434.

```sh
os.environ[OPENAI_API_BASE]='http://localhost:11434'
os.environ[OPENAI_MODEL_NAME]='llama2'  # Adjust based on available model
os.environ[OPENAI_API_KEY]='' # No API Key required for Ollama
```

## Ollama Integration Step by Step (ex. for using Llama 3.1 8B locally)
1. [Download and install Ollama](https://ollama.com/download).   
2. After setting up the Ollama, Pull the Llama3.1 8B model by typing following lines into your terminal ```ollama run llama3.1```.   
3. Llama3.1 should now be served locally on `http://localhost:11434`
```
from crewai import Agent, Task, Crew
from langchain_ollama import ChatOllama
import os
os.environ["OPENAI_API_KEY"] = "NA"

llm = Ollama(
    model = "llama3.1",
    base_url = "http://localhost:11434")

general_agent = Agent(role = "Math Professor",
                      goal = """Provide the solution to the students that are asking mathematical questions and give them the answer.""",
                      backstory = """You are an excellent math professor that likes to solve math questions in a way that everyone can understand your solution""",
                      allow_delegation = False,
                      verbose = True,
                      llm = llm)

task = Task(description="""what is 3 + 5""",
             agent = general_agent,
             expected_output="A numerical answer.")

crew = Crew(
            agents=[general_agent],
            tasks=[task],
            verbose=True
        )

result = crew.kickoff()

print(result)
```

## HuggingFace Integration
There are a couple of different ways you can use HuggingFace to host your LLM.

### Your own HuggingFace endpoint
```python
from langchain_huggingface import HuggingFaceEndpoint,

llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

agent = Agent(
    role="HuggingFace Agent",
    goal="Generate text using HuggingFace",
    backstory="A diligent explorer of GitHub docs.",
    llm=llm
)
```

## OpenAI Compatible API Endpoints
Switch between APIs and models seamlessly using environment variables, supporting platforms like FastChat, LM Studio, Groq, and Mistral AI.

### Configuration Examples
#### FastChat
```sh
os.environ[OPENAI_API_BASE]="http://localhost:8001/v1"
os.environ[OPENAI_MODEL_NAME]='oh-2.5m7b-q51'
os.environ[OPENAI_API_KEY]=NA
```

#### LM Studio
Launch [LM Studio](https://lmstudio.ai) and go to the Server tab. Then select a model from the dropdown menu and wait for it to load. Once it's loaded, click the green Start Server button and use the URL, port, and API key that's shown (you can modify them). Below is an example of the default settings as of LM Studio 0.2.19:
```sh
os.environ[OPENAI_API_BASE]="http://localhost:1234/v1"
os.environ[OPENAI_API_KEY]="lm-studio"
```

#### Groq API
```sh
os.environ[OPENAI_API_KEY]=your-groq-api-key
os.environ[OPENAI_MODEL_NAME]='llama3-8b-8192'
os.environ[OPENAI_API_BASE]=https://api.groq.com/openai/v1
```

#### Mistral API
```sh
os.environ[OPENAI_API_KEY]=your-mistral-api-key
os.environ[OPENAI_API_BASE]=https://api.mistral.ai/v1
os.environ[OPENAI_MODEL_NAME]="mistral-small"
```

### Solar
```sh
from langchain_community.chat_models.solar import SolarChat
```
```sh
os.environ[SOLAR_API_BASE]="https://api.upstage.ai/v1/solar"
os.environ[SOLAR_API_KEY]="your-solar-api-key"
```

# Free developer API key available here: https://console.upstage.ai/services/solar
# Langchain Example: https://github.com/langchain-ai/langchain/pull/18556


### Cohere
```python
from langchain_cohere import ChatCohere
# Initialize language model
os.environ["COHERE_API_KEY"] = "your-cohere-api-key"
llm = ChatCohere()

# Free developer API key available here: https://cohere.com/
# Langchain Documentation: https://python.langchain.com/docs/integrations/chat/cohere
```

### Azure Open AI Configuration
For Azure OpenAI API integration, set the following environment variables:
```sh

os.environ[AZURE_OPENAI_DEPLOYMENT] = "You deployment"
os.environ["OPENAI_API_VERSION"] = "2023-12-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = "Your Endpoint"
os.environ["AZURE_OPENAI_API_KEY"] = "<Your API Key>"
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
