# How-To: Connect CrewAI to LLMs

There are different types of connections.
Ollama is the recommended way to connect to local LLMs.
Azure uses a slightly different API and therefore has it's own connection object.

crewAI is compatible with any of the LangChain LLM components. See this page for more information: https://python.langchain.com/docs/integrations/llms/

## Ollama

crewAI supports integration with local models thorugh [Ollama](https://ollama.ai/) for enhanced flexibility and customization. This allows you to utilize your own models, which can be particularly useful for specialized tasks or data privacy concerns. We will conver other options for using local models in later sections. However, ollama is the recommended tool to use to host local models when possible.

### Setting Up Ollama

- **Install Ollama**: Ensure that Ollama is properly installed in your environment. Follow the installation guide provided by Ollama for detailed instructions.
- **Configure Ollama**: Set up Ollama to work with your local model. You will probably need to [tweak the model using a Modelfile](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md). I'd recommend adding `Observation` as a stop word and playing with `top_p` and `temperature`.

### Integrating Ollama with CrewAI
- Instantiate Ollama Model: Create an instance of the Ollama model. You can specify the model and the base URL during instantiation. For example:

```python
from langchain.llms import Ollama
ollama_openhermes = Ollama(model="openhermes")
# Pass Ollama Model to Agents: When creating your agents within the CrewAI framework, you can pass the Ollama model as an argument to the Agent constructor. For instance:

local_expert = Agent(
  role='Local Expert at this city',
  goal='Provide the BEST insights about the selected city',
  backstory="""A knowledgeable local guide with extensive information
  about the city, it's attractions and customs""",
  tools=[
    SearchTools.search_internet,
    BrowserTools.scrape_and_summarize_website,
  ],
  llm=ollama_openhermes, # Ollama model passed here
  verbose=True
)
```

## Open AI Compatible API Endpoints

In the context of integrating various language models with CrewAI, the flexibility to switch between different API endpoints is a crucial feature. By utilizing environment variables for configuration details such as `OPENAI_API_BASE_URL`, `OPENAI_API_KEY`, and `MODEL_NAME`, you can easily transition between different APIs or models. For instance, if you want to switch from using the standard OpenAI GPT model to a custom or alternative version, simply update the values of these environment variables. 

The `OPENAI_API_BASE_URL` variable allows you to define the base URL of the API to connect to, while `OPENAI_API_KEY` is used for authentication purposes. Lastly, the `MODEL_NAME` variable specifies the particular language model to be used, such as "gpt-3.5-turbo" or any other available model. 

This method offers an easy way to adapt the system to different models or plataforms, be it for testing, scaling, or accessing different features available on various platforms. By centralizing the configuration in environment variables, the process becomes streamlined, reducing the need for extensive code modifications when switching between APIs or models.


```python
from dotenv import load_dotenv
from langchain.chat_models.openai import ChatOpenAI

load_dotenv()

defalut_llm = ChatOpenAI(openai_api_base=os.environ.get("OPENAI_API_BASE_URL", "https://api.openai.com/v1"),
                        openai_api_key=os.environ.get("OPENAI_API_KEY", "NA"),
                        model_name=os.environ.get("MODEL_NAME", "gpt-3.5-turbo"))

# Create an agent and assign the LLM
example_agent = Agent(
  role='Example Agent',
  goal='Show how to assign a custom configured LLM',
  backstory='You hang out in the docs section of GitHub repos.',
  llm=default_llm
)

```


### Open AI

OpenAI is the default LLM that will be used if you do not specify a value for the `llm` argument when creating an agent. It will also use default values for the `OPENAI_API_BASE_URL` and `MODEL_NAME`. So the only value you need to set when using the OpenAI endpoint is the API key that from your account.

```sh
#REQUIRED
OPENAI_API_KEY="sk-..."

#Optional
OPENAI_API_BASE_URL=https://api.openai.com/v1
MODEL_NAME="gpt-3.5-turbo"
```

### LM Studio

https://lmstudio.ai/

Configuration settings:
```sh
#REQUIRED
OPENAI_API_BASE_URL="http://localhost:8000/v1"

OPENAI_API_KEY=NA
MODEL_NAME=NA
```

### FastChat

https://github.com/lm-sys/FastChat?tab=readme-ov-file#api

Configuration settings:
```sh
OPENAI_API_BASE_URL="http://localhost:8001/v1"
OPENAI_API_KEY=NA
MODEL_NAME='oh-2.5m7b-q51'
```

### text-gen-web-ui

https://github.com/oobabooga/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API


Configuration settings:

```sh
API_BASE_URL=http://localhost:5000
OPENAI_API_KEY=NA
MODEL_NAME=NA
```

## Other Inference API Endpoints

Other platforms offer inference APIs such as Anthropic, Azure, and HuggingFace to name a few. Unfortunately, the APIs on the following platforms are not compatible with the OpenAI API specification. So, the following platforms will require a slightly different configuration than the examples in the previous section.

### Azure Open AI

Azure hosted OpenAI API endpoints have their own LLM component that needs to be imported from `langchain_openai`.

For more information, check out the langchain documenation for [Azure OpenAI](https://python.langchain.com/docs/integrations/llms/azure_openai).

```python
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv()

default_llm = AzureChatOpenAI(
    openai_api_version=os.environ.get("AZURE_OPENAI_VERSION", "2023-07-01-preview"),
    azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt35"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", "https://<your-endpoint>.openai.azure.com/"),
    api_key=os.environ.get("AZURE_OPENAI_KEY")
)

# Create an agent and assign the LLM
example_agent = Agent(
  role='Example Agent',
  goal='Show how to assign a custom configured LLM',
  backstory='You hang out in the docs section of GitHub repos.',
  llm=default_llm
)

```


Configuration settings:
```sh
AZURE_OPENAI_VERSION="2022-12-01"
AZURE_OPENAI_DEPLOYMENT=""
AZURE_OPENAI_ENDPOINT=""
AZURE_OPENAI_KEY=""
```
