# ApifyActorsTool

Integrate [Apify Actors](https://apify.com/actors) into your CrewAI workflows.

## Description

The `ApifyActorsTool` connects [Apify Actors](https://apify.com/actors), cloud-based programs for web scraping and automation, to your CrewAI workflows.
Use any of the 4,000+ Actors on [Apify Store](https://apify.com/store) for use cases such as extracting data from social media, search engines, online maps, e-commerce sites, travel portals, or general websites.

For details, see the [Apify CrewAI integration](https://docs.apify.com/platform/integrations/crewai) in Apify documentation.

## Installation

To use `ApifyActorsTool`, install the necessary packages and set up your Apify API token. Follow the [Apify API documentation](https://docs.apify.com/platform/integrations/api) for steps to obtain the token.

### Steps

1. **Install dependencies**
   Install `crewai[tools]` and `langchain-apify`:
   ```bash
   pip install 'crewai[tools]' langchain-apify
   ```

2. **Set your API token**
   Export the token as an environment variable:
   ```bash
   export APIFY_API_TOKEN='your-api-token-here'
   ```

## Usage example

Use the `ApifyActorsTool` manually to run the [RAG Web Browser Actor](https://apify.com/apify/rag-web-browser) to perform a web search:

```python
from crewai_tools import ApifyActorsTool

# Initialize the tool with an Apify Actor
tool = ApifyActorsTool(actor_name="apify/rag-web-browser")

# Run the tool with input parameters
results = tool.run(run_input={"query": "What is CrewAI?", "maxResults": 5})

# Process the results
for result in results:
    print(f"URL: {result['metadata']['url']}")
    print(f"Content: {result.get('markdown', 'N/A')[:100]}...")
```

### Expected output

Here is the output from running the code above:

```text
URL: https://www.example.com/crewai-intro
Content: CrewAI is a framework for building AI-powered workflows...
URL: https://docs.crewai.com/
Content: Official documentation for CrewAI...
```

The `ApifyActorsTool` automatically fetches the Actor definition and input schema from Apify using the provided `actor_name` and then constructs the tool description and argument schema. This means you need to specify only a valid `actor_name`, and the tool handles the rest when used with agents—no need to specify the `run_input`. Here's how it works:

```python
from crewai import Agent
from crewai_tools import ApifyActorsTool

rag_browser = ApifyActorsTool(actor_name="apify/rag-web-browser")

agent = Agent(
    role="Research Analyst",
    goal="Find and summarize information about specific topics",
    backstory="You are an experienced researcher with attention to detail",
    tools=[rag_browser],
)
```

You can run other Actors from [Apify Store](https://apify.com/store) simply by changing the `actor_name` and, when using it manually, adjusting the `run_input` based on the Actor input schema.

For an example of usage with agents, see the [CrewAI Actor template](https://apify.com/templates/python-crewai).

## Configuration

The `ApifyActorsTool` requires these inputs to work:

- **`actor_name`**
  The ID of the Apify Actor to run, e.g., `"apify/rag-web-browser"`. Browse all Actors on [Apify Store](https://apify.com/store).
- **`run_input`**
  A dictionary of input parameters for the Actor when running the tool manually.
  - For example, for the `apify/rag-web-browser` Actor: `{"query": "search term", "maxResults": 5}`
  - See the Actor's [input schema](https://apify.com/apify/rag-web-browser/input-schema) for the list of input parameters.

## Resources

- **[Apify](https://apify.com/)**: Explore the Apify platform.
- **[How to build an AI agent on Apify](https://blog.apify.com/how-to-build-an-ai-agent/)** - A complete step-by-step guide to creating, publishing, and monetizing AI agents on the Apify platform.
- **[RAG Web Browser Actor](https://apify.com/apify/rag-web-browser)**: A popular Actor for web search for LLMs.
- **[CrewAI Integration Guide](https://docs.apify.com/platform/integrations/crewai)**: Follow the official guide for integrating Apify and CrewAI.
