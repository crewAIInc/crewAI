# ApifyActorsTool

Integrate [Apify Actors](https://apify.com/) into your CrewAI workflows.

## Description

ApifyActorsTool connects [Apify Actors](https://apify.com/), cloud-based programs for web scraping and automation, to your CrewAI workflows. You can extract data, crawl websites, and automate tasks, all without requiring infrastructure management.

**Key features**:
- **Run Actors** directly, like the [RAG Web Browser](https://apify.com/apify/rag-web-browser), with CrewAI agents.
- **Access real-time data** for tasks that need fresh web content or automation.

See the [Apify CrewAI documentation](https://docs.apify.com/platform/integrations/crewai) for a detailed integration guide.

## Installation

To use ApifyActorsTool, install the necessary packages and set up your Apify API token. Follow the [Apify API documentation](https://docs.apify.com/platform/integrations/api) for steps to obtain the token.

### Steps

1. **Install dependencies**
   Use pip to install `crewai[tools]` and `langchain-apify`:
   ```bash
   pip install 'crewai[tools]' langchain-apify
   ```
   Or, with `uv`:
   ```bash
   uv pip install 'crewai[tools]' langchain-apify
   ```

2. **Set your API token**
   Export the token as an environment variable:
   ```bash
   export APIFY_API_TOKEN='your-api-token-here'
   ```

## Usage example

Use ApifyActorsTool to run the [RAG Web Browser Actor](https://apify.com/apify/rag-web-browser) and perform a web search:

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

Experiment with other Actors from the [Apify Store](https://apify.com/store) by updating `actor_name` and `run_input` based on each Actor's input schema.

For an example of usage with agents, see the [CrewAI Apify Actor template](https://apify.com/templates/python-crewai).

## Configuration

ApifyActorsTool requires these inputs to work:

- **`actor_name`**
  The ID of the Apify Actor to run, e.g., `"apify/rag-web-browser"`. Browse options in the [Apify Store](https://apify.com/store).
- **`run_input`**
  A dictionary of input parameters for the Actor. Examples:
  - For `apify/rag-web-browser`: `{"query": "search term", "maxResults": 5}`
  - See each Actor's [input schema](https://apify.com/apify/rag-web-browser/input-schema) for details.

## Resources

- **[Apify Platform](https://apify.com/)**: Dive into the Apify ecosystem.
- **[RAG Web Browser Actor](https://apify.com/apify/rag-web-browser)**: Test this popular Actor for web data retrieval.
- **[CrewAI Integration Guide](https://docs.apify.com/platform/integrations/crewai)**: Follow the official guide for Apify and CrewAI.
