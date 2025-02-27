# ApifyActorsTool

## Description
The `ApifyActorsTool` is a powerful utility that enables seamless integration of [Apify Actors](https://apify.com/) into your CrewAI workflows. Apify Actors are cloud-based web scraping and automation programs that allow you to extract data, crawl websites, and automate tasks without managing infrastructure. This tool provides an efficient way to run Actors like the [RAG Web Browser](https://apify.com/apify/rag-web-browser) directly within your agents, making it ideal for tasks requiring real-time web data extraction or automation. For more Actors, visit the [Apify Store](https://apify.com/store).

For more details on using Apify with CrewAI, visit the [Apify CrewAI integration documentation](https://docs.apify.com/platform/integrations/crewai).

## Installation
To use the `ApifyActorsTool`, you'll need to install the `crewai[tools]` package along with the `langchain-apify` package. Additionally, you must have an Apify API token, which you can obtain by following the instructions in the [Apify API documentation](https://docs.apify.com/platform/integrations/api). Set your API token as an environment variable (`APIFY_API_TOKEN`) to authenticate requests.

Install the required packages using pip:

```shell
pip install 'crewai[tools]' langchain-apify
```

Set your Apify API token in your environment:

```shell
export APIFY_API_TOKEN='Your Apify API token'
```

## Example
The `ApifyActorsTool` is straightforward to integrate into your CrewAI projects. Below is an example of how to initialize and use the tool to run the [RAG Web Browser Actor](https://apify.com/apify/rag-web-browser) to search the web:

```python
from crewai_tools import ApifyActorsTool

# Initialize the tool with the desired Apify Actor
tool = ApifyActorsTool(actor_name="apify/rag-web-browser")

# Run the tool with a specific input, e.g., a search query
results = tool.run(run_input={"query": "What is CrewAI?", "maxResults": 5})
print(results)
```

## Arguments
The `ApifyActorsTool` requires a few key arguments to function correctly:

- `actor_name`: A mandatory argument specifying the ID of the Apify Actor to run (e.g., `"apify/rag-web-browser"`). You can explore available Actors in the [Apify Actors documentation](https://docs.apify.com/platform/actors).
- `run_input`: A dictionary containing the input parameters for the Actor, such as `query` or `maxResults`. The specific inputs depend on the Actor being used. Refer to the Actor's detail page for input schema; for example, [RAG Web Browser Actor](https://apify.com/apify/rag-web-browser/input-schema)

The tool dynamically adapts to the chosen Actor, offering flexibility and ease of use for a wide range of automation and scraping tasks.

## Resources
- [Apify Platform](https://apify.com/) - Learn more about Apify and its ecosystem.
- [RAG Web Browser Actor](https://apify.com/apify/rag-web-browser) - A popular Actor for web searching and data retrieval.
- [Apify Actors Documentation](https://docs.apify.com/platform/actors) - Detailed guide to Apify Actors and their capabilities.
- [CrewAI Integration Guide](https://docs.apify.com/platform/integrations/crewai) - Official documentation for integrating Apify with CrewAI.

The `ApifyActorsTool` empowers your CrewAI agents with robust web scraping and automation capabilities, streamlining complex workflows with minimal setup.
