# SerperDevTool Documentation

!!! note "Experimental"
    We are still working on improving tools, so there might be unexpected behavior or changes in the future.

## Description
This tool is designed to perform a semantic search for a specified query from a text's content across the internet. It utilizes the [serper.dev](https://serper.dev) API to fetch and display the most relevant search results based on the query provided by the user.

## Installation
To incorporate this tool into your project, follow the installation instructions below:
```shell
pip install 'crewai[tools]'
```

## Example
The following example demonstrates how to initialize the tool and execute a search with a given query:

```python
from crewai_tools import SerperDevTool

# Initialize the tool for internet searching capabilities
tool = SerperDevTool()
```

## Steps to Get Started
To effectively use the `SerperDevTool`, follow these steps:

1. **Package Installation**: Confirm that the `crewai[tools]` package is installed in your Python environment.
2. **API Key Acquisition**: Acquire a `serper.dev` API key by registering for a free account at `serper.dev`.
3. **Environment Configuration**: Store your obtained API key in an environment variable named `SERPER_API_KEY` to facilitate its use by the tool.

## Parameters

The `SerperDevTool` comes with several parameters that will be passed to the API :

- **search_url**: The URL endpoint for the search API. (Default is `https://google.serper.dev/search`)

- **country**: Optional. Specify the country for the search results.
- **location**: Optional. Specify the location for the search results.
- **locale**: Optional. Specify the locale for the search results.
- **n_results**: Number of search results to return. Default is `10`.

The values for `country`, `location`, `lovale` and `search_url` can be found on the [Serper Playground](https://serper.dev/playground).

## Example with Parameters

Here is an example demonstrating how to use the tool with additional parameters:

```python
from crewai_tools import SerperDevTool

tool = SerperDevTool(
    search_url="https://google.serper.dev/scholar",
    n_results=2,
)

print(tool.run(search_query="ChatGPT"))

# Using Tool: Search the internet

# Search results: Title: Role of chat gpt in public health
# Link: https://link.springer.com/article/10.1007/s10439-023-03172-7
# Snippet: … ChatGPT in public health. In this overview, we will examine the potential uses of ChatGPT in
# ---
# Title: Potential use of chat gpt in global warming
# Link: https://link.springer.com/article/10.1007/s10439-023-03171-8
# Snippet: … as ChatGPT, have the potential to play a critical role in advancing our understanding of climate
# ---

```

```python
from crewai_tools import SerperDevTool

tool = SerperDevTool(
    country="fr",
    locale="fr",
    location="Paris, Paris, Ile-de-France, France",
    n_results=2,
)

print(tool.run(search_query="Jeux Olympiques"))

# Using Tool: Search the internet

# Search results: Title: Jeux Olympiques de Paris 2024 - Actualités, calendriers, résultats
# Link: https://olympics.com/fr/paris-2024
# Snippet: Quels sont les sports présents aux Jeux Olympiques de Paris 2024 ? · Athlétisme · Aviron · Badminton · Basketball · Basketball 3x3 · Boxe · Breaking · Canoë ...
# ---
# Title: Billetterie Officielle de Paris 2024 - Jeux Olympiques et Paralympiques
# Link: https://tickets.paris2024.org/
# Snippet: Achetez vos billets exclusivement sur le site officiel de la billetterie de Paris 2024 pour participer au plus grand événement sportif au monde.
# ---

```

## Conclusion
By integrating the `SerperDevTool` into Python projects, users gain the ability to conduct real-time, relevant searches across the internet directly from their applications. The updated parameters allow for more customized and localized search results. By adhering to the setup and usage guidelines provided, incorporating this tool into projects is streamlined and straightforward.
