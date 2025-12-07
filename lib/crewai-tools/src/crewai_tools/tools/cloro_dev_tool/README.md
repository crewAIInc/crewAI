# CloroDevTool

Use the `CloroDevTool` to search the web or query AI models using the cloro API.

## Installation

```shell
pip install 'crewai[tools]'
```

## Example

```python
from crewai_tools import CloroDevTool

# make sure CLORO_API_KEY variable is set
tool = CloroDevTool()

result = tool.run(search_query="latest news about AI agents")

print(result)
```

## Arguments

- `api_key` (str, optional): cloro API key.
- `engine` (str, optional): The engine to use for the query. Options are `google`, `chatgpt`, `gemini`, `copilot`, `perplexity`, `aimode`. Defaults to `google`.
- `country` (str, optional): The ISO 3166-1 alpha-2 country code for localized results (e.g., "US", "BR"). For a full list of supported country codes, refer to the [cloro API /v1/countries endpoint](https://docs.cloro.dev/api-reference/endpoint/countries). Defaults to "US".
- `device` (str, optional): The device type for Google search results (`desktop` or `mobile`). Defaults to "desktop".
- `pages` (int, optional): The number of pages to retrieve for Google search results. Defaults to 1.
- `save_file` (bool, optional): Whether to save the search results to a file. Defaults to `False`.

Get the credentials by creating a [cloro account](https://dashboard.cloro.dev).

## Response Format

The tool returns a structured dictionary containing different fields depending on the selected engine.

### Google Engine
- `organic`: List of organic search results with title, link, snippet, etc.
- `peopleAlsoAsk`: List of related questions.
- `relatedSearches`: List of related search queries.
- `knowledgeGraph`: Knowledge graph data (if available).
- `ai_overview`: Google AI Overview data (if available).

### LLM Engines (ChatGPT, Gemini, etc.)
- `text`: The main response text from the model.
- `sources`: List of sources cited by the model (if available).

## Advanced example

Check out the cloro [documentation](https://docs.cloro.dev/api-reference/introduction) to get the full list of parameters.

```python
from crewai_tools import CloroDevTool

# make sure CLORO_API_KEY variable is set
tool = CloroDevTool(
    engine="chatgpt",
    country="BR",
    save_file=True
)

result = tool.run(search_query="Say 'Hello, Brazil!'")

print(result)
```