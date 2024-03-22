# WebsiteSearchTool

## Description
This tool is specifically crafted for conducting semantic searches within the content of a particular website. Leveraging a Retrieval-Augmented Generation (RAG) model, it navigates through the information provided on a given URL. Users have the flexibility to either initiate a search across any website known or discovered during its usage or to concentrate the search on a predefined, specific website.

## Installation
Install the crewai_tools package by executing the following command in your terminal:

```shell
pip install 'crewai[tools]'
```

## Example
To utilize the WebsiteSearchTool for different use cases, follow these examples:

```python
from crewai_tools import WebsiteSearchTool

# To enable the tool to search any website the agent comes across or learns about during its operation
tool = WebsiteSearchTool()

# OR

# To restrict the tool to only search within the content of a specific website.
tool = WebsiteSearchTool(website='https://example.com')
```

## Arguments
- `website` : An optional argument that specifies the valid website URL to perform the search on. This becomes necessary if the tool is initialized without a specific website. In the `WebsiteSearchToolSchema`, this argument is mandatory. However, in the `FixedWebsiteSearchToolSchema`, it becomes optional if a website is provided during the tool's initialization, as it will then only search within the predefined website's content.

## Custom model and embeddings

By default, the tool uses OpenAI for both embeddings and summarization. To customize the model, you can use a config dictionary as follows:

```python
tool = WebsiteSearchTool(
    config=dict(
        llm=dict(
            provider="ollama", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="llama2",
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="google",
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)
```
