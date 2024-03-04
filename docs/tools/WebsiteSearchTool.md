# WebsiteSearchTool

!!! note "Depend on OpenAI"
    All RAG tools at the moment can only use openAI to generate embeddings, we are working on adding support for other providers.

!!! note "Experimental"
    We are still working on improving tools, so there might be unexpected behavior or changes in the future.

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