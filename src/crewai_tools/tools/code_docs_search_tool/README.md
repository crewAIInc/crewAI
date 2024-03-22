# CodeDocsSearchTool

## Description
The CodeDocsSearchTool is a powerful RAG (Retrieval-Augmented Generation) tool designed for semantic searches within code documentation. It enables users to efficiently find specific information or topics within code documentation. By providing a `docs_url` during initialization, the tool narrows down the search to that particular documentation site. Alternatively, without a specific `docs_url`, it searches across a wide array of code documentation known or discovered throughout its execution, making it versatile for various documentation search needs.

## Installation
To start using the CodeDocsSearchTool, first, install the crewai_tools package via pip:
```shell
pip install 'crewai[tools]'
```

## Example
Utilize the CodeDocsSearchTool as follows to conduct searches within code documentation:
```python
from crewai_tools import CodeDocsSearchTool

# To search any code documentation content if the URL is known or discovered during its execution:
tool = CodeDocsSearchTool()

# OR

# To specifically focus your search on a given documentation site by providing its URL:
tool = CodeDocsSearchTool(docs_url='https://docs.example.com/reference')
```
Note: Substitute 'https://docs.example.com/reference' with your target documentation URL and 'How to use search tool' with the search query relevant to your needs.

## Arguments
- `docs_url`: Optional. Specifies the URL of the code documentation to be searched. Providing this during the tool's initialization focuses the search on the specified documentation content.

## Custom model and embeddings

By default, the tool uses OpenAI for both embeddings and summarization. To customize the model, you can use a config dictionary as follows:

```python
tool = YoutubeVideoSearchTool(
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
