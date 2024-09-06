# MDXSearchTool

!!! note "Experimental"
    The MDXSearchTool is in continuous development. Features may be added or removed, and functionality could change unpredictably as we refine the tool.

## Description
The MDX Search Tool is a component of the `crewai_tools` package aimed at facilitating advanced markdown language extraction. It enables users to effectively search and extract relevant information from MD files using query-based searches. This tool is invaluable for data analysis, information management, and research tasks, streamlining the process of finding specific information within large document collections.

## Installation
Before using the MDX Search Tool, ensure the `crewai_tools` package is installed. If it is not, you can install it with the following command:

```shell
pip install 'crewai[tools]'
```

## Usage Example
To use the MDX Search Tool, you must first set up the necessary environment variables. Then, integrate the tool into your crewAI project to begin your market research. Below is a basic example of how to do this:

```python
from crewai_tools import MDXSearchTool

# Initialize the tool to search any MDX content it learns about during execution
tool = MDXSearchTool()

# OR

# Initialize the tool with a specific MDX file path for an exclusive search within that document
tool = MDXSearchTool(mdx='path/to/your/document.mdx')
```

## Parameters
- mdx: **Optional**. Specifies the MDX file path for the search. It can be provided during initialization.

## Customization of Model and Embeddings

The tool defaults to using OpenAI for embeddings and summarization. For customization, utilize a configuration dictionary as shown below:

```python
tool = MDXSearchTool(
    config=dict(
        llm=dict(
            provider="ollama", # Options include google, openai, anthropic, llama2, etc.
            config=dict(
                model="llama2",
                # Optional parameters can be included here.
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="google", # or openai, ollama, ...
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
                # Optional title for the embeddings can be added here.
                # title="Embeddings",
            ),
        ),
    )
)
```
