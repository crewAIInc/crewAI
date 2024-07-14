# LlamaIndexTool Documentation

## Description
This tool is designed to be a general wrapper around LlamaIndex tools and query engines, enabling you to leverage LlamaIndex resources
in terms of RAG/agentic pipelines as tools to plug into CrewAI agents.

## Installation
To incorporate this tool into your project, follow the installation instructions below:
```shell
pip install 'crewai[tools]'
```

## Example
The following example demonstrates how to initialize the tool and execute a search with a given query:

```python
from crewai_tools import LlamaIndexTool

# Initialize the tool from a LlamaIndex Tool

## Example 1: Initialize from FunctionTool
from llama_index.core.tools import FunctionTool

your_python_function = lambda ...: ...
og_tool = FunctionTool.from_defaults(your_python_function, name="<name>", description='<description>')
tool = LlamaIndexTool.from_tool(og_tool)

## Example 2: Initialize from LlamaHub Tools
from llama_index.tools.wolfram_alpha import WolframAlphaToolSpec
wolfram_spec = WolframAlphaToolSpec(app_id="<app_id>")
wolfram_tools = wolfram_spec.to_tool_list()
tools = [LlamaIndexTool.from_tool(t) for t in wolfram_tools]


# Initialize Tool from a LlamaIndex Query Engine

## NOTE: LlamaIndex has a lot of query engines, define whatever query engine you want
query_engine = index.as_query_engine() 
query_tool = LlamaIndexTool.from_query_engine(
    query_engine,
    name="Uber 2019 10K Query Tool",
    description="Use this tool to lookup the 2019 Uber 10K Annual Report"
)

```

## Steps to Get Started
To effectively use the `LlamaIndexTool`, follow these steps:

1. **Install CrewAI**: Confirm that the `crewai[tools]` package is installed in your Python environment.
2. **Install and use LlamaIndex**: Follow LlamaIndex documentation (https://docs.llamaindex.ai/) to setup a RAG/agent pipeline.


