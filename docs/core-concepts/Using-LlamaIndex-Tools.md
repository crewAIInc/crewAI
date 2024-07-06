---
title: Using LlamaIndex Tools
description: Learn how to integrate LlamaIndex tools with CrewAI agents to enhance search-based queries and more.
---

## Using LlamaIndex Tools

!!! info "LlamaIndex Integration"
    CrewAI seamlessly integrates with LlamaIndex’s comprehensive toolkit for RAG (Retrieval-Augmented Generation) and agentic pipelines, enabling advanced search-based queries and more. Here are the available built-in tools offered by LlamaIndex.

```python
from crewai import Agent
from crewai_tools import LlamaIndexTool

# Example 1: Initialize from FunctionTool
from llama_index.core.tools import FunctionTool

your_python_function = lambda ...: ...
og_tool = FunctionTool.from_defaults(your_python_function, name="<name>", description='<description>')
tool = LlamaIndexTool.from_tool(og_tool)

# Example 2: Initialize from LlamaHub Tools
from llama_index.tools.wolfram_alpha import WolframAlphaToolSpec
wolfram_spec = WolframAlphaToolSpec(app_id="<app_id>")
wolfram_tools = wolfram_spec.to_tool_list()
tools = [LlamaIndexTool.from_tool(t) for t in wolfram_tools]

# Example 3: Initialize Tool from a LlamaIndex Query Engine
query_engine = index.as_query_engine()
query_tool = LlamaIndexTool.from_query_engine(
    query_engine,
    name="Uber 2019 10K Query Tool",
    description="Use this tool to lookup the 2019 Uber 10K Annual Report"
)

# Create and assign the tools to an agent
agent = Agent(
  role='Research Analyst',
  goal='Provide up-to-date market analysis',
  backstory='An expert analyst with a keen eye for market trends.',
  tools=[tool, *tools, query_tool]
)

# rest of the code ...
```

## Steps to Get Started

To effectively use the LlamaIndexTool, follow these steps:

1. **Package Installation**: Confirm that the `crewai[tools]` package is installed in your Python environment.

    ```shell
    pip install 'crewai[tools]'
    ```

2. **Install and Use LlamaIndex**: Follow LlamaIndex documentation [LlamaIndex Documentation](https://docs.llamaindex.ai/) to set up a RAG/agent pipeline.