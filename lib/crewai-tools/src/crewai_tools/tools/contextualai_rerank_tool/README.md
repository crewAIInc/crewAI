# ContextualAIRerankTool

## Description
This tool is designed to integrate Contextual AI's enterprise-grade instruction-following reranker with CrewAI, enabling you to intelligently reorder documents based on relevance and custom criteria. Use this tool to enhance search result quality and document retrieval for RAG systems using Contextual AI's reranking models that understand context and follow specific instructions for optimal document ordering.

## Installation
To incorporate this tool into your project, follow the installation instructions below:

```shell
pip install 'crewai[tools]' contextual-client
```

**Note**: You'll need a Contextual AI API key. Sign up at [app.contextual.ai](https://app.contextual.ai) to get your free API key.

## Example

```python
from crewai_tools import ContextualAIRerankTool

tool = ContextualAIRerankTool(api_key="your_api_key_here")

result = tool._run(
    query="financial performance and revenue metrics",
    documents=[
        "Q1 report content with revenue data", 
        "Q2 report content with growth metrics", 
        "News article about market trends"
    ],
    instruction="Prioritize documents with specific financial metrics and quantitative data"
)
print(result)
```

The result will contain the document ranking. For example: 
```
Rerank Result:
{
  "results": [
    {
      "index": 1,
      "relevance_score": 0.88227631
    },
    {
      "index": 0,
      "relevance_score": 0.61159354
    },
    {
      "index": 2,
      "relevance_score": 0.28579462
    }
  ]
}
```

## Parameters
- `api_key`: Your Contextual AI API key
- `query`: Search query for reranking
- `documents`: List of document texts to rerank
- `instruction`: Optional reranking instruction for custom criteria
- `metadata`: Optional metadata for each document
- `model`: Reranker model (default: "ctxl-rerank-en-v1-instruct")

## Key Features
- **Instruction-Following Reranking**: Follows custom instructions for domain-specific document ordering
- **Metadata Integration**: Incorporates document metadata for enhanced ranking decisions

## Use Cases
- Improve search result relevance in document collections
- Reorder documents by custom business criteria (recency, authority, relevance)
- Filter and prioritize documents for research and analysis workflows

For more detailed information about Contextual AI's capabilities, visit the [official documentation](https://docs.contextual.ai).