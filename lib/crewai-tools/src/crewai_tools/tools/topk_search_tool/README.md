# TopKSearchTool

Keyword (BM25), semantic and hybrid search over a [TopK](https://topk.io) collection. On each call the agent picks a `search_mode`: `keyword` for exact strings such as error codes or IDs, `semantic` for paraphrased questions, and `hybrid` (the default) for both in one ranked query. TopK embeds text on the server, so no embedding model or OpenAI key is needed. You only send text. TopK's `semantic_index()` embeds every document when you write it and every query when you search, and builds the BM25 index used by keyword and hybrid search, so there's no embedding model, vector column or embedding pipeline to manage.

## Installation

```bash
uv add topk-sdk
```

Set `TOPK_API_KEY` (from https://console.topk.io) and `TOPK_REGION` (for example `aws-us-east-1-elastica`).

## Usage

```python
from crewai import Agent
from crewai_tools import TopKSearchTool

search = TopKSearchTool(collection_name="runbook", return_fields=["text", "source"])
agent = Agent(
    role="Support Engineer",
    goal="Answer from the runbook",
    backstory="...",
    tools=[search],
)

search.run(query="E-4021", search_mode="keyword")
search.run(
    query="payment failures", filters={"source": "runbook", "year": {"$gte": 2024}}
)
```

The collection's text field needs a `semantic_index()`:

```python
from topk_sdk.schema import semantic_index, text

client.collections().create(
    "runbook", schema={"text": text().required().index(semantic_index())}
)
```

Pass `partition="user-42"` to search only one user's or tenant's partition. See the [docs page](https://docs.crewai.com/en/tools/database-data/topksearchtool) for every argument.
