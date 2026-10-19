# PixeltableSearchTool

## Description

Search a [Pixeltable](https://docs.pixeltable.com) table by embedding similarity. The text column must already have an embedding index. Pixeltable embeds the query with that index and returns the closest rows with a similarity score.

## Installation

Pixeltable requires Python 3.11 or newer.

```shell
uv add 'crewai-tools[pixeltable]'
```

## Example

```python
from crewai import Agent
from crewai_tools import PixeltableSearchTool

tool = PixeltableSearchTool(
    table="dir.docs",  # or a pixeltable.Table
    column="text",  # text column with an embedding index
    limit=5,
    select=["title"],  # optional extra columns
    idx="minilm_idx",  # optional; required when the column has multiple indexes
)

agent = Agent(
    role="Researcher",
    goal="Find documents similar to a question",
    tools=[tool],
)
```

The indexed column is created in Pixeltable before the tool runs:

```python
import pixeltable as pxt
from pixeltable.functions.huggingface import sentence_transformer

docs = pxt.get_table("dir.docs")
docs.add_embedding_index(
    "text",
    idx_name="minilm_idx",
    embedding=sentence_transformer.using(
        model_id="sentence-transformers/all-MiniLM-L12-v2"
    ),
)
```

## Arguments

- `table`: Pixeltable table path (for example `dir.docs`) or a `pixeltable.Table`. Required.
- `column`: Name of the text column that has an embedding index. Required.
- `limit`: Number of rows to return. Defaults to `5`.
- `select`: Extra column names to include with the text and score. Optional.
- `idx`: Embedding index name. Optional unless the column has more than one index.

The tool input is a `query` string. The output is a JSON list of rows, including a `score` for each row.
