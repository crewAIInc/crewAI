# NewsflashNewsTool Documentation

## Description

Search real-time news as **deduplicated, corroborated events** using
[Newsflash](https://newsflash.sh). Instead of returning one raw article per hit,
Newsflash clusters coverage from many outlets into a single event with a
corroboration count and a confidence score (`min(1, sources / 3)`). The
`min_confidence` argument lets an agent require that a story has been
independently reported by multiple outlets before acting on it — a simple,
effective gate against single-source and fabricated news.

## Installation

```shell
pip install 'crewai[tools]'
```

No extra dependencies are needed (the tool uses the standard `requests` package).

## Example

```python
from crewai_tools import NewsflashNewsTool

# Works without an API key (50 requests/day)
tool = NewsflashNewsTool()

# Semantic search, only events corroborated by 2+ sources
tool.run(query="AI chip export restrictions", min_confidence=0.6)

# Keyword search restricted to a category
tool.run(query="bitcoin", semantic=False, category="crypto", limit=5)
```

## Arguments

- `query` (str, required): Search query.
- `semantic` (bool, default `True`): Meaning-based search; set `False` for exact
  keyword matching.
- `category` (str, optional): One of `crypto`, `tradfi`, `business`, `tech`,
  `politics`, `world`, `science`, `health`, `energy`, `sports`.
- `limit` (int, default `10`): Maximum number of events to return (max 50).
- `min_confidence` (float, default `0.0`): Only return events with confidence at
  or above this value. `1.0` means at least 3 independent outlets reported the
  same happening.

## Environment Variables

- `NEWSFLASH_API_KEY` (optional): API key for higher rate limits and deeper
  history (free keys via email at https://newsflash.sh). Keyless access works
  out of the box with 50 requests/day.
