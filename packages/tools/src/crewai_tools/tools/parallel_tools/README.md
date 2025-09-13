# ParallelSearchTool

Unified Parallel web search tool using the Parallel Search API (v1beta). Returns ranked results with compressed excerpts optimized for LLMs.

- **Quickstart**: see the official docs: [Search API Quickstart](https://docs.parallel.ai/search-api/search-quickstart)
- **Processors**: guidance on `base` vs `pro`: [Processors](https://docs.parallel.ai/search-api/processors)

## Why this tool

- **Single-call pipeline**: Replaces search → scrape → extract with a single, low‑latency API call.
- **LLM‑ready**: Returns compressed excerpts that feed directly into LLM prompts (fewer tokens, less pre/post‑processing).
- **Flexible**: Control result count and excerpt length; optionally restrict sources via `source_policy`.

## Environment

- `PARALLEL_API_KEY` (required)

Optional (for the agent example):
- `OPENAI_API_KEY` or other LLM provider keys supported by CrewAI

## Parameters

- `objective` (str, optional): Natural‑language research goal (≤ 5000 chars)
- `search_queries` (list[str], optional): Up to 5 keyword queries (each ≤ 200 chars)
- `processor` (str, default `base`): `base` (fast/low cost) or `pro` (freshness/quality)
- `max_results` (int, default 10): ≤ 40 (subject to processor limits)
- `max_chars_per_result` (int, default 6000): ≥ 100; values > 30000 not guaranteed
- `source_policy` (dict, optional): Source policy for domain inclusion/exclusion

Notes:
- API is in beta; default rate limit is 600 RPM. Contact support for production capacity.

## Direct usage (when published)

```python
from crewai_tools import ParallelSearchTool

tool = ParallelSearchTool()
resp_json = tool.run(
  objective="When was the United Nations established? Prefer UN's websites.",
  search_queries=["Founding year UN", "Year of founding United Nations"],
  processor="base",
  max_results=5,
  max_chars_per_result=1500,
)
print(resp_json)  # => {"search_id": ..., "results": [{"url", "title", "excerpts": [...]}, ...]}
```

### Parameters you can pass

Call `run(...)` with any of the following (at least one of `objective` or `search_queries` is required):

```python
tool.run(
  objective: str | None = None,                 # ≤ 5000 chars
  search_queries: list[str] | None = None,      # up to 5 items, each ≤ 200 chars
  processor: str = "base",                      # "base" (fast) or "pro" (freshness/quality)
  max_results: int = 10,                        # ≤ 40 (processor limits apply)
  max_chars_per_result: int = 6000,             # ≥ 100 (values > 30000 not guaranteed)
  source_policy: dict | None = None,            # optional SourcePolicy config
)
```

Example with `source_policy`:

```python
source_policy = {
  "allow": {"domains": ["un.org"]},
  # "deny": {"domains": ["example.com"]},  # optional
}

resp_json = tool.run(
  objective="When was the United Nations established?",
  processor="base",
  max_results=5,
  max_chars_per_result=1500,
  source_policy=source_policy,
)
```

## Example with agents

Here’s a minimal example that calls `ParallelSearchTool` to fetch sources and has an LLM produce a short, cited answer.

```python
import os
from crewai import Agent, Task, Crew, LLM, Process
from crewai_tools import ParallelSearchTool

# LLM
llm = LLM(
  model="gemini/gemini-2.0-flash",
  temperature=0.5,
  api_key=os.getenv("GEMINI_API_KEY")
)

# Parallel Search
search = ParallelSearchTool()

# User query
query = "find all the recent concerns about AI evals? please cite the sources"

# Researcher agent 
researcher = Agent(
  role="Web Researcher",
  backstory="You are an expert web researcher",
  goal="Find cited, high-quality sources and provide a brief answer.",
  tools=[search],
  llm=llm,
  verbose=True,
)

# Research task
task = Task(
  description=f"Research the {query} and produce a short, cited answer.",
  expected_output="A concise, sourced answer to the question. The answer should be in this format: [query]: [answer] - [source]",
  agent=researcher,
  output_file="answer.mdx",
)

# Crew
crew = Crew(
    agents=[researcher], 
    tasks=[task], 
    verbose=True,
    process=Process.sequential,
)

# Run the crew
result = crew.kickoff(inputs={'query': query})
print(result)
```

Output from the agent above:

```md
Recent concerns about AI evaluations include: the rise of AI-related incidents alongside a lack of standardized Responsible AI (RAI) evaluations among major industrial model developers - [https://hai.stanford.edu/ai-index/2025-ai-index-report]; flawed benchmark datasets that fail to account for critical factors, leading to unrealistic estimates of AI model abilities - [https://www.nature.com/articles/d41586-025-02462-5]; the need for multi-metric, context-aware evaluations in medical imaging AI to ensure reliability and clinical relevance - [https://www.sciencedirect.com/science/article/pii/S3050577125000283]; challenges related to data sets (insufficient, imbalanced, or poor quality), communication gaps, and misaligned expectations in AI model training - [https://www.oracle.com/artificial-intelligence/ai-model-training-challenges/]; the argument that LLM agents should be evaluated primarily on their riskiness, not just performance, due to unreliability, hallucinations, and brittleness - [https://www.technologyreview.com/2025/06/24/1119187/fix-ai-evaluation-crisis/]; the fact that the AI industry's embraced benchmarks may be close to meaningless, with top makers of AI models picking and choosing different responsible AI benchmarks, complicating efforts to systematically compare risks and limitations - [https://themarkup.org/artificial-intelligence/2024/07/17/everyone-is-judging-ai-by-these-tests-but-experts-say-theyre-close-to-meaningless]; and the difficulty of building robust and reliable model evaluations, as many existing evaluation suites are limited in their ability to serve as accurate indicators of model capabilities or safety - [https://www.anthropic.com/research/evaluating-ai-systems].
```

Tips:
- Ensure your LLM provider keys are set (e.g., `GEMINI_API_KEY`) and CrewAI model config is in place.
- For longer analyses, raise `max_chars_per_result` or use `processor="pro"` (higher quality, higher latency).

## Behavior

- Single‑request web research; no scraping/post‑processing required.
- Returns `search_id` and ranked `results` with compressed `excerpts`.
- Clear error handling on HTTP/timeouts.

## References

- Search API Quickstart: https://docs.parallel.ai/search-api/search-quickstart
- Processors: https://docs.parallel.ai/search-api/processors
