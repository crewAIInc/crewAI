# KCP Benchmark Results — CrewAI

## Summary

**76% reduction in tool calls** when using the Knowledge Context Protocol (KCP) manifest compared to unguided repository exploration.

- Baseline total: **123 tool calls**
- KCP total: **30 tool calls**
- Saved: **93 tool calls** across 8 queries

## Results Table

| Query | Baseline | KCP | Saved |
| :---- | -------: | --: | ----: |
| What is the difference between Flows and Crews in CrewAI? | 14 | 2 | 12 |
| How do I create my first agent and assign it a task? | 7 | 3 | 4 |
| How do I create a custom tool for my agent? | 8 | 3 | 5 |
| How do I add memory to my crew? | 7 | 3 | 4 |
| Which LLM providers does CrewAI support? | 17 | 5 | 12 |
| How do I build a flow that triggers a crew? | 15 | 2 | 13 |
| How do I implement a hierarchical crew with a manager agent? | 22 | 9 | 13 |
| How do I add knowledge (RAG) to my crew? | 33 | 3 | 30 |
| **TOTAL** | **123** | **30** | **93** |

## Methodology

Each query was run twice against the CrewAI repository (`/src/totto/crewAI`):

1. **Baseline**: The agent was told the repository path and instructed to explore it freely using `read_file`, `glob_files`, and `grep_content` tools to find the answer.
2. **KCP**: The agent was instructed to first read `knowledge.yaml`, match the query against unit triggers, and read only the files pointed to by matching units — preferring TL;DR summary files when available.

Both runs used `claude-haiku-4-5-20251001` with `max_tokens=2048` and up to 20 turns. Tool call counts measure retrieval efficiency only (not answer quality).

## Findings

The KCP manifest delivered a **76% reduction in tool calls**, with the largest gains on broad or unfamiliar queries. The "knowledge (RAG)" query showed the most dramatic improvement (33 → 3 calls, 91% reduction): without KCP the agent recursively explored the docs directory; with KCP it read `knowledge.yaml`, matched the `rag crew` trigger directly to `tools-memory-tldr.mdx`, and answered immediately. The hierarchical crew query had the smallest relative gain (22 → 9), because the answer required reading the full `crews.mdx` and `tasks.mdx` even with guidance — demonstrating that KCP eliminates exploration overhead but cannot shrink inherently large source files.
