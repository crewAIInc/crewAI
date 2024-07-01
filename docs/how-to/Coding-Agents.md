---
title: Coding Agents
description: Learn how to enable your crewAI Agents to write code and execute it.
---

## Introduction
TLDR: strongly recommended to use bigger models like gpt-4 and such

EXAMPLE:
```python
Agent(
    role="Senior Python Developer",
    goal="Craft well design and thought out code",
    backstory="You are a senior python…”,
    allow_code_execution=True,
)
```