# VerifierAgent Pattern: Validating Agent Outputs in CrewAI

## Overview

In multi-agent systems, Large Language Model (LLM) agents may generate outputs that are
factually incorrect, inconsistent, or hallucinated. While CrewAI enables powerful
agent orchestration, it does not enforce output validation by default.

This example demonstrates a **VerifierAgent pattern**, where a secondary agent is used
to evaluate and validate the output of a primary agent before it is accepted or used
downstream.

---

## Problem

- LLM agents can hallucinate facts
- Incorrect outputs may propagate through agent pipelines
- High-stakes workflows require an additional validation step

Without verification, agent-based systems risk producing unreliable results.

---

## Solution: VerifierAgent Pattern

The VerifierAgent pattern introduces a dedicated agent responsible for:
- Fact-checking
- Logical validation
- Hallucination detection
- Confidence scoring

This separates **generation** from **verification**, improving overall reliability.

---

## Architecture

```text
User Query
   ↓
Generator Agent
   ↓
Verifier Agent
   ↓
Decision (Accept / Reject / Revise)
```
## How It Works

A Generator Agent produces an initial response.

A Verifier Agent reviews the response for factual accuracy and consistency.

The Verifier Agent returns a structured evaluation.

The result can be used to accept, reject, or revise the output.

## Example Implementation

A runnable example is provided in:

verifier_agent_crew.py
This example includes:

One Generator Agent

One Verifier Agent

Sequential task execution


## Sample Output


{
  "verdict": "incorrect",
  "confidence": 0.45,
  "issues": [
    "Incorrect factual claim regarding climate change causes"
  ],
  "suggested_fix": "Revise the response using verified scientific sources"
}


## When to Use This Pattern
Research assistants

Enterprise decision systems

RAG pipelines

High-stakes AI workflows

Multi-agent collaboration systems

## Limitations
Verification quality depends on the underlying LLM

This pattern reduces risk but does not guarantee correctness

For critical applications, combine with external tools or trusted data sources