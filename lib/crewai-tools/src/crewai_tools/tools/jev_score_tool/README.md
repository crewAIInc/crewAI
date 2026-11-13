# Jev Score Tool

## Description

Use `JevScoreTool` to rate task state against one TypeSafe Jev Score rubric and
return the expected score, the probability of each level, and confidence.

## Installation

Install the optional dependency before using the tool:

```bash
uv add "crewai-tools[typesafe-sdk]"
```

Set `TYPESAFE_API_KEY`, then opt in from a task or direct tool call.

## Example

```python
from crewai_tools import JevScoreTool

tool = JevScoreTool()

result = tool.run(
    state="The production database is down and customers cannot log in.",
    question="How urgent is this message?",
    criteria=[
        "Can wait",
        "Needs attention this week",
        "Needs attention today",
    ],
)

if result["success"]:
    print(result["score"])          # e.g. 1.94, may fall between levels
    print(result["probabilities"])  # e.g. {0: 0.01, 1: 0.04, 2: 0.95}
    print(result["confidence"])
else:
    print(result["error"])
```

## Arguments

- `state`: Text, JSON object, or array for Jev to evaluate.
- `question`: Score question or instruction to rate the state against.
- `criteria`: Ordered descriptions of the score levels, lowest first. Each
  level's position is its score, starting at 0.

`score` is the probability-weighted average of the levels, not a pass/fail
flag. `success` only reports whether TypeSafe returned a usable answer; decide
what score is acceptable in your own code.
