# Correctover GuardrailProvider

First third-party reference implementation of the `GuardrailProvider` protocol
from [crewAIInc/crewAI#4877](https://github.com/crewAIInc/crewAI/issues/4877).

## What is this?

A drop-in authorization provider for CrewAI's pre-tool-call hook system.
Instead of returning a simple `bool`, providers return structured `GuardrailDecision`
objects with deny reasons, metadata, and fail-closed semantics.

## Quick Start

```python
from crewai.hooks import enable_guardrail
from crewai.guardrails.providers import CorrectoverGuardrailProvider

provider = CorrectoverGuardrailProvider(
    blocked_tools={"dangerous_tool"},
    allowed_agents={"agent-123"},
    require_agent_identity=True,
)

enable_guardrail(provider)
```

## 6-Dimensional Verification

| Dimension | Checks |
|-----------|--------|
| Structure | tool_name and tool_input have valid types |
| Schema | Tool passes allow/block list policy |
| Identity | Agent identity matches policy |
| Integrity | tool_input verified as immutable snapshot (no mutation since construction) |
| Latency | Request within time bounds |
| Cost | Estimated cost within policy limits |

## Test Coverage

31 tests, all passing.

Targets upstream patch kit: safal207/ibex-agent-verification#68
