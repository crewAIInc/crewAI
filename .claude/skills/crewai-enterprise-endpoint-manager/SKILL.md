---
name: crewai-enterprise-endpoint-manager
description: |
  Guide for interacting with deployed CrewAI Enterprise (AOP) crews and flows via API endpoints.
  Use this skill when: (1) Kicking off deployed crews/flows programmatically, (2) Monitoring execution status and progress,
  (3) Retrieving results from completed executions, (4) Understanding the REST API workflow for deployed agentic workflows,
  (5) Implementing authentication with Bearer tokens, (6) Building integrations that consume deployed CrewAI endpoints,
  (7) Handling human-in-the-loop webhooks with deployed crews, (8) Managing concurrent executions with semaphore patterns.
---

# CrewAI Enterprise Endpoint Manager

Deployed crews and flows in CrewAI Enterprise (AOP) are accessible via REST API endpoints for programmatic execution, monitoring, and result retrieval.

## Contents

- [API Workflow](#api-workflow-overview)
- [Authentication](#authentication)
- [Endpoint Reference](#endpoint-reference)
- [Python Integration](#python-integration)
- [Human-in-the-Loop](#human-in-the-loop-hitl-webhooks)
- [Flow Integration](#integration-with-crewai-flows)
- [Best Practices](#best-practices)
- [Status States](#status-states-reference)

**Detailed References:**
- [references/python-client.md](references/python-client.md) - Full Python client class, async patterns, batch execution with semaphore
- [references/error-handling.md](references/error-handling.md) - Retry strategies, rate limiting, circuit breaker patterns

## API Workflow Overview

```
1. GET /inputs      → Discover required input parameters
2. POST /kickoff    → Start execution (returns kickoff_id)
3. GET /{id}/status → Monitor progress and retrieve results
```

## Authentication

All requests require a Bearer token from the **Status tab** of your crew's detail page in the AOP dashboard.

```bash
curl -H "Authorization: Bearer YOUR_CREW_TOKEN" \
  https://your-crew-url.crewai.com/endpoint
```

## Endpoint Reference

### 1. Discover Inputs - `GET /inputs`

```bash
curl -H "Authorization: Bearer YOUR_CREW_TOKEN" \
  https://your-crew-url.crewai.com/inputs
```

**Response:**
```json
{
  "inputs": [
    {"name": "topic", "type": "string", "required": true},
    {"name": "max_results", "type": "integer", "required": false}
  ]
}
```

### 2. Start Execution - `POST /kickoff`

```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_CREW_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"inputs": {"topic": "AI Research", "max_results": 10}}' \
  https://your-crew-url.crewai.com/kickoff
```

**Response:**
```json
{"kickoff_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef"}
```

### 3. Monitor Status - `GET /{kickoff_id}/status`

```bash
curl -H "Authorization: Bearer YOUR_CREW_TOKEN" \
  https://your-crew-url.crewai.com/a1b2c3d4-e5f6-7890-1234-567890abcdef/status
```

**Running:**
```json
{
  "status": "running",
  "current_task": "research_task",
  "progress": {"completed_tasks": 1, "total_tasks": 3}
}
```

**Completed:**
```json
{
  "status": "completed",
  "result": {
    "output": "Final output from the crew...",
    "tasks": [
      {"task_id": "research_task", "output": "Research findings...", "agent": "Travel Researcher", "execution_time": 45.2}
    ]
  },
  "execution_time": 108.5
}
```

**Error:**
```json
{"status": "error", "error_message": "Failed to complete task due to invalid input."}
```

## Python Integration

Basic synchronous pattern for simple integrations:

```python
import requests
import time

BASE_URL = "https://your-crew-url.crewai.com"
TOKEN = "YOUR_CREW_TOKEN"
headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

def kickoff_crew(inputs: dict) -> str:
    """Start execution and return kickoff_id."""
    resp = requests.post(f"{BASE_URL}/kickoff", headers=headers, json={"inputs": inputs})
    resp.raise_for_status()
    return resp.json()["kickoff_id"]

def wait_for_completion(kickoff_id: str, poll_interval: float = 2.0) -> dict:
    """Poll until execution completes or fails."""
    while True:
        resp = requests.get(f"{BASE_URL}/{kickoff_id}/status", headers=headers)
        resp.raise_for_status()
        status = resp.json()
        if status["status"] in ("completed", "error"):
            return status
        time.sleep(poll_interval)

# Usage
kickoff_id = kickoff_crew({"topic": "AI Research"})
result = wait_for_completion(kickoff_id)
print(result["result"]["output"])
```

**For advanced patterns, see [references/python-client.md](references/python-client.md):**
- Full `CrewAIClient` class with sync/async methods
- Batch execution with semaphore-controlled concurrency
- Rate limiting and progress callbacks
- Structured output parsing with Pydantic

**For error handling, see [references/error-handling.md](references/error-handling.md):**
- Exponential backoff and retry strategies
- Rate limit handling with adaptive concurrency
- Circuit breaker pattern for resilience

## Human-in-the-Loop (HITL) Webhooks

For crews requiring human input during execution:

```bash
curl -X POST {BASE_URL}/kickoff \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {"topic": "AI Research"},
    "humanInputWebhook": {
      "url": "https://your-webhook.com/hitl",
      "authentication": {"strategy": "bearer", "token": "your-webhook-secret"}
    }
  }'
```

**Webhook payload received:**
```json
{"kickoff_id": "abc123", "task_id": "review_task", "prompt": "Please review...", "context": {...}}
```

**Respond:**
```json
{"response": "Approved with minor edits...", "continue": true}
```

## Integration with CrewAI Flows

Deployed Flows expose the same API. Inputs map to your Flow's state model:

```python
# Local Flow definition
class ResearchState(BaseModel):
    topic: str = ""
    depth: int = 1

class ResearchFlow(Flow[ResearchState]):
    @start()
    def begin_research(self):
        ...
```

```bash
# API call maps to state fields
curl -X POST -H "Authorization: Bearer YOUR_FLOW_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"inputs": {"topic": "Quantum Computing", "depth": 3}}' \
  https://your-flow-url.crewai.com/kickoff
```

## Best Practices

1. **Discover inputs first** - Always call `GET /inputs` to understand required parameters
2. **Handle all status states** - Check for "running", "completed", and "error"
3. **Use semaphores for batches** - Limit concurrent executions (see [python-client.md](references/python-client.md))
4. **Implement exponential backoff** - For retries: `2^attempt` seconds (see [error-handling.md](references/error-handling.md))
5. **Store kickoff_ids** - Persist IDs for debugging and resumption
6. **Set appropriate timeouts** - Long-running crews may need 10+ minute timeouts

## Status States Reference

| Status | Meaning | Next Action |
|--------|---------|-------------|
| `pending` | Queued for execution | Continue polling |
| `running` | Execution in progress | Continue polling |
| `completed` | Successfully finished | Extract results |
| `error` | Execution failed | Check error_message, retry if transient |
