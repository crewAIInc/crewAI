# Taskmarket Requester Tool

`TaskMarketRequesterTool` lets a CrewAI agent prepare and monitor Taskmarket
bounties while keeping spending authorization in trusted host code. It uses the
first-party `taskmarket` CLI, never accepts wallet keys, and never exposes
accept/reject operations to the model.

## Setup

```bash
npm install -g @lucid-agents/taskmarket@latest
taskmarket init
taskmarket legal status
```

Fund the CLI wallet with Base mainnet USDC before enabling creation. Configure a
hard per-task cap when constructing the tool:

```python
from decimal import Decimal

from crewai_tools import TaskMarketRequesterTool

tool = TaskMarketRequesterTool(max_reward_usdc=Decimal("5.00"))
```

## Human-approved creation

The model first calls `prepare_create`. The result shows the exact description,
deliverables, reward, projected UTC deadline, Base network, permanent visibility,
maximum spend, and a SHA-256 fingerprint. Preparation does not transact.

Trusted UI or controller code must display that exact preview and then approve it:

```python
preview = await_agent_preview()
show_to_user(preview)
if user_confirmed():
    tool.approve(preview["preview_id"], preview["fingerprint_sha256"])
```

Only then can the agent call `create` with that `preview_id`. Approval is bound to
the exact prepared arguments, expires after five minutes by default, and is
claimed atomically before preflight so concurrent calls cannot duplicate a spend.
After any create attempt begins, a new preview and approval are required. A
timed-out or ambiguous write is never retried: the tool tells the operator to
reconcile using Taskmarket inbox and wallet history.

Immediately before creation, the tool verifies:

- Base chain ID `8453` and canonical Base USDC;
- acting Taskmarket wallet and available USDC balance;
- the current legal-bundle enforcement state;
- reward does not exceed the configured maximum spend.

Successful creation returns the task ID, Taskmarket link, acting wallet, maximum
spend, and live task status when the follow-up read succeeds.

## Review without autonomous selection

Use `status` to retrieve current task state and `submissions` to present candidates.
Both results state that human review is required. This tool intentionally provides
no accept, reject, rate, or winner-selection operation.

## Reproducible demo

The following transcript uses an injected runner, so it exercises the complete
approval and settlement-handling path without spending USDC or exposing secrets:

```python
import json
from decimal import Decimal

from crewai_tools import TaskMarketRequesterTool


def recorded_runner(args, timeout):
    command = tuple(args[1:])
    responses = {
        ("address",): {"ok": True, "data": {"address": "0x1111111111111111111111111111111111111111"}},
        ("deposit",): {"ok": True, "data": {"chainId": 8453, "usdcContract": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"}},
        ("wallet", "balance"): {"ok": True, "data": {"balanceUsdc": "5.0"}},
        ("legal", "status"): {"ok": True, "data": {"enforcementEnabled": False}},
    }
    if command[:2] == ("task", "create"):
        payload = {"ok": True, "data": {"taskId": "0x" + "a" * 64}}
    elif command[:2] == ("task", "get"):
        payload = {"ok": True, "data": {"status": "open"}}
    else:
        payload = responses[command]
    return 0, json.dumps(payload), ""


tool = TaskMarketRequesterTool(
    runner=recorded_runner,
    max_reward_usdc=Decimal("2"),
)
preview = json.loads(
    tool.run(
        operation="prepare_create",
        description="Audit the release candidate.",
        deliverables=["Report", "Reproduction steps"],
        reward_usdc=Decimal("1.25"),
        duration_hours=24,
    )
)

# This call belongs in trusted UI/controller code after displaying the preview.
tool.approve(preview["preview_id"], preview["fingerprint_sha256"])
created = json.loads(tool.run(operation="create", preview_id=preview["preview_id"]))
assert created["created"] is True
assert created["task_id"] == "0x" + "a" * 64
```

Run the focused behavior suite from the repository root:

```bash
uv run pytest lib/crewai-tools/tests/tools/taskmarket_requester_tool_test.py -q
```

The tests cover exact previews, spend caps, Base/USDC/legal/balance preflights,
single-use authorization, unknown-settlement handling, and read-only submission
review.
