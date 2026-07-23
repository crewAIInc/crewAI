# Comply54 Compliance Tool

Pre-execution African regulatory enforcement for CrewAI agents. Evaluates agent actions and parameters against Nigerian, Kenyan, South African, and other African jurisdiction policy packs before execution — returning a structured allow/deny/escalate/audit decision with exact regulatory citations.

**No API key required.** Evaluation runs fully offline via in-process Rego evaluation (regopy). No OPA binary, no network call.

## Covered jurisdictions

| Country | Regulations |
|---|---|
| Nigeria | CBN NIP Framework, NDPA 2023, NIIRA 2025, NFIU AML, BVN/NIN |
| Kenya | KDPA 2019 |
| South Africa | POPIA |
| Ghana | Ghana DPA 2012 |
| Rwanda | Rwanda DPA 2021 |
| Egypt | Egypt PDPL No. 151/2020 |
| Tanzania, Uganda, Ethiopia, Mauritius | Jurisdiction-specific packs |

## Installation

```bash
pip install crewai-tools[comply54]
```

## Usage

### Pattern 1 — Explicit self-check

The agent calls the tool explicitly before a regulated action:

```python
from crewai import Agent
from crewai_tools import Comply54ComplianceTool
from comply54.sectors import NigeriaFintechCompliance

tool = Comply54ComplianceTool(compliance=NigeriaFintechCompliance())

agent = Agent(
    role="Fintech Payment Agent",
    tools=[transfer_funds_tool, tool],
    ...
)
```

The tool returns a JSON decision:

```json
{
  "overall": "deny",
  "blocked": true,
  "audit_id": "c54-20260701-abc123",
  "violations": [
    {
      "regulation": "CBN NIP Framework",
      "jurisdiction": "NG",
      "messages": ["Transaction of ₦15,000,000 exceeds single-transfer cap of ₦10,000,000"],
      "rule_triggered": "cbn_nip_single_transfer_cap",
      "citations": [
        {
          "document": "CBN NIP Framework",
          "section": "Section 4.2",
          "authority": "Central Bank of Nigeria",
          "year": 2021
        }
      ]
    }
  ]
}
```

### Pattern 2 — Automatic pre-execution guard

Wrap existing tools so compliance runs transparently before each call:

```python
from crewai_tools import comply54_guard_tools
from comply54.sectors import NigeriaFintechCompliance

guarded = comply54_guard_tools(
    NigeriaFintechCompliance(),
    [transfer_funds_tool, approve_loan_tool],
    context={"kyc_tier": 3, "customer_verified": True},
)

agent = Agent(role="Fintech Agent", tools=guarded, ...)
```

Or wrap a single tool with `Comply54GuardedTool`:

```python
from crewai_tools import Comply54GuardedTool

guarded = Comply54GuardedTool(
    TransferFundsTool(),
    NigeriaFintechCompliance(),
    context={"kyc_tier": 3},
    block_on_escalate=True,
)
```

### Pattern 3 — Output-level task guardrail

Block PII leakage (BVN, NIN, account numbers) in task output before delivery:

```python
from comply54.crewai import Comply54TaskGuardrail
from comply54.sectors import NigeriaFintechCompliance

guardrail = Comply54TaskGuardrail(NigeriaFintechCompliance())

task = Task(
    description="Summarise the customer account",
    expected_output="Plain text summary",
    guardrail=guardrail,
    agent=agent,
)
```

## Available sector packs

```python
from comply54.sectors import (
    NigeriaFintechCompliance,   # CBN, NDPA 2023, NFIU AML, NIIRA 2025
    NigeriaHealthCompliance,    # NHA, NDPA 2023
    NigeriaInsuranceCompliance, # NAICOM, NDPA 2023
    KenyaCompliance,            # KDPA 2019
    SouthAfricaCompliance,      # POPIA
    PanAfricanCompliance,       # All 21 packs across 12 jurisdictions
)
```

## Environment variables

None required. Comply54 is fully self-contained.

## Links

- [comply54 on PyPI](https://pypi.org/project/comply54/)
- [comply54 docs](https://comply54.io)
- [comply54 on GitHub](https://github.com/kingztech2019/comply54)
