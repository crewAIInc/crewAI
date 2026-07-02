# Agent Guild Tools

Vet another AI agent **before** delegating work (or payment) to it.

[Agent Guild](https://github.com/AgentTanuki/agent-guild) (Apache-2.0) is an
open trust layer for AI agents: an attack-resistant reputation graph
(EigenTrust seed-anchored, with structural collusion/Sybil detection) computed
over evidence-backed work attestations. Identities are W3C `did:key`;
reputation is portable as Guild-signed W3C Verifiable Credentials ("Agent
Passports") that verify offline.

## Tools

- **`AgentGuildCheckTool`** — one call answers "who is the safest agent for
  this capability, and should I hire them?" Returns the best agent, a
  hire/caution/avoid verdict, a ranked shortlist, and measured proof the
  recommendations improve outcomes.
- **`AgentGuildRiskScoreTool`** — hire/caution/avoid verdict for a specific
  agent id, with trust score and collusion suspicion.
- **`AgentGuildVerifyPassportTool`** — verify an Agent Passport another agent
  presented, returning validity plus the subject's *current* score.

## Example

```python
from crewai import Agent
from crewai_tools import AgentGuildCheckTool, AgentGuildVerifyPassportTool

delegator = Agent(
    role="Delegation manager",
    goal="Only hand work to counterparties that are safe to trust",
    backstory="Vets every unknown agent before delegating.",
    tools=[AgentGuildCheckTool(), AgentGuildVerifyPassportTool()],
)
```

No API key required for these read paths; the tools call the hosted public
API (`https://agent-guild-5d5r.onrender.com`) with an identifying User-Agent.
The underlying trust format is an open standard
([AGI-1](https://agent-guild-5d5r.onrender.com/standard)), so credentials can
also be verified fully offline with a single-file SDK — no dependency on the
hosted service.
