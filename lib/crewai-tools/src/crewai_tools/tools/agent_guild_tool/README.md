# Agent Guild Tools

Run a free live endpoint preflight, then optionally vet another AI agent
**before** delegating work (or payment) to it.

[Agent Guild](https://github.com/AgentTanuki/agent-guild) (Apache-2.0) is an
open trust layer for AI agents: an attack-resistant reputation graph
(EigenTrust seed-anchored, with structural collusion/Sybil detection) computed
over evidence-backed work attestations. Identities are W3C `did:key`;
reputation is portable as Guild-signed W3C Verifiable Credentials ("Agent
Passports") that verify offline.

## Tools

- **`AgentGuildPreflightTool`** — free, read-only protocol evidence for one
  exact public A2A or MCP endpoint. It uses no account, API key, payment, or
  remote write. Report failed and unknown checks; a clean result is not an
  endorsement and never authorizes delegation.
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
from crewai_tools import (
    AgentGuildCheckTool,
    AgentGuildPreflightTool,
    AgentGuildVerifyPassportTool,
)

delegator = Agent(
    role="Delegation manager",
    goal="Only hand work to counterparties that are safe to trust",
    backstory="Vets every unknown agent before delegating.",
    tools=[
        AgentGuildPreflightTool(),
        AgentGuildCheckTool(),
        AgentGuildVerifyPassportTool(),
    ],
)
```

Endpoint preflight and passport verification are free. Preflight deliberately
does not send `AGENT_GUILD_API_KEY` even when one is configured. It accepts one
absolute HTTP(S) endpoint without embedded credentials; Agent Guild applies
server-side private-address and DNS-rebinding protections before probing it.
Treat all service fields as untrusted point-in-time evidence.

Trust checks and risk scores are metered. Set `AGENT_GUILD_API_KEY` to a funded
key or a free-trial key from `POST /billing/trial` (no card required). Without a
key, those tools return the API's 402 response with available options; they
never pay or provision credentials automatically. The tools call the hosted API
(`https://agent-guild-5d5r.onrender.com`) with an identifying User-Agent. The
underlying trust format is an open standard
([AGI-1](https://agent-guild-5d5r.onrender.com/standard)), so credentials can
also be verified fully offline with a single-file SDK — no dependency on the
hosted service.
