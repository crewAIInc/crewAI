---
title: MandateZ Security Integration
description: Add cryptographic identity, OWASP Agentic Top 10 compliance, and tamper-proof audit trails to CrewAI agents.
---

# MandateZ Security Integration for CrewAI

[MandateZ](https://mandatez.mintlify.app) is the neutral trust infrastructure layer for AI agents. It gives every CrewAI agent a cryptographic identity, policy enforcement, human oversight gates, and a tamper-proof audit trail — addressing the [OWASP Agentic AI Top 10](https://owasp.org/www-project-agentic-ai-top-10/) security risks.

## Install

```bash
pip install crewai
npm install @mandatez/sdk
```

## Setup

```typescript
import { generateAgentIdentity, MandateZClient } from '@mandatez/sdk';

// 1. Generate a cryptographic identity for your agent
const identity = await generateAgentIdentity();

// 2. Initialize the MandateZ client
const client = new MandateZClient({
  agentId:         identity.agent_id,
  ownerId:         'your_org_id',
  privateKey:      identity.private_key,
  supabaseUrl:     process.env.SUPABASE_URL!,
  supabaseAnonKey: process.env.SUPABASE_ANON_KEY!,
});
```

## Wrapping a CrewAI Agent

Track every action your CrewAI agent performs with a signed, tamper-proof event:

```typescript
// Track agent actions with cryptographic signatures
const event = await client.track({
  action_type: 'read',
  resource: 'customer-database',
});
// → Ed25519-signed event emitted to your audit stream

// Track writes
await client.track({
  action_type: 'write',
  resource: 'crm/contacts',
});

// Track API calls
await client.track({
  action_type: 'call',
  resource: 'api/sendgrid',
});
```

## OWASP Agentic AI Top 10 Coverage

MandateZ addresses these OWASP Agentic AI risks for CrewAI deployments:

| OWASP Risk | MandateZ Control |
|---|---|
| **A01 — Agent Identity Spoofing** | Ed25519 cryptographic identity per agent |
| **A02 — Unauthorized Actions** | Policy engine with allow/block/flag rules |
| **A03 — Missing Audit Trails** | Every action signed and logged to tamper-proof stream |
| **A04 — No Human Oversight** | Approval gates with Slack/webhook alerts, auto-block on timeout |
| **A05 — Uncontrolled Data Access** | Resource-level policy enforcement with wildcard matching |

## Policy Enforcement

Define rules that control what your CrewAI agents can and cannot do:

```typescript
// Policies are enforced before any action executes
// Actions matching 'flag' rules pause for human approval
// Actions matching 'block' rules are rejected immediately
const oversight = {
  require_human_approval: ['export', 'delete', 'payment'],
  alert_channel: 'slack',
  timeout_seconds: 300,
  timeout_action: 'block',
};
```

## Links

- [MandateZ Documentation](https://mandatez.mintlify.app)
- [npm: @mandatez/sdk](https://www.npmjs.com/package/@mandatez/sdk)
- [GitHub: mandatez/core](https://github.com/mandatez/core)
