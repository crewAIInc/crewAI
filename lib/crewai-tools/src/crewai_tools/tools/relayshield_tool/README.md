# RelayShield Tools

Two agent-specific security checks from [RelayShield](https://api.relayshield.net/developers), a live threat-intelligence API.

## RelayShieldMCPRiskTool

Checks an MCP server URL or package name for:
- Typosquat/near-miss matches against known MCP ecosystem domains
- Presence in RelayShield's criminal IOC corpus
- Domain-registration age (newly registered domains are higher risk)

```python
from crewai_tools import RelayShieldMCPRiskTool

tool = RelayShieldMCPRiskTool()
result = tool.run(server_url="https://example.com/mcp")
```

## RelayShieldPromptInjectionBreachTool

Checks whether an email's credentials were exposed via a breach sourced specifically from a prompt-injection attack against an AI agent (distinct from ordinary phishing/malware-sourced breaches).

```python
from crewai_tools import RelayShieldPromptInjectionBreachTool

tool = RelayShieldPromptInjectionBreachTool()
result = tool.run(email="user@example.com")
```

## Setup

Both tools require a RelayShield API key:

```bash
export RELAYSHIELD_API_KEY="your-key-here"
```

Get a key at [api.relayshield.net/developers](https://api.relayshield.net/developers) ($499/mo for 10,000 calls, or PAYG via x402 USDC with no key required — see docs).
