# PR: Add Mekong x402 tools (pay-per-call APIs, no API key)

**Title:** `feat(tools): add Mekong x402 pay-per-call tool family`

**Target repo:** `crewAIInc/crewAI` → `lib/crewai-tools/src/crewai_tools/tools/mekong_x402_tool/`
**Files:** `__init__.py`, `mekong_x402_tool.py`, `schemas.py`, `README.md`

---

## What

7 native CrewAI tools for the [Mekong x402 fleet](https://api.mekongtradeintel.com/tools/) —
37 pay-per-call APIs on Base (USDC):

- `MekongTaskRouterTool` — **FREE** task router: describe a task in plain language → top-3 endpoints + curls
- `MekongWebSearchTool` — live web search ($0.01)
- `MekongTokenPriceTool` — price/mcap/liquidity ($0.001)
- `MekongAIVerdictTool` — token safety verdict ($0.01)
- `MekongEnrichTool` — company enrichment ($0.02)
- `MekongSTTTool` — speech-to-text ($0.05)
- `MekongBriefTool` — news+trends+sentiment bundle ($0.05)

## Why this belongs in crewai-tools

- **Zero-config for the agent operator:** no API keys, no signup, no env vars.
  Every endpoint has a free daily trial quota per IP — the tools work out of the box.
- **Honest paid path:** when the trial is exhausted the tool raises
  `MekongPaymentRequired` with the exact price, `payTo`, asset, network and the
  3-step x402 flow. Never a silent charge, never a fake success.
- **Independently verifiable:** https://api.mekongtradeintel.com/tools/verify —
  one-command script checking 402 paywall shape, CDP `/x402/validate` preflight
  and on-chain settlement from the outside.

## Tested

- All 7 tools instantiate; free router tool run live against production
  (`task="Is this token a honeypot?"` → `rugcheck` top-1, correct).
- 402 path unit-tested with mocked responses (raises with pay instructions).
- Follows repo conventions: `BaseTool` + `args_schema` pydantic models in
  `schemas.py`, `_run(**kwargs)`, per-tool README.

## Notes for reviewers

- No `env_vars`: there is deliberately nothing to configure — that is the point
  of x402 (the wallet is the credential).
- The fleet is operated by Mekong Trade Intelligence; endpoints are plain HTTPS
  + x402 v2 (`exact` scheme, USDC on Base `eip155:8453`).
- Happy to split into smaller PRs per tool if preferred.
