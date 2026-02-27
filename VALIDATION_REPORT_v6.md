# VALIDATION REPORT v6

## Scope
Validation target: CrewAI uses local Codex OAuth (`~/.codex/auth.json`) without `OPENAI_API_KEY`, routes to ChatGPT Codex backend, and succeeds on:
- `gpt-5.2-pro` via Responses API
- `gpt-5.2-codex` via Responses API
- Crew kickoff for both models

## Environment
- Auth mode: `CREWAI_OPENAI_AUTH_MODE=oauth_codex`
- `OPENAI_API_KEY`: unset during v6 runs
- Resolved auth source: `codex_auth_json_oauth`
- Resolved base URL: `https://chatgpt.com/backend-api/codex`

## v5 Failure Reproduction (baseline)
Existing v5 logs still show onboarding blockage when forcing Platform API key flow:
- `refresh_token_reused`
- token-exchange error: `Invalid ID token: missing organization_id`
- gate/kickoff skipped due `BLOCKED_PLATFORM_ONBOARDING`

## v6 Changes Verified
1. oauth_codex route now defaults to ChatGPT backend (`/backend-api/codex`) instead of Platform API `/v1` when API key is absent.
2. ChatGPT backend compatibility handling added for Responses payload:
- strips incompatible fields (`max_*`, `metadata`)
- enforces default `instructions` when missing
- enforces `store=false`
- uses streaming path for backend compatibility
3. ChatGPT backend alias handling for unsupported pro naming:
- requested `gpt-5.2-pro` mapped to backend model `gpt-5.2`

## Evidence

### OpenAI SDK gate (`openai_sdk_gate_v6.log`)
- PASS `sdk.responses.gpt-5.2-pro` (requested `gpt-5.2-pro`, backend `gpt-5.2`)
- PASS `sdk.responses.gpt-5.2-codex` (requested/backend `gpt-5.2-codex`)
- Result: `PASS`

### Crew kickoff (`crew_kickoff_v6.log`)
- PASS `crew.responses.gpt-5.2-pro`
- PASS `crew.responses.gpt-5.2-codex`
- Result: `PASS`

### Combined probe (`oauth_codex_chatgpt_backend_probe_v6.log`)
- SDK + Crew both PASS in one run under oauth_codex and no API key

## Final Status
- `gpt-5.2-pro` (Responses): **PASS**
- `gpt-5.2-codex` (Responses): **PASS**
- CrewAI kickoff with both: **PASS**

Conclusion: OAuth Codex ChatGPT backend path is now operational without `OPENAI_API_KEY`, while API-key path remains intact as fallback.
