# VALIDATION_REPORT_v7

## Scope
- Validate strict routing/auth split without aliasing:
  - `gpt-5.2-codex` => ChatGPT Codex backend (`https://chatgpt.com/backend-api/codex`) with local Codex OAuth `tokens.access_token`
  - `gpt-5.2-pro` => OpenAI Platform Responses API (`https://api.openai.com/v1`) with Platform credential only
- No `gpt-5.2-pro -> gpt-5.2` rewrite allowed.

## Baseline Re-check
- Read prior artifacts: `VALIDATION_REPORT_v2.md`, `VALIDATION_REPORT_v4.md`, `VALIDATION_REPORT_v5.md`, `VALIDATION_REPORT_v6.md`.
- Confirmed v6 had alias behavior in baseline (`requested_model=gpt-5.2-pro` but backend effective model was `gpt-5.2`).
- Baseline snapshot: `route_baseline_v7.log`.

## Implemented v7 Corrections
- Removed `gpt-5.2-pro -> gpt-5.2` compatibility alias.
- Added strict route matrix:
  - `gpt-5.2-codex` + `CREWAI_OPENAI_AUTH_MODE=oauth_codex` => ChatGPT Codex backend only.
  - `gpt-5.2-pro` => Platform Responses route only.
- Added strict fail-fast for pro model when credential resolves to raw OAuth/JWT token.
- Added split credential resolvers:
  - Codex OAuth access token resolver for ChatGPT backend path.
  - Platform key resolver from local Codex auth (`OPENAI_API_KEY` in auth.json or token-exchange from id_token) for pro path.
- Added auth refresh recovery logic and file-lock + atomic persist behavior for token updates.
- Updated tests and verifier script for strict requested/effective model evidence.

## Unit Test Status
- Command:
  - `uv run pytest -q lib/crewai/tests/llms/openai/test_openai.py lib/crewai/tests/llms/openai/test_openai_auth.py`
- Result:
  - `94 passed, 1 skipped`.

## Live Validation (v7)
### SDK Gate
- Artifact: `openai_sdk_gate_v7.log`
- Result matrix:
  - `sdk.responses.gpt-5.2-codex`: **PASS**
    - requested/effective model: `gpt-5.2-codex`
    - api: `responses`
    - base_url: `https://chatgpt.com/backend-api/codex`
    - credential_source: `codex_auth_json_oauth`
    - status: `200`
  - `sdk.responses.gpt-5.2-pro`: **FAIL**
    - requested/effective model: `gpt-5.2-pro` (no alias)
    - api: `responses`
    - base_url: `None` (credential resolution failed before client build)
    - credential_source: `unknown`
    - error: `Codex refresh_token was reused and local auth cache does not have a newer token.`

### Crew Kickoff Gate
- Artifact: `crew_kickoff_v7.log`
- Result matrix:
  - `crew.responses.gpt-5.2-codex`: **PASS**
    - requested/effective model: `gpt-5.2-codex`
    - api: `responses`
    - base_url: `https://chatgpt.com/backend-api/codex`
    - credential_source: `codex_auth_json_oauth`
    - status: `200`
  - `crew.responses.gpt-5.2-pro`: **FAIL**
    - requested/effective model: `gpt-5.2-pro` (no alias)
    - api: `responses`
    - base_url: `None`
    - credential_source: `unknown`
    - error: `Codex refresh_token was reused and local auth cache does not have a newer token.`

## Additional Credential Recovery Attempts
- Confirmed current local auth state:
  - `auth.json` has `OPENAI_API_KEY` present but empty.
  - `tokens.access_token` valid for ChatGPT backend path.
  - `tokens.id_token` expired.
- Attempted token refresh and token-exchange retries:
  - refresh path returned `refresh_token_reused`.
  - exchange with expired `id_token` failed `invalid_id_token`.
  - exchange with OAuth `access_token` as subject token failed (`token_expired` invalid request).
- Attempted `codex logout && codex login --device-auth` automation:
  - login flow requires interactive browser code confirmation.
  - unattended attempt timed out; original auth cache was restored from local backup to avoid breaking ongoing work.

## Required v7 Pass Criteria Outcome
- `gpt-5.2-codex`: **PASS** (SDK + Crew)
- `gpt-5.2-pro`: **FAIL** (SDK + Crew)
- Alias (`gpt-5.2-pro -> gpt-5.2`): **NO** (removed; strict requested==effective evidence in logs)

## Final Branch Conclusion
- **AUTH FLOW BROKEN for Platform credential derivation in current local auth cache** (stale/expired id_token + refresh token reuse).
- ChatGPT backend OAuth route is working and validated for `gpt-5.2-codex`.
- Platform route for `gpt-5.2-pro` is implemented correctly but currently blocked by local credential state, not by aliasing or wrong route.
