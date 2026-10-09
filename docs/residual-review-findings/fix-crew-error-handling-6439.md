# Residual Review Findings

Source: ce-code-review run `20260817-233222-aa369bf8` on branch `fix/crew-error-handling-6439`, head `6dce35b`.
Plan: `docs/plans/2026-08-17-001-fix-crew-error-handling-plan.md`.

## Findings

- **P2 (settled_conflict: KTD3)** - `lib/crewai/src/crewai/task.py:120` - Third+ copy of run-coroutine-from-sync idiom. The new `_run_awaitable_from_sync` duplicates the await-response pattern in `lib/crewai/src/crewai/utilities/agent_utils.py:99-128` (and ~7 other files). Extract into a shared utility module (e.g. `lib/crewai/src/crewai/utilities/async_utils.py`) and implement both callers on top of it. Defer failed: GitHub issue creation returned HTTP 503 (GitHub API service outage) - `gh returned 503: No server is currently available`. Recorded here as the durable sink. This is preference-grade against session-settled KTD3 (the in-module mirror was chosen deliberately to avoid importing agent_utils into task.py), so it is report-only and deferred for future consolidation.

## Source run context

- Reviewers: correctness, project-standards, testing, reliability, maintainability, adversarial-codex (cross-model, independence verified)
- Actionable findings applied in `6dce35b`: #1 (P1 loop-bound Task/Future guard), #2 (P2 async akickoff replace-result test), #4 (P2 crew.task_callback async path test)
- Verdict: Ready with fixes
