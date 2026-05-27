# CLAUDE.md — AI-Assisted Development Guidelines

This file provides guidance for AI coding agents (Claude Code, Codex, etc.) working on the crewAI codebase.

## Dependency Management

**Do NOT use `override-dependencies`** (e.g. in `pyproject.toml` or similar) to resolve security audit or dependency issues.

This approach:
- Does not carry over to downstream projects that depend on crewAI
- Can cause dependency conflicts that are potentially unresolvable

**Instead:** Fix dependency issues by updating the actual dependency versions directly. Pin or bump the real dependency so the fix propagates correctly to all consumers.
