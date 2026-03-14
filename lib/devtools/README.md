# crewai-devtools

CLI for versioning and releasing crewAI packages.

## Setup

Installed automatically via the workspace (`uv sync`). Requires:

- [GitHub CLI](https://cli.github.com/) (`gh`) — authenticated
- `OPENAI_API_KEY` env var — for release note generation and translation

## Commands

### `devtools release <version>`

Full end-to-end release. Bumps versions, creates PRs, tags, and publishes a GitHub release.

```
devtools release 1.10.3
devtools release 1.10.3a1      # pre-release
devtools release 1.10.3 --no-edit   # skip editing release notes
devtools release 1.10.3 --dry-run   # preview without changes
```

**Flow:**

1. Bumps `__version__` and dependency pins across all `lib/` packages
2. Runs `uv sync`
3. Creates version bump PR against main, polls until merged
4. Generates release notes (OpenAI) from commits since last release
5. Updates changelogs (en, pt-BR, ko) and docs version switcher
6. Creates docs PR against main, polls until merged
7. Tags main and creates GitHub release

### `devtools bump <version>`

Bump versions only (phase 1 of `release`).

```
devtools bump 1.10.3
devtools bump 1.10.3 --no-push      # don't push or create PR
devtools bump 1.10.3 --no-commit    # only update files
devtools bump 1.10.3 --dry-run
```

### `devtools tag`

Tag and release only (phase 2 of `release`). Run after the bump PR is merged.

```
devtools tag
devtools tag --no-edit
devtools tag --dry-run
```