# crewai-devtools

CLI for versioning and releasing crewAI packages.

## Setup

Installed automatically via the workspace (`uv sync`). Requires:

- [GitHub CLI](https://cli.github.com/) (`gh`) — authenticated
- `OPENAI_API_KEY` env var — for release note generation and translation
- `ENTERPRISE_REPO` env var — GitHub repo for enterprise releases
- `ENTERPRISE_VERSION_DIRS` env var — comma-separated directories to bump in the enterprise repo
- `ENTERPRISE_CREWAI_DEP_PATH` env var — path to the pyproject.toml with the `crewai[tools]` pin in the enterprise repo
- `ENTERPRISE_WORKFLOW_PATHS` env var — comma-separated workflow file paths in the enterprise repo whose `crewai[extras]==<version>` pins should be rewritten on each release (e.g. `.github/workflows/tests.yml`)
- `ENTERPRISE_EXTRA_PACKAGES` env var — comma-separated packages to also pin in enterprise pyproject files, in addition to `crewai` / `crewai[extras]`

## Commands

### `devtools release <version>`

Full end-to-end release. Bumps versions, creates PRs, tags, publishes a GitHub release, and releases the enterprise repo.

```
devtools release 1.10.3
devtools release 1.10.3a1                # pre-release
devtools release 1.10.3 --no-edit        # skip editing release notes
devtools release 1.10.3 --dry-run        # preview without changes
devtools release 1.10.3 --skip-enterprise  # skip enterprise release phase
```

**Flow:**

1. Bumps `__version__` and dependency pins across all `lib/` packages
2. Runs `uv sync`
3. Creates version bump PR against main, polls until merged
4. Generates release notes (OpenAI) from commits since last release
5. Updates changelogs (en, pt-BR, ko) and docs version switcher
6. Creates docs PR against main, polls until merged
7. Tags main and creates GitHub release
8. Triggers PyPI publish workflow
9. Clones enterprise repo, bumps versions and `crewai[tools]` dep, runs `uv sync`
10. Creates enterprise bump PR, polls until merged
11. Tags and creates GitHub release on enterprise repo

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