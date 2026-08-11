# Sync Docs Translations

After English documentation changes, sync the same updates to Arabic (`ar`),
Korean (`ko`), and Brazilian Portuguese (`pt-BR`).

Supported locales: `ar`, `ko`, `pt-BR`.

## Step 1 — Find changed English files with git

From the repo root:

```bash
# Uncommitted changes (staged or unstaged)
git diff --name-only HEAD -- docs/edge/en/

# All changes on this branch vs main
git diff --name-only main...HEAD -- docs/edge/en/

# Newly added files
git status --porcelain docs/edge/en/
```

Only process `*.mdx` under `docs/edge/en/`. Do not edit `docs/v*/` snapshots.

## Step 2 — Map each file to locale targets

For `docs/edge/en/<path>.mdx`, update or create:

- `docs/edge/ar/<path>.mdx`
- `docs/edge/ko/<path>.mdx`
- `docs/edge/pt-BR/<path>.mdx`

If English is a **new page**, also add matching entries in `docs/docs.json`
navigation for each locale.

## Step 3 — Translate

Use the updated English file as source of truth. When locale files already
exist, apply the same semantic change — do not rewrite unrelated sections.

Rules:

- Translate prose and frontmatter values (`title`, `description`, `sidebarTitle`)
- Keep MDX/JSX tags, code blocks, URLs, and identifiers unchanged
- Keep terms like Agent, Crew, Task, Flow, LLM, API, CLI, MCP in English where
  appropriate
- Rewrite internal links: `/en/` → `/{lang}/` (`/ar/`, `/ko/`, `/pt-BR/`)
- Do not add translator notes

## Step 4 — Verify (optional)

```bash
cd docs && mintlify broken-links
```

Commit English and locale files together.

## Checklist

```markdown
- [ ] Git: listed changed docs/edge/en/*.mdx files
- [ ] ar: updated/created matching files
- [ ] ko: updated/created matching files
- [ ] pt-BR: updated/created matching files
- [ ] Links use /{lang}/ prefix; code blocks unchanged
- [ ] docs/docs.json updated if new English page added
```

## Example

`git diff --name-only HEAD -- docs/edge/en/` returns:

```text
docs/edge/en/concepts/llms.mdx
```

Update:

- `docs/edge/ar/concepts/llms.mdx`
- `docs/edge/ko/concepts/llms.mdx`
- `docs/edge/pt-BR/concepts/llms.mdx`
