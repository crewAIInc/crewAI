"""Prompt templates for AI-generated content."""

from string import Template


RELEASE_NOTES_PROMPT = Template(
    """Generate concise release notes for version $version based on these commits:

$commits

The commits follow the Conventional Commits standard (feat:, fix:, chore:, etc.).

Use this exact template format:

## What's Changed

### Features
- [List feat: commits here, using imperative mood like "Add X", "Implement Y"]

### Bug Fixes
- [List fix: commits here, using imperative mood like "Fix X", "Resolve Y"]

### Documentation
- [List docs: commits here, using imperative mood like "Update X", "Add Y"]

### Performance
- [List perf: commits here, using imperative mood like "Improve X", "Optimize Y"]

### Refactoring
- [List refactor: commits here, using imperative mood like "Refactor X", "Simplify Y"]

### Breaking Changes
- [List commits with BREAKING CHANGE in footer or ! after type, using imperative mood]$contributors_section

Instructions:
- Parse conventional commit format (type: description or type(scope): description)
- Only include sections that have relevant changes from the commits
- Skip chore:, ci:, test:, and style: commits unless significant
- Convert commit messages to imperative mood if needed (e.g., "adds" → "Add")
- Be concise but informative
- Focus on user-facing changes
- Use the exact Contributors list provided above, do not modify it

Keep it professional and clear."""
)
