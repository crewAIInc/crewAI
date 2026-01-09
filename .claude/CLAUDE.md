# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a Claude Code Skills repository containing custom skills that extend Claude Code's capabilities. Skills are knowledge modules that Claude Code can invoke when specific conditions are met.

## Skill Structure

Each skill lives in `skills/<skill-name>/` with this structure:
```
skills/<skill-name>/
├── SKILL.md           # Required: frontmatter (name, description) + knowledge content
└── references/        # Optional: supplementary reference files
    └── *.md
```

### SKILL.md Format

```markdown
---
name: skill-name
description: |
  Multi-line description of when to use this skill.
  Include specific trigger conditions.
---

# Main content follows...
```

The `description` field is critical - it tells Claude Code **when** to activate this skill. Be specific about trigger conditions.

## Writing Effective Skills

1. **Trigger Description**: Write clear, specific conditions in the YAML `description` field
2. **Actionable Content**: The markdown body should be directly usable guidance, not abstract theory
3. **Code Examples**: Include working code patterns that can be adapted
4. **Reference Files**: Split detailed reference material into `references/*.md` and link from SKILL.md

## Current Skills

- **crewai-architect**: Flow-first design patterns for CrewAI applications
- **crewai-enterprise-endpoint-manager**: REST API integration for deployed CrewAI crews/flows
- **crewai-tool-creator**: Custom tool development following best practices
- **software-architect**: Clean code with SOLID principles
- **streamlit**: Building interactive Python web apps
