# Claude Code Skills

A collection of custom skills that extend Claude Code's capabilities with specialized domain knowledge.

## What Are Skills?

Skills are knowledge modules that Claude Code automatically activates when working on specific tasks. Each skill contains:

- **Trigger conditions** - When to activate (defined in YAML frontmatter)
- **Expert knowledge** - Patterns, best practices, and code examples
- **Reference materials** - Deep-dive documentation for complex topics

## Available Skills

| Skill | Description |
|-------|-------------|
| **[crewai-architect](skills/crewai-architect/)** | Flow-first design patterns for CrewAI applications. Covers direct LLM calls, single agents, and crews within flows. |
| **[crewai-enterprise-endpoint-manager](skills/crewai-enterprise-endpoint-manager/)** | REST API integration for deployed CrewAI crews/flows in Enterprise (AOP). Authentication, monitoring, and result retrieval. |
| **[crewai-tool-creator](skills/crewai-tool-creator/)** | Custom tool development following CrewAI and Anthropic best practices. Input schemas, error handling, caching. |
| **[software-architect](skills/software-architect/)** | Clean code with SOLID principles. Single responsibility, dependency injection, interface segregation. |
| **[streamlit](skills/streamlit/)** | Building interactive Python web apps. Widgets, layouts, caching, session state, multipage apps. |

## Using These Skills

### Option 1: Clone to Claude Code Skills Directory

```bash
# Clone into Claude Code's skills directory
git clone https://github.com/YOUR_USERNAME/skills.git ~/.claude/skills/my-skills
```

Skills are automatically discovered and activated based on their trigger descriptions.

### Option 2: Reference in Project

Add to your project's `.claude/settings.json`:

```json
{
  "skills": {
    "paths": ["/path/to/this/repo/skills"]
  }
}
```

## Skill Structure

```
skills/<skill-name>/
├── SKILL.md           # Required: frontmatter + knowledge content
└── references/        # Optional: supplementary deep-dives
    └── *.md
```

### SKILL.md Format

```markdown
---
name: skill-name
description: |
  When to activate this skill.
  Be specific about trigger conditions.
---

# Main Content

Actionable guidance, patterns, and code examples.
```

## Contributing a New Skill

1. Create a folder: `skills/<your-skill-name>/`
2. Add `SKILL.md` with:
   - YAML frontmatter (`name`, `description`)
   - Practical, actionable content
   - Working code examples
3. Optionally add `references/*.md` for detailed documentation
4. Submit a PR

### Tips for Effective Skills

- **Trigger Description**: Be specific about *when* the skill should activate
- **Actionable Content**: Include patterns and examples, not just theory
- **Code Examples**: Provide copy-paste-ready snippets
- **Reference Files**: Split detailed material into separate files

## Repository Structure

```
.
├── CLAUDE.md          # Instructions for Claude Code
├── README.md          # This file
└── skills/            # Skill modules
    ├── crewai-architect/
    ├── crewai-enterprise-endpoint-manager/
    ├── crewai-tool-creator/
    ├── software-architect/
    └── streamlit/
```

## License

MIT
