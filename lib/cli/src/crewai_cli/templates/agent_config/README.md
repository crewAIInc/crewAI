# {{package_name}}

An experimental general-purpose CrewAI Agent with three local example tools and an empty skills directory.
The example describes a fictional service and does not access real accounts or external systems.
Standalone agent project formats and CLI support may change. Use matching preview CLI and SDK builds that include this feature.

YAML is the default for `crewai create agent {{package_name}}`.
Choose JSON with `--format json`, or a Python Agent with `--format python`.
The default model is `openai/gpt-6.1-sol` through the Responses API.

```text
{{package_name}}/
├── {{definition}}
├── pyproject.toml
├── README.md
├── .env.example
├── .gitignore
├── tools/
│   ├── __init__.py
│   ├── support_hours.py
│   ├── lookup_plan.py
│   └── support_checklist.py
└── skills/
    └── .gitkeep
```

## Run the reference Agent

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`, then run:

```sh
crewai install
crewai run
crewai run --inputs '{"messages":"Use support_hours to tell me when support is available."}'
crewai run --inputs '{"messages":"I use the Pro plan and cannot sign in. Use the available tools to explain what I should try and when support is available. Write a short customer reply."}'
crewai deploy validate
```

`crewai install` creates `uv.lock`. Commit it before deploying; keep `.env` local.
Edit `llm.model` and `llm.api` in `{{definition}}` to choose another model and endpoint.
`MODEL` in `.env` is used by the Python template; config projects use the definition's `llm` fields.

## Customize the Agent

The default identity is general-purpose. Edit its role, goal, and backstory to specialize it, and add, replace, or remove tools and skills to match the task. The included support tools apply only when relevant. No skills are installed by default.

## Example tools and skills

| Tool | Input | Example result |
| --- | --- | --- |
| `support_hours` | None | Monday to Friday, 09:00-17:00 UTC |
| `lookup_plan` | `plan`: `starter` or `pro` | Seat allowance and support channel |
| `support_checklist` | `topic`: `sign-in` or `billing` | Troubleshooting steps |

The plan and checklist tools demonstrate typed Pydantic argument schemas and JSON results.
Unknown values return the available choices. Replace these example facts with your own application logic.

The `skills/` directory starts empty, with only `.gitkeep` to preserve it in Git and packaged builds.
Add your own `skills/<name>/SKILL.md` with `name` and `description` frontmatter.
Once added, skills begin as a metadata catalog; the Agent can call `load_skill` to read relevant instructions on demand.

The definition explicitly registers `custom:<name>` tools from `tools/<name>.py` and discovers skills under `./skills`.
Local skill paths resolve against the project root, including when using `load_agent_project(path)` from another directory.
Add another `BaseTool` subclass in its own file and register its `custom:` reference to extend the tools.
Add `skills/<name>/SKILL.md` with `name` and `description` frontmatter to extend the skill catalog.
Custom tools and Python references execute project code; load only trusted projects.

The wheel and deployment archive include the definition, tools, and skills. No `main.py` or source package is required.
The scaffold enables `conversational: true` (`Agent(conversational=True)` in Python).
In a terminal, `crewai run` opens interactive chat and retains history for that session.
`crewai run --inputs` performs one independent kickoff and accepts a string or list of role/content messages.
From Python, load the project and call `agent.handle_turn("Hello")` for each turn; `agent.as_flow()` exposes its Flow session.
Local chat history is in memory by default. Configure Flow persistence explicitly for recovery across processes; managed deployments use their server-owned session storage.
Set `conversational` to false to disable chat and use independent kickoffs only.
