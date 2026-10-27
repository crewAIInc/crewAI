# {{package_name}}

A standalone CrewAI agent defined in `{{definition}}`.

```text
{{package_name}}/
├── pyproject.toml
├── .env.example
└── {{definition}}
```

Copy `.env.example` to `.env` and set your model provider key, then run:

```sh
crewai install
crewai run --inputs '{"messages":"What are the support hours?"}'
crewai deploy validate
```

Edit `{{definition}}` to configure the agent. The local runner and managed runtime
use the same definition; no Python entrypoint or source package is required.
`messages` accepts a string or a list of role/content messages.
For a local custom tool, add `tools/support_hours.py` and reference its exported
`BaseTool` subclass with `tools: ["custom:support_hours"]` (JSON uses the same list).
The `tools/` directory is included in the package when present.
Each kickoff creates a fresh agent; this template does not persist conversation history.
Commit `uv.lock` after installation before deploying.
