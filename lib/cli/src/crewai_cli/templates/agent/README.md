# {{package_name}}

An experimental standalone agent with one local tool, ready to run locally or in a managed deployment.
Standalone agent project formats and CLI support may change.

Create this Python layout explicitly with `crewai create agent {{package_name}} --format python`.
The default `crewai create agent {{package_name}}` creates a YAML definition instead.

```text
{{package_name}}/
├── pyproject.toml
├── .env.example
└── src/{{package_name}}/
    ├── __init__.py
    └── agent.py
```

Copy `.env.example` to `.env` and set your model provider key, then run:

```sh
crewai install
crewai run --inputs '{"messages":"Use your tool to tell me support hours."}'
crewai deploy validate
```

Edit `src/{{package_name}}/agent.py` to configure the exported `agent` and its tools.
The local runner and managed runtime load the same `agent = Agent(...)` definition.
`messages` accepts a string or a list of role/content messages.
Each kickoff creates a fresh agent; this template does not persist conversation history.
Commit `uv.lock` after installation before deploying.
