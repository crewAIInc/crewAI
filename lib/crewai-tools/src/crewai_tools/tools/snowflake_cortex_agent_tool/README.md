# Snowflake Cortex Agent Tool

Delegate natural language data questions to a [Snowflake Cortex Agent](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-agents) so the planning, retrieval, text-to-SQL, and reasoning all happen inside Snowflake's secure perimeter. Your CrewAI agent only orchestrates — it does not need to write SQL or pick between Cortex Analyst and Cortex Search itself.

## Why use this tool?

- Keep semantic models, role-based access, and governance in Snowflake.
- Get high-quality text-to-SQL via Cortex Analyst on structured data.
- Get retrieval over unstructured documents via Cortex Search.
- Let the Cortex Agent decide when to use which tool and reflect on the result.
- The CrewAI agent picks this tool whenever a question is best answered with governed Snowflake data.

## Authentication

The tool calls the Cortex Agents REST API and authenticates with a bearer token. The recommended option is a [Snowflake programmatic access token (PAT)](https://docs.snowflake.com/en/user-guide/programmatic-access-tokens). OAuth tokens and JWTs are also accepted.

You can pass the token directly via `auth_token`, or set the `SNOWFLAKE_CORTEX_AGENT_TOKEN` environment variable. Similarly, the Snowflake account identifier can be passed via `account` or the `SNOWFLAKE_ACCOUNT` environment variable.

## Quick start (referencing an existing agent object)

```python
from crewai import Agent, Task, Crew
from crewai_tools import SnowflakeCortexAgentTool

cortex_agent = SnowflakeCortexAgentTool(
    account="myorg-myaccount",
    auth_token="<programmatic-access-token>",
    database="MY_DB",
    snowflake_schema="MY_SCHEMA",
    agent_name="SALES_AGENT",
)

analyst = Agent(
    role="Sales analyst",
    goal="Answer revenue questions using governed Snowflake data",
    backstory="An analyst that knows when to defer to Snowflake.",
    tools=[cortex_agent],
)

task = Task(
    description="What was total revenue last quarter, broken down by region?",
    expected_output="A short summary with regional totals.",
    agent=analyst,
)

Crew(agents=[analyst], tasks=[task]).kickoff()
```

## Quick start (without an agent object)

If you have not pre-created an agent object in Snowflake, you can describe the agent's tools inline:

```python
tool = SnowflakeCortexAgentTool(
    account="myorg-myaccount",
    auth_token="<programmatic-access-token>",
    tools=[
        {
            "tool_spec": {
                "type": "cortex_analyst_text_to_sql",
                "name": "analyst_tool",
            }
        },
        {
            "tool_spec": {
                "type": "cortex_search",
                "name": "search_tool",
            }
        },
    ],
    tool_resources={
        "analyst_tool": {"semantic_model_file": "@MY_DB.MY_SCHEMA.SEMANTIC_MODELS/sales.yaml"},
        "search_tool": {"name": "MY_DB.MY_SCHEMA.MY_SEARCH_SVC"},
    },
    models={"orchestration": "claude-4-sonnet"},
    instructions={
        "response": "Respond concisely with citations when available.",
    },
    tool_choice={"type": "auto"},
)

print(tool.run(query="What is the total revenue for 2025?"))
```

## Configuration

| Parameter | Required | Description |
|-----------|----------|-------------|
| `account` | one of `account`/`host` | Snowflake account identifier (e.g. `myorg-myaccount`). Falls back to the `SNOWFLAKE_ACCOUNT` environment variable. |
| `host` | one of `account`/`host` | Override the API hostname (e.g. for Snowflake private link). Takes precedence over `account`. |
| `auth_token` | yes | Bearer token (PAT, OAuth, or JWT). Falls back to `SNOWFLAKE_CORTEX_AGENT_TOKEN`. |
| `database`, `snowflake_schema`, `agent_name` | when referencing an agent object | All three must be set together to call the agent-object endpoint. |
| `tools` | when running without an agent object | List of tool specifications (Cortex Analyst, Cortex Search, custom). |
| `tool_resources` | optional | Per-tool resource configuration keyed by tool name. |
| `tool_choice` | optional | Tool selection policy (`{"type": "auto"}`, `{"type": "required", "name": [...]}`). |
| `models` | optional | Model configuration (e.g. `{"orchestration": "claude-4-sonnet"}`). |
| `instructions` | optional | Agent instructions (`response`, `orchestration`, `system`, `sample_questions`). |
| `orchestration` | optional | Orchestration configuration such as budget constraints. |
| `timeout` | optional | Per-request timeout in seconds (default 600; the server itself times out at 15 minutes). |

## Notes

- The tool sends `stream: false` and parses the single JSON response. The first textual content item is returned to the calling agent; the full JSON is returned as a fallback when no text content is present (for example, when the agent only calls tools).
- If both an agent object and inline `tools` are configured, the tool calls the agent-object endpoint and ignores the inline configuration, since the agent object's stored tools are authoritative.
- HTTP errors are returned as a string starting with `Snowflake Cortex Agent returned HTTP ...` so the calling agent can react instead of raising.
