# chDB Tools

Give agents analytical SQL over local and remote data with [chDB](https://clickhouse.com/docs/en/chdb), the in-process ClickHouse engine. The engine runs inside the Python process — no server to start, no connection string, no credentials — and queries local files (Parquet/CSV/JSON), object storage, and remote databases through ClickHouse table functions.

Seven tools are provided, mirroring the tool surface chDB ships for agents (`chdb.agents`):

- **`ChDBRunSelectQueryTool`** (`run_select_query`) — read-only ClickHouse SQL with `{name:Type}` parameter binding.
- **`ChDBListDatabasesTool`** (`list_databases`) — list databases in the session.
- **`ChDBListTablesTool`** (`list_tables`) — list tables in a database.
- **`ChDBDescribeTableTool`** (`describe_table`) — columns and types of a table *or* a table-function expression such as `s3('https://bucket/f.parquet','Parquet')`.
- **`ChDBGetSampleDataTool`** (`get_sample_data`) — a few real rows before querying.
- **`ChDBListFunctionsTool`** (`list_functions`) — discover the 1000+ ClickHouse SQL functions.
- **`ChDBAttachFileTool`** (`attach_file`) — register a local file as a named table (writable sessions only).

## Installation

```shell
uv add "crewai-tools[chdb]"
# or
pip install "crewai-tools[chdb]"
```

No API key or environment variable is needed.

## Quick start

`chdb_tools()` builds the whole suite over **one shared engine**, so what one tool attaches or discovers, the others see:

```python
from crewai import Agent
from crewai_tools import chdb_tools

analyst = Agent(
    role="Data Analyst",
    goal="Answer questions about the events dataset",
    backstory="Expert in ClickHouse SQL",
    tools=chdb_tools(attachments={"events": "data/events.parquet"}),
)
```

A single tool works standalone too:

```python
from crewai_tools import ChDBRunSelectQueryTool

tool = ChDBRunSelectQueryTool()
print(tool.run(sql="SELECT count() FROM file('data/events.parquet')"))
```

## Safety defaults

- **Read-only by default.** The session is locked with the ClickHouse setting `readonly=2`: `INSERT`/`CREATE`/`ALTER`/`DROP` are rejected at the engine (not by prompt), while `SELECT` and the `file()`/`s3()`/`url()` table functions keep working. Pass `read_only=False` for a writable session — `chdb_tools()` then also includes `attach_file`.
- **Capped results.** `max_rows` (default 1000) and `max_bytes` (default 1 MB) bound every payload; truncated results carry a `truncated` flag the agent can react to. `max_execution_time` adds an engine-side wall-clock limit.
- **Optional source allowlist.** With `file_allowlist=["/data/"]`, file-like sources may only read under the listed prefixes and DSN-based sources (`postgresql()`, `mysql()`, `remote()`, …) are refused.
- **Errors go to the model, not the run.** Every call returns a JSON envelope — `{"ok": true, "result": …}` or `{"ok": false, "error": {"code", "type", "message"}}` — so a bad query becomes feedback the agent can correct instead of an exception that kills the crew.

## Querying remote data

ClickHouse table functions work inline, subject to the read-only lock and allowlist:

```python
tools = chdb_tools()
tools[0].run(
    sql="SELECT status, count() FROM url('https://example.com/logs.ndjson', 'JSONEachRow') GROUP BY status"
)
```

`s3(...)`, `postgresql(...)`, `mysql(...)`, `mongodb(...)`, `remoteSecure(...)`, `iceberg(...)`, and `deltaLake(...)` are available the same way.

## Sharing and lifecycle

Tools built by `chdb_tools()` share one engine. To control the engine's lifetime yourself, create it explicitly and inject it:

```python
from chdb.agents import ChDBTool
from crewai_tools import chdb_tools

engine = ChDBTool("analytics_dir", read_only=False)
tools = chdb_tools(engine=engine)
...
engine.close()
```

Tool names default to the canonical chDB agent-tool names (`run_select_query`, `list_tables`, …). If they collide with another tool suite in your crew, rename at construction: `ChDBRunSelectQueryTool(name="chdb_query")`.
