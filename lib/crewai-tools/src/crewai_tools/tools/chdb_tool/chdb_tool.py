from __future__ import annotations

import json
import threading
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


def _import_chdb_tool_class() -> Any:
    """Import chdb.agents.ChDBTool lazily so crewai-tools imports without chdb."""
    try:
        from chdb.agents import ChDBTool
    except ImportError as exc:
        raise ImportError(
            "The 'chdb' package is required for the chDB tools. "
            "Install it with: uv add chdb  (or) pip install 'crewai-tools[chdb]'"
        ) from exc
    return ChDBTool


class ChDBBaseTool(BaseTool):
    """Shared base for tools that run against one in-process chDB engine.

    chDB embeds the ClickHouse engine in the Python process: there is no
    server to start, no connection string, and no credentials. Each tool
    lazily creates its own chDB session from the config fields on first use,
    or reuses an existing ``chdb.agents.ChDBTool`` passed as ``engine``.
    Tools that must see each other's state (``attach_file`` followed by
    ``run_select_query``) have to share one engine — build the suite with
    ``chdb_tools()`` or pass the same ``engine`` instance to each tool.

    Every call returns a JSON envelope string rather than raising:
    ``{"ok": true, "result": {...}}`` on success, ``{"ok": false, "error":
    {"code", "type", "message"}}`` on failure, so the agent reads the engine
    error and can self-correct instead of crashing the run.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    package_dependencies: list[str] = Field(default_factory=lambda: ["chdb"])

    path: str = Field(
        default=":memory:",
        description=(
            "Directory for the chDB session's data; ':memory:' (default) uses "
            "an ephemeral in-memory session."
        ),
    )
    read_only: bool = Field(
        default=True,
        description=(
            "If True (default), the session is locked with the ClickHouse "
            "setting readonly=2: INSERT/CREATE/ALTER/DROP are rejected while "
            "SELECT and the file()/s3()/url() table functions keep working. "
            "Fixed at construction — chDB cannot lower readonly once set."
        ),
    )
    max_rows: int = Field(
        default=1000,
        description=(
            "Row cap per result; rows beyond it are dropped and the result "
            "carries a `truncated` flag the agent can react to."
        ),
    )
    max_bytes: int = Field(
        default=1_000_000,
        description="UTF-8 byte cap on each serialized result payload.",
    )
    max_execution_time: int | None = Field(
        default=None,
        description=(
            "Engine-side wall-clock limit per query, in seconds; a runaway "
            "query fails with a TIMEOUT_EXCEEDED error instead of hanging."
        ),
    )
    file_allowlist: list[str] | None = Field(
        default=None,
        description=(
            "When set, file-like sources (file/url/s3/...) may only read "
            "paths under these prefixes, and DSN-based sources (postgresql/"
            "mysql/remote/...) are refused outright."
        ),
    )
    attachments: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Local files to expose as named tables at construction, as "
            "{table_name: path} or {table_name: (path, format)}. This is how "
            "read-only tools get file-backed tables (attach_file needs "
            "read_only=False)."
        ),
    )
    engine: Any | None = Field(
        default=None,
        exclude=True,
        description=(
            "An existing chdb.agents.ChDBTool to reuse; the config fields "
            "above are ignored when set. Pass the same engine to several "
            "tools so they share tables and attachments."
        ),
    )

    _owned_engine: Any | None = PrivateAttr(default=None)
    _engine_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def _get_engine(self) -> Any:
        if self.engine is not None:
            return self.engine
        with self._engine_lock:
            if self._owned_engine is None:
                chdb_tool_class = _import_chdb_tool_class()
                self._owned_engine = chdb_tool_class(
                    self.path,
                    read_only=self.read_only,
                    max_rows=self.max_rows,
                    max_bytes=self.max_bytes,
                    max_execution_time=self.max_execution_time,
                    file_allowlist=self.file_allowlist,
                    attachments=self.attachments,
                )
            return self._owned_engine

    def _dispatch(self, tool_name: str, arguments: dict[str, Any]) -> str:
        envelope = self._get_engine().call(tool_name, arguments)
        return json.dumps(envelope)

    async def _adispatch(self, tool_name: str, arguments: dict[str, Any]) -> str:
        engine = self._get_engine()
        envelope = await engine.acall(tool_name, arguments)
        return json.dumps(envelope)

    def close(self) -> None:
        """Close the engine this tool created; an injected ``engine`` is left to its owner."""
        with self._engine_lock:
            engine, self._owned_engine = self._owned_engine, None
        if engine is not None:
            engine.close()


class ChDBRunSelectQuerySchema(BaseModel):
    """Input for run_select_query."""

    sql: str = Field(
        ...,
        description=(
            "A complete read-only ClickHouse SQL statement; use {name:Type} "
            "placeholders for values."
        ),
    )
    params: dict[str, Any] | None = Field(
        default=None,
        description="Values bound to the {name:Type} placeholders in `sql`.",
    )


class ChDBRunSelectQueryTool(ChDBBaseTool):
    """Run read-only ClickHouse SQL in-process with chDB."""

    name: str = "run_select_query"
    description: str = (
        "Run a read-only ClickHouse SQL query with chDB, an in-process "
        "ClickHouse engine (full SQL dialect: 1000+ functions, window "
        "functions, arrays, JSON). Pass values via `params` as {name:Type} "
        "placeholders (e.g. WHERE id = {id:Int64}); never concatenate values "
        "into the SQL. Read external data inline with table functions: "
        "file('path'[, 'format']), s3('url','format'), url(...), "
        "postgresql(...), mysql(...). Returns rows plus a `truncated` flag. "
        "First use list_tables / describe_table to learn the schema."
    )
    args_schema: type[BaseModel] = ChDBRunSelectQuerySchema

    def _run(self, sql: str, params: dict[str, Any] | None = None) -> str:
        return self._dispatch("run_select_query", {"sql": sql, "params": params})

    async def _arun(self, sql: str, params: dict[str, Any] | None = None) -> str:
        return await self._adispatch("run_select_query", {"sql": sql, "params": params})


class ChDBListDatabasesSchema(BaseModel):
    """Input for list_databases (no arguments)."""


class ChDBListDatabasesTool(ChDBBaseTool):
    """List the databases in the chDB session."""

    name: str = "list_databases"
    description: str = "List the databases available in the chDB session."
    args_schema: type[BaseModel] = ChDBListDatabasesSchema

    def _run(self) -> str:
        return self._dispatch("list_databases", {})

    async def _arun(self) -> str:
        return await self._adispatch("list_databases", {})


class ChDBListTablesSchema(BaseModel):
    """Input for list_tables."""

    database: str | None = Field(
        default=None,
        description="Database to list tables from; current database if omitted.",
    )


class ChDBListTablesTool(ChDBBaseTool):
    """List the tables in a database of the chDB session."""

    name: str = "list_tables"
    description: str = (
        "List the tables in a database (the current database if `database` is omitted)."
    )
    args_schema: type[BaseModel] = ChDBListTablesSchema

    def _run(self, database: str | None = None) -> str:
        return self._dispatch("list_tables", {"database": database})

    async def _arun(self, database: str | None = None) -> str:
        return await self._adispatch("list_tables", {"database": database})


class ChDBDescribeTableSchema(BaseModel):
    """Input for describe_table."""

    target: str = Field(
        ...,
        description=(
            'A table name or a table-function expression, e.g. "events" or '
            "\"s3('s3://b/f.parquet','Parquet')\"."
        ),
    )
    database: str | None = Field(
        default=None,
        description="Database qualifier for a table name (invalid for a table function).",
    )


class ChDBDescribeTableTool(ChDBBaseTool):
    """Describe the columns of a table or table-function expression."""

    name: str = "describe_table"
    description: str = (
        "Describe the columns and types of a table (optionally "
        "database-qualified) or a table-function expression, e.g. "
        "s3('https://bucket/f.parquet','Parquet') or file('data.csv')."
    )
    args_schema: type[BaseModel] = ChDBDescribeTableSchema

    def _run(self, target: str, database: str | None = None) -> str:
        return self._dispatch(
            "describe_table", {"target": target, "database": database}
        )

    async def _arun(self, target: str, database: str | None = None) -> str:
        return await self._adispatch(
            "describe_table", {"target": target, "database": database}
        )


class ChDBGetSampleDataSchema(BaseModel):
    """Input for get_sample_data."""

    target: str = Field(
        ...,
        description="A table name or a table-function expression.",
    )
    database: str | None = Field(
        default=None,
        description="Database qualifier for a table name (invalid for a table function).",
    )
    limit: int | None = Field(
        default=None,
        description="Number of sample rows (default 5).",
    )


class ChDBGetSampleDataTool(ChDBBaseTool):
    """Fetch a few sample rows from a table or table-function expression."""

    name: str = "get_sample_data"
    description: str = (
        "Return a few sample rows from a table or table-function expression, "
        "to see real values before querying."
    )
    args_schema: type[BaseModel] = ChDBGetSampleDataSchema

    def _run(
        self,
        target: str,
        database: str | None = None,
        limit: int | None = None,
    ) -> str:
        return self._dispatch(
            "get_sample_data",
            {"target": target, "database": database, "limit": limit},
        )

    async def _arun(
        self,
        target: str,
        database: str | None = None,
        limit: int | None = None,
    ) -> str:
        return await self._adispatch(
            "get_sample_data",
            {"target": target, "database": database, "limit": limit},
        )


class ChDBListFunctionsSchema(BaseModel):
    """Input for list_functions."""

    like: str | None = Field(
        default=None,
        description='ILIKE pattern to filter function names, e.g. "%array%".',
    )
    limit: int | None = Field(
        default=None,
        description="Max function names to return (default 200).",
    )


class ChDBListFunctionsTool(ChDBBaseTool):
    """List the ClickHouse SQL functions available in chDB."""

    name: str = "list_functions"
    description: str = (
        "List available ClickHouse SQL functions, optionally filtered by an "
        "ILIKE pattern."
    )
    args_schema: type[BaseModel] = ChDBListFunctionsSchema

    def _run(self, like: str | None = None, limit: int | None = None) -> str:
        return self._dispatch("list_functions", {"like": like, "limit": limit})

    async def _arun(self, like: str | None = None, limit: int | None = None) -> str:
        return await self._adispatch("list_functions", {"like": like, "limit": limit})


class ChDBAttachFileSchema(BaseModel):
    """Input for attach_file."""

    name: str = Field(
        ...,
        description="The table name to register the file under.",
    )
    path: str = Field(
        ...,
        description="Path to the local file.",
    )
    format: str | None = Field(
        default=None,
        description=(
            "chDB/ClickHouse input format (auto-detected from the extension if omitted)."
        ),
    )


class ChDBAttachFileTool(ChDBBaseTool):
    """Register a local file as a named queryable table in the chDB session."""

    name: str = "attach_file"
    description: str = (
        "Register a local file as a queryable named table (a view over "
        "file()). Writable tools only; on a read-only tool this returns a "
        "READONLY error — declare files via the tool's attachments option "
        "instead."
    )
    args_schema: type[BaseModel] = ChDBAttachFileSchema

    def _run(self, name: str, path: str, format: str | None = None) -> str:
        return self._dispatch(
            "attach_file", {"name": name, "path": path, "format": format}
        )

    async def _arun(self, name: str, path: str, format: str | None = None) -> str:
        return await self._adispatch(
            "attach_file", {"name": name, "path": path, "format": format}
        )


def chdb_tools(
    path: str = ":memory:",
    *,
    read_only: bool = True,
    max_rows: int = 1000,
    max_bytes: int = 1_000_000,
    max_execution_time: int | None = None,
    file_allowlist: list[str] | None = None,
    attachments: dict[str, Any] | None = None,
    engine: Any | None = None,
) -> list[BaseTool]:
    """Build the chDB tool suite over one shared in-process engine.

    All returned tools run against the same chDB session, so a table
    registered by attach_file (or declared via ``attachments``) is visible to
    run_select_query, describe_table, and the rest. attach_file is included
    only for writable suites (``read_only=False``); on a read-only engine it
    could only ever return a READONLY error, and files should be declared via
    ``attachments`` instead.

    Pass an existing ``chdb.agents.ChDBTool`` as ``engine`` to control the
    engine's lifecycle yourself (the other arguments are then ignored);
    otherwise one is created from the arguments and lives until the process
    exits or ``engine.close()`` is called on any tool's ``engine`` attribute.
    """
    if engine is None:
        chdb_tool_class = _import_chdb_tool_class()
        engine = chdb_tool_class(
            path,
            read_only=read_only,
            max_rows=max_rows,
            max_bytes=max_bytes,
            max_execution_time=max_execution_time,
            file_allowlist=file_allowlist,
            attachments=attachments,
        )
    writable = not getattr(engine, "read_only", True)

    tools: list[BaseTool] = [
        ChDBRunSelectQueryTool(engine=engine),
        ChDBListDatabasesTool(engine=engine),
        ChDBListTablesTool(engine=engine),
        ChDBDescribeTableTool(engine=engine),
        ChDBGetSampleDataTool(engine=engine),
        ChDBListFunctionsTool(engine=engine),
    ]
    if writable:
        tools.append(ChDBAttachFileTool(engine=engine))

    return tools
