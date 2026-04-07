import logging
import os
from typing import Any


try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, model_validator


try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Commands allowed in read-only mode
# NOTE: WITH is intentionally excluded — writable CTEs start with WITH, so the
# CTE body must be inspected separately (see _validate_statement).
_READ_ONLY_COMMANDS = {"SELECT", "SHOW", "DESCRIBE", "DESC", "EXPLAIN"}

# Commands that mutate state and are blocked by default
_WRITE_COMMANDS = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "GRANT",
    "REVOKE",
    "EXEC",
    "EXECUTE",
    "CALL",
    "MERGE",
    "REPLACE",
    "UPSERT",
    "LOAD",
    "COPY",
    "VACUUM",
    "ANALYZE",
    "ANALYSE",
    "REINDEX",
    "CLUSTER",
    "REFRESH",
    "COMMENT",
    "SET",
    "RESET",
}


class NL2SQLToolInput(BaseModel):
    sql_query: str = Field(
        title="SQL Query",
        description="The SQL query to execute.",
    )


class NL2SQLTool(BaseTool):
    """Tool that converts natural language to SQL and executes it against a database.

    By default the tool operates in **read-only mode**: only SELECT, SHOW,
    DESCRIBE, EXPLAIN, and read-only CTEs (WITH … SELECT) are permitted.  Write
    operations (INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, …) are
    blocked unless ``allow_dml=True`` is set explicitly or the environment
    variable ``CREWAI_NL2SQL_ALLOW_DML=true`` is present.

    Writable CTEs (``WITH d AS (DELETE …) SELECT …``) and
    ``EXPLAIN ANALYZE <write-stmt>`` are treated as write operations and are
    blocked in read-only mode.

    The ``_fetch_all_available_columns`` helper uses parameterised queries so
    that table names coming from the database catalogue cannot be used as an
    injection vector.
    """

    name: str = "NL2SQLTool"
    description: str = (
        "Converts natural language to SQL queries and executes them against a "
        "database. Read-only by default — only SELECT/SHOW/DESCRIBE/EXPLAIN "
        "queries (and read-only CTEs) are allowed unless configured with "
        "allow_dml=True."
    )
    db_uri: str = Field(
        title="Database URI",
        description="The URI of the database to connect to.",
    )
    allow_dml: bool = Field(
        default=False,
        title="Allow DML",
        description=(
            "When False (default) only read statements are permitted. "
            "Set to True to allow INSERT/UPDATE/DELETE/DROP and other "
            "write operations."
        ),
    )
    tables: list[dict[str, Any]] = Field(default_factory=list)
    columns: dict[str, list[dict[str, Any]] | str] = Field(default_factory=dict)
    args_schema: type[BaseModel] = NL2SQLToolInput

    @model_validator(mode="after")
    def _apply_env_override(self) -> Self:
        """Allow CREWAI_NL2SQL_ALLOW_DML=true to override allow_dml at runtime."""
        if os.environ.get("CREWAI_NL2SQL_ALLOW_DML", "").strip().lower() == "true":
            if not self.allow_dml:
                logger.warning(
                    "NL2SQLTool: CREWAI_NL2SQL_ALLOW_DML env var is set — "
                    "DML/DDL operations are enabled. Ensure this is intentional."
                )
            self.allow_dml = True
        return self

    def model_post_init(self, __context: Any) -> None:
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError(
                "sqlalchemy is not installed. Please install it with "
                "`pip install crewai-tools[sqlalchemy]`"
            )

        if self.allow_dml:
            logger.warning(
                "NL2SQLTool: allow_dml=True — write operations (INSERT/UPDATE/"
                "DELETE/DROP/…) are permitted. Use with caution."
            )

        data: dict[str, list[dict[str, Any]] | str] = {}
        result = self._fetch_available_tables()
        if isinstance(result, str):
            raise RuntimeError(f"Failed to fetch tables: {result}")
        tables: list[dict[str, Any]] = result

        for table in tables:
            table_columns = self._fetch_all_available_columns(table["table_name"])
            data[f"{table['table_name']}_columns"] = table_columns

        self.tables = tables
        self.columns = data

    # ------------------------------------------------------------------
    # Query validation
    # ------------------------------------------------------------------

    def _validate_query(self, sql_query: str) -> None:
        """Raise ValueError if *sql_query* is not permitted under the current config.

        Splits the query on semicolons and validates each statement
        independently.  When ``allow_dml=False`` (the default), multi-statement
        queries are rejected outright to prevent ``SELECT 1; DROP TABLE users``
        style bypasses.  When ``allow_dml=True`` every statement is checked and
        a warning is emitted for write operations.
        """
        statements = [s.strip() for s in sql_query.split(";") if s.strip()]

        if not statements:
            raise ValueError("NL2SQLTool received an empty SQL query.")

        if not self.allow_dml and len(statements) > 1:
            raise ValueError(
                "NL2SQLTool blocked a multi-statement query in read-only mode. "
                "Semicolons are not permitted when allow_dml=False."
            )

        for stmt in statements:
            self._validate_statement(stmt)

    def _validate_statement(self, stmt: str) -> None:
        """Validate a single SQL statement (no semicolons)."""
        command = self._extract_command(stmt)

        # EXPLAIN ANALYZE / EXPLAIN ANALYSE actually *executes* the underlying
        # query.  Resolve the real command so write operations are caught.
        # Handles both space-separated ("EXPLAIN ANALYZE DELETE …") and
        # parenthesized ("EXPLAIN (ANALYZE) DELETE …", "EXPLAIN (ANALYZE, VERBOSE) DELETE …").
        if command == "EXPLAIN":
            rest = stmt.strip()[len("EXPLAIN"):].strip()
            analyze_found = False

            if rest.startswith("("):
                # Parenthesized options: EXPLAIN (ANALYZE, VERBOSE, …) <stmt>
                close = rest.find(")")
                if close != -1:
                    options_str = rest[1:close].upper()
                    analyze_found = any(
                        opt.strip() in ("ANALYZE", "ANALYSE")
                        for opt in options_str.split(",")
                    )
                    rest = rest[close + 1:].strip()
            else:
                # Space-separated: EXPLAIN ANALYZE <stmt>
                first_opt = rest.split()[0].upper().rstrip(";") if rest.split() else ""
                if first_opt in ("ANALYZE", "ANALYSE"):
                    analyze_found = True
                    rest = rest[len(first_opt):].strip()

            if analyze_found and rest:
                command = rest.split()[0].upper().rstrip(";")

        # WITH starts a CTE.  Read-only CTEs are fine; writable CTEs
        # (e.g. WITH d AS (DELETE …) SELECT …) must be blocked in read-only mode.
        if command == "WITH":
            tokens_upper = {t.upper().strip("();,") for t in stmt.split()}
            write_found = tokens_upper & _WRITE_COMMANDS
            if write_found:
                found = next(iter(write_found))
                if not self.allow_dml:
                    raise ValueError(
                        f"NL2SQLTool is configured in read-only mode and blocked a "
                        f"writable CTE containing a '{found}' statement. To allow "
                        f"write operations set allow_dml=True or "
                        f"CREWAI_NL2SQL_ALLOW_DML=true."
                    )
                logger.warning(
                    "NL2SQLTool: executing writable CTE with '%s' because allow_dml=True.",
                    found,
                )
            # Both read-only and writable-but-permitted CTEs need no further checks.
            return

        if command in _WRITE_COMMANDS:
            if not self.allow_dml:
                raise ValueError(
                    f"NL2SQLTool is configured in read-only mode and blocked a "
                    f"'{command}' statement. To allow write operations set "
                    f"allow_dml=True or CREWAI_NL2SQL_ALLOW_DML=true."
                )
            logger.warning(
                "NL2SQLTool: executing write statement '%s' because allow_dml=True.",
                command,
            )
        elif command not in _READ_ONLY_COMMANDS:
            # Unknown command — block by default unless DML is explicitly enabled
            if not self.allow_dml:
                raise ValueError(
                    f"NL2SQLTool blocked an unrecognised SQL command '{command}'. "
                    f"Only {sorted(_READ_ONLY_COMMANDS)} are allowed in read-only "
                    f"mode."
                )

    @staticmethod
    def _extract_command(sql_query: str) -> str:
        """Return the uppercased first keyword of *sql_query*."""
        stripped = sql_query.strip().lstrip("(")
        first_token = stripped.split()[0] if stripped.split() else ""
        return first_token.upper().rstrip(";")

    # ------------------------------------------------------------------
    # Schema introspection helpers
    # ------------------------------------------------------------------

    def _fetch_available_tables(self) -> list[dict[str, Any]] | str:
        return self.execute_sql(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public';"
        )

    def _fetch_all_available_columns(
        self, table_name: str
    ) -> list[dict[str, Any]] | str:
        """Fetch columns for *table_name* using a parameterised query.

        The table name is bound via SQLAlchemy's ``:param`` syntax to prevent
        SQL injection from catalogue values.
        """
        return self.execute_sql(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = :table_name",
            params={"table_name": table_name},
        )

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def _run(self, sql_query: str) -> list[dict[str, Any]] | str:
        try:
            self._validate_query(sql_query)
            data = self.execute_sql(sql_query)
        except ValueError:
            raise
        except Exception as exc:
            data = (
                f"Based on these tables {self.tables} and columns {self.columns}, "
                "you can create SQL queries to retrieve data from the database. "
                f"Get the original request {sql_query} and the error {exc} and "
                "create the correct SQL query."
            )

        return data

    def execute_sql(
        self,
        sql_query: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]] | str:
        """Execute *sql_query* and return the results as a list of dicts.

        Parameters
        ----------
        sql_query:
            The SQL statement to run.
        params:
            Optional mapping of bind parameters (e.g. ``{"table_name": "users"}``).
        """
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError(
                "sqlalchemy is not installed. Please install it with "
                "`pip install crewai-tools[sqlalchemy]`"
            )

        # Check ALL statements so that e.g. "SELECT 1; DROP TABLE t" triggers a
        # commit when allow_dml=True, regardless of statement order.
        _stmts = [s.strip() for s in sql_query.split(";") if s.strip()]
        is_write = any(self._extract_command(s) in _WRITE_COMMANDS for s in _stmts)

        engine = create_engine(self.db_uri)
        Session = sessionmaker(bind=engine)  # noqa: N806
        session = Session()
        try:
            result = session.execute(text(sql_query), params or {})

            # Only commit when the operation actually mutates state
            if self.allow_dml and is_write:
                session.commit()

            if result.returns_rows:  # type: ignore[attr-defined]
                columns = result.keys()
                return [
                    dict(zip(columns, row, strict=False)) for row in result.fetchall()
                ]
            return f"Query {sql_query} executed successfully"

        except Exception as e:
            session.rollback()
            raise e

        finally:
            session.close()
