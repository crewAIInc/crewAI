from collections.abc import Iterator
import logging
import os
import re
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


# Subset of write commands that can realistically appear *inside* a CTE body.
# Narrower than _WRITE_COMMANDS to avoid false positives on identifiers like
# ``comment``, ``set``, or ``reset`` which are common column/table names.
_CTE_WRITE_INDICATORS = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "MERGE",
}


_AS_PAREN_RE = re.compile(r"\bAS\s*\(", re.IGNORECASE)


def _iter_as_paren_matches(stmt: str) -> Iterator[re.Match[str]]:
    """Yield regex matches for ``AS\\s*(`` outside of string literals."""
    # Build a set of character positions that are inside string literals.
    in_string: set[int] = set()
    i = 0
    while i < len(stmt):
        if stmt[i] == "'":
            start = i
            end = _skip_string_literal(stmt, i)
            in_string.update(range(start, end))
            i = end
        else:
            i += 1

    for m in _AS_PAREN_RE.finditer(stmt):
        if m.start() not in in_string:
            yield m


def _detect_writable_cte(stmt: str) -> str | None:
    """Return the first write command inside a CTE body, or None.

    Instead of tokenizing the whole statement (which falsely matches column
    names like ``comment``), this walks through parenthesized CTE bodies and
    checks only the *first keyword after* an opening ``AS (`` for a write
    command.  Uses a regex to handle any whitespace (spaces, tabs, newlines)
    between ``AS`` and ``(``.  Skips matches inside string literals.
    """
    for m in _iter_as_paren_matches(stmt):
        body = stmt[m.end() :].lstrip()
        first_word = body.split()[0].upper().strip("()") if body.split() else ""
        if first_word in _CTE_WRITE_INDICATORS:
            return first_word
    return None


def _skip_string_literal(stmt: str, pos: int) -> int:
    """Skip past a string literal starting at pos (single-quoted).

    Handles escaped quotes ('') inside the literal.
    Returns the index after the closing quote.
    """
    quote_char = stmt[pos]
    i = pos + 1
    while i < len(stmt):
        if stmt[i] == quote_char:
            # Check for escaped quote ('')
            if i + 1 < len(stmt) and stmt[i + 1] == quote_char:
                i += 2
                continue
            return i + 1
        i += 1
    return i  # Unterminated literal — return end


def _find_matching_close_paren(stmt: str, start: int) -> int:
    """Find the matching close paren, skipping string literals."""
    depth = 1
    i = start
    while i < len(stmt) and depth > 0:
        ch = stmt[i]
        if ch == "'":
            i = _skip_string_literal(stmt, i)
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        i += 1
    return i


def _extract_main_query_after_cte(stmt: str) -> str | None:
    """Extract the main (outer) query that follows all CTE definitions.

    For ``WITH cte AS (SELECT 1) DELETE FROM users``, returns ``DELETE FROM users``.
    Returns None if no main query is found after the last CTE body.
    Handles parentheses inside string literals (e.g., ``SELECT '(' FROM t``).
    """
    last_cte_end = 0
    for m in _iter_as_paren_matches(stmt):
        last_cte_end = _find_matching_close_paren(stmt, m.end())

    if last_cte_end > 0:
        remainder = stmt[last_cte_end:].strip().lstrip(",").strip()
        if remainder:
            return remainder
    return None


def _resolve_explain_command(stmt: str) -> str | None:
    """Resolve the underlying command from an EXPLAIN [ANALYZE] [VERBOSE] statement.

    Returns the real command (e.g., 'DELETE') if ANALYZE is present, else None.
    Handles both space-separated and parenthesized syntax.
    """
    rest = stmt.strip()[len("EXPLAIN") :].strip()
    if not rest:
        return None

    analyze_found = False
    explain_opts = {"ANALYZE", "ANALYSE", "VERBOSE"}

    if rest.startswith("("):
        close = rest.find(")")
        if close != -1:
            options_str = rest[1:close].upper()
            analyze_found = any(
                opt.strip() in ("ANALYZE", "ANALYSE") for opt in options_str.split(",")
            )
            rest = rest[close + 1 :].strip()
    else:
        while rest:
            first_opt = rest.split()[0].upper().rstrip(";") if rest.split() else ""
            if first_opt in ("ANALYZE", "ANALYSE"):
                analyze_found = True
            if first_opt not in explain_opts:
                break
            rest = rest[len(first_opt) :].strip()

    if analyze_found and rest:
        return rest.split()[0].upper().rstrip(";")
    return None


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
        # EXPLAIN ANALYZE actually executes the underlying query — resolve the
        # real command so write operations are caught.
        if command == "EXPLAIN":
            resolved = _resolve_explain_command(stmt)
            if resolved:
                command = resolved

        # WITH starts a CTE.  Read-only CTEs are fine; writable CTEs
        # (e.g. WITH d AS (DELETE …) SELECT …) must be blocked in read-only mode.
        if command == "WITH":
            # Check for write commands inside CTE bodies.
            write_found = _detect_writable_cte(stmt)
            if write_found:
                found = write_found
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
                return

            # Check the main query after the CTE definitions.
            main_query = _extract_main_query_after_cte(stmt)
            if main_query:
                main_cmd = main_query.split()[0].upper().rstrip(";")
                if main_cmd in _WRITE_COMMANDS:
                    if not self.allow_dml:
                        raise ValueError(
                            f"NL2SQLTool is configured in read-only mode and blocked a "
                            f"'{main_cmd}' statement after a CTE. To allow write "
                            f"operations set allow_dml=True or "
                            f"CREWAI_NL2SQL_ALLOW_DML=true."
                        )
                    logger.warning(
                        "NL2SQLTool: executing '%s' after CTE because allow_dml=True.",
                        main_cmd,
                    )
                elif main_cmd not in _READ_ONLY_COMMANDS:
                    if not self.allow_dml:
                        raise ValueError(
                            f"NL2SQLTool blocked an unrecognised SQL command '{main_cmd}' "
                            f"after a CTE. Only {sorted(_READ_ONLY_COMMANDS)} are allowed "
                            f"in read-only mode."
                        )
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

        def _is_write_stmt(s: str) -> bool:
            cmd = self._extract_command(s)
            if cmd in _WRITE_COMMANDS:
                return True
            if cmd == "EXPLAIN":
                # Resolve the underlying command for EXPLAIN ANALYZE
                resolved = _resolve_explain_command(s)
                if resolved and resolved in _WRITE_COMMANDS:
                    return True
            if cmd == "WITH":
                if _detect_writable_cte(s):
                    return True
                main_q = _extract_main_query_after_cte(s)
                if main_q:
                    return main_q.split()[0].upper().rstrip(";") in _WRITE_COMMANDS
            return False

        is_write = any(_is_write_stmt(s) for s in _stmts)

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
