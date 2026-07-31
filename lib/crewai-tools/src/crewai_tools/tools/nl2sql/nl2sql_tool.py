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


# Keywords that may legitimately open a CTE body in read-only mode. This is an
# allowlist rather than a write-command denylist: anything unrecognised at the
# head of a CTE body is treated as a write and blocked, so a dialect keyword we
# have not enumerated cannot slip through (see _validate_cte_statement).
_CTE_READ_ONLY_LEADS = _READ_ONLY_COMMANDS | {
    "VALUES",
    "TABLE",
    "WITH",
    "SEARCH",
    "CYCLE",
}


# ``AS (`` optionally preceded by PostgreSQL's [NOT] MATERIALIZED modifier.
# Without the modifier branch, ``WITH d AS MATERIALIZED (DELETE …)`` parses as
# having no CTE body at all and skips validation entirely.
_AS_PAREN_RE = re.compile(
    r"\bAS\s+(?:NOT\s+)?MATERIALIZED\s*\(|\bAS\s*\(", re.IGNORECASE
)

# MySQL runs the body of a version-gated comment (``/*!40001 … */``) as real
# SQL, so those must not be masked away as inert comment text.
_MYSQL_EXEC_COMMENT_PREFIX = "/*!"

# PostgreSQL dollar-quoted string delimiters: $$ … $$ or $tag$ … $tag$.
_DOLLAR_QUOTE_RE = re.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*)?\$")

# Server-side file writes reachable from a plain SELECT, which no amount of
# transaction-level read-only enforcement prevents.
_FILE_SINK_RE = re.compile(r"\bINTO\s+(?:OUTFILE|DUMPFILE)\b", re.IGNORECASE)

# Functions that read or write the database server's filesystem, or open a new
# connection that escapes the current (read-only) transaction. Callable from a
# SELECT, so the first-keyword check never sees them.
_SERVER_FILE_FUNC_RE = re.compile(
    r"\b(?:pg_read_file|pg_read_binary_file|pg_ls_dir|pg_stat_file|pg_logdir_ls"
    r"|lo_import|lo_export|load_file|dblink|dblink_exec|dblink_send_query)\s*\(",
    re.IGNORECASE,
)


def _skip_quoted(stmt: str, pos: int) -> int:
    """Skip past the quoted run starting at *pos*.

    Handles single-quoted literals, double-quoted identifiers (or strings under
    ANSI_QUOTES) and MySQL backtick identifiers, including the doubled-delimiter
    escape (``''``). Returns the index just past the closing delimiter, or the
    end of the string when the run is unterminated.
    """
    quote_char = stmt[pos]
    i = pos + 1
    while i < len(stmt):
        if stmt[i] == quote_char:
            if i + 1 < len(stmt) and stmt[i + 1] == quote_char:
                i += 2
                continue
            return i + 1
        i += 1
    return i  # Unterminated literal — return end


# Kept as an alias because the single-quote case is the one callers reason about.
_skip_string_literal = _skip_quoted


def _mask_inert_spans(stmt: str) -> str:
    """Blank out quoted runs and comments, preserving every character offset.

    Analysis runs over the mask so that keywords hidden inside strings are not
    matched, and keywords hidden *behind* comments are. Offsets are preserved so
    a match found in the mask can be sliced out of the original statement.

    MySQL executable comments (``/*! … */``) are deliberately left visible: the
    server executes their contents, so the validator must see them too.

    Args:
        stmt: The SQL statement to mask.

    Returns:
        A same-length copy of *stmt* with inert spans replaced by spaces.
    """
    out = list(stmt)
    n = len(stmt)
    i = 0

    def blank(start: int, end: int) -> None:
        for k in range(start, min(end, n)):
            if out[k] != "\n":  # keep line structure for "--" comment scanning
                out[k] = " "

    while i < n:
        ch = stmt[i]
        if stmt.startswith("--", i):
            end = stmt.find("\n", i)
            end = n if end == -1 else end
            blank(i, end)
            i = end
        elif stmt.startswith("/*", i) and not stmt.startswith(
            _MYSQL_EXEC_COMMENT_PREFIX, i
        ):
            depth = 1
            j = i + 2
            while j < n and depth > 0:
                if stmt.startswith("/*", j):
                    depth += 1
                    j += 2
                elif stmt.startswith("*/", j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            blank(i, j)
            i = j
        elif ch in ("'", '"', "`"):
            end = _skip_quoted(stmt, i)
            blank(i, end)
            i = end
        elif ch == "$" and (m := _DOLLAR_QUOTE_RE.match(stmt, i)):
            tag = m.group(0)
            close = stmt.find(tag, m.end())
            end = n if close == -1 else close + len(tag)
            blank(i, end)
            i = end
        else:
            i += 1

    return "".join(out)


def _split_statements(sql_query: str) -> list[str]:
    """Split *sql_query* on semicolons that are not inside a string or comment.

    A naive ``str.split(";")`` both rejects legitimate queries containing a
    semicolon in a literal and miscounts statements when one is hidden in a
    comment.
    """
    masked = _mask_inert_spans(sql_query)
    statements: list[str] = []
    start = 0
    for i, ch in enumerate(masked):
        if ch == ";":
            chunk = sql_query[start:i].strip()
            if chunk:
                statements.append(chunk)
            start = i + 1
    tail = sql_query[start:].strip()
    if tail:
        statements.append(tail)
    return statements


def _iter_as_paren_matches(masked: str) -> Iterator[re.Match[str]]:
    """Yield ``AS (`` matches over an already-masked statement."""
    return _AS_PAREN_RE.finditer(masked)


def _first_keyword(text_: str) -> str:
    """Return the leading SQL keyword of *text_*, uppercased."""
    tokens = text_.split()
    if not tokens:
        return ""
    return tokens[0].upper().strip("()").rstrip(";")


def _iter_cte_bodies(masked: str) -> Iterator[str]:
    """Yield the leading keyword of each top-level CTE body in *masked*.

    Matches nested inside a body already consumed are skipped, so a subquery
    that itself contains ``AS (`` does not shift the outer parse.
    """
    consumed_until = 0
    for m in _iter_as_paren_matches(masked):
        if m.start() < consumed_until:
            continue
        consumed_until = _find_matching_close_paren(masked, m.end())
        yield _first_keyword(masked[m.end() :])


def _detect_writable_cte(stmt: str) -> str | None:
    """Return the first non-read-only keyword opening a CTE body, or None.

    Kept for backwards compatibility with callers that only need a yes/no
    answer; :func:`_validate_cte_statement` is the enforcing path.
    """
    masked = _mask_inert_spans(stmt)
    for lead in _iter_cte_bodies(masked):
        if lead and lead not in _CTE_READ_ONLY_LEADS:
            return lead
    return None


def _find_matching_close_paren(masked: str, start: int) -> int:
    """Find the matching close paren in an already-masked statement."""
    depth = 1
    i = start
    while i < len(masked) and depth > 0:
        ch = masked[i]
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
    """
    masked = _mask_inert_spans(stmt)
    return _extract_main_query_from_masked(masked)


def _extract_main_query_from_masked(masked: str) -> str | None:
    """Same as :func:`_extract_main_query_after_cte` for pre-masked input."""
    last_cte_end = 0
    for m in _iter_as_paren_matches(masked):
        if m.start() < last_cte_end:
            continue
        last_cte_end = _find_matching_close_paren(masked, m.end())

    if last_cte_end > 0:
        remainder = masked[last_cte_end:].strip().lstrip(",").strip()
        if remainder:
            return remainder
    return None


def _resolve_explain_command(stmt: str) -> str | None:
    """Resolve the underlying command from an EXPLAIN [ANALYZE] [VERBOSE] statement.

    Returns the real command (e.g., 'DELETE') if ANALYZE is present, else None.
    Handles both space-separated and parenthesized syntax. Comments are masked
    first, so ``EXPLAIN /*x*/ ANALYZE DELETE …`` resolves to ``DELETE`` rather
    than stalling on the comment token.
    """
    masked = _mask_inert_spans(stmt).strip()
    if not masked.upper().startswith("EXPLAIN"):
        return None
    rest = masked[len("EXPLAIN") :].strip()
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
        # Consume option tokens one at a time. Slicing by token *length* would
        # desynchronise whenever the raw token differs from its normalised form.
        tokens = rest.split()
        consumed = 0
        for token in tokens:
            normalised = token.upper().rstrip(";")
            if normalised in ("ANALYZE", "ANALYSE"):
                analyze_found = True
            if normalised not in explain_opts:
                break
            consumed += 1
        rest = " ".join(tokens[consumed:])

    if analyze_found and rest:
        return _first_keyword(rest)
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

    Writable CTEs (``WITH d AS (DELETE …) SELECT …``, including the
    ``AS [NOT] MATERIALIZED`` spelling) and ``EXPLAIN ANALYZE <write-stmt>`` are
    treated as write operations and are blocked in read-only mode. Statements
    are analysed with strings and comments masked out, so neither a keyword
    hidden in a literal nor a comment inserted between keywords changes the
    verdict, and a ``WITH`` statement that cannot be parsed is rejected rather
    than allowed.

    In read-only mode the transaction is additionally marked
    ``SET TRANSACTION READ ONLY`` where the backend supports it, so enforcement
    does not rest on statement parsing alone.

    .. warning::
       Keyword validation cannot fully express "read-only": a SELECT can still
       reach the database server's filesystem (``INTO OUTFILE``,
       ``pg_read_file``) or invoke a side-effecting function. The known sinks
       are blocked explicitly, but the only complete control is to point
       ``db_uri`` at a **least-privileged, read-only database role**. Treat the
       checks in this class as defence in depth, not as a substitute.

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

    # Query validation

    def _validate_query(self, sql_query: str) -> None:
        """Raise ValueError if *sql_query* is not permitted under the current config.

        Splits the query on semicolons and validates each statement
        independently.  When ``allow_dml=False`` (the default), multi-statement
        queries are rejected outright to prevent ``SELECT 1; DROP TABLE users``
        style bypasses.  When ``allow_dml=True`` every statement is checked and
        a warning is emitted for write operations.
        """
        statements = _split_statements(sql_query)

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
        masked = _mask_inert_spans(stmt)
        command = self._extract_command(stmt)

        # Some writes are reachable from a statement whose first keyword is
        # SELECT, so they are invisible to the command check below and survive a
        # transaction-level rollback. Check them before anything else.
        self._reject_select_level_side_effects(masked)

        # EXPLAIN ANALYZE / EXPLAIN ANALYSE actually *executes* the underlying
        # query, in both the space-separated and parenthesized spellings
        # ("EXPLAIN (ANALYZE) DELETE …"). Resolve the real command so write
        # operations are caught.
        if command == "EXPLAIN":
            resolved = _resolve_explain_command(stmt)
            if resolved:
                command = resolved

        # (e.g. WITH d AS (DELETE …) SELECT …) must be blocked in read-only mode.
        if command == "WITH":
            self._validate_cte_statement(masked)
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

    def _reject_select_level_side_effects(self, masked: str) -> None:
        """Block writes and server-file access that a SELECT can reach.

        ``SELECT … INTO OUTFILE`` writes a file on the database server, and
        functions like ``pg_read_file`` or ``dblink_exec`` read the server's
        filesystem or open a connection outside the current transaction. None of
        these are undone by a rollback, and all of them present as a read-only
        first keyword, so they need an explicit check.

        These checks are a backstop, not the primary control: only a
        least-privileged database role can properly bound what the tool reaches.

        Args:
            masked: The statement with strings and comments already masked.

        Raises:
            ValueError: If a file sink or server-file function is present and
                ``allow_dml`` is False.
        """
        if self.allow_dml:
            return

        if _FILE_SINK_RE.search(masked):
            raise ValueError(
                "NL2SQLTool is configured in read-only mode and blocked a query "
                "writing to the database server's filesystem (INTO OUTFILE / "
                "INTO DUMPFILE). Grant the tool a read-only database role rather "
                "than enabling allow_dml."
            )

        if match := _SERVER_FILE_FUNC_RE.search(masked):
            raise ValueError(
                f"NL2SQLTool is configured in read-only mode and blocked a call to "
                f"'{match.group(0).rstrip('( ')}', which reaches the database "
                f"server's filesystem or opens a connection outside the current "
                f"transaction. Grant the tool a read-only database role rather "
                f"than enabling allow_dml."
            )

    def _validate_cte_statement(self, masked: str) -> None:
        """Validate a statement whose first keyword is ``WITH``.

        Fails closed: a ``WITH`` statement whose CTE bodies cannot be located, or
        which has no query after them, is rejected in read-only mode rather than
        passed through unchecked.

        Args:
            masked: The statement with strings and comments already masked.

        Raises:
            ValueError: If the statement writes, or cannot be parsed, while
                ``allow_dml`` is False.
        """
        leads = list(_iter_cte_bodies(masked))

        if not leads:
            if not self.allow_dml:
                raise ValueError(
                    "NL2SQLTool blocked a WITH statement whose CTE definitions "
                    "could not be parsed, so it cannot be confirmed read-only. "
                    "To allow write operations set allow_dml=True or "
                    "CREWAI_NL2SQL_ALLOW_DML=true."
                )
            return

        for lead in leads:
            if lead and lead not in _CTE_READ_ONLY_LEADS:
                if not self.allow_dml:
                    raise ValueError(
                        f"NL2SQLTool is configured in read-only mode and blocked a "
                        f"writable CTE containing a '{lead}' statement. To allow "
                        f"write operations set allow_dml=True or "
                        f"CREWAI_NL2SQL_ALLOW_DML=true."
                    )
                logger.warning(
                    "NL2SQLTool: executing writable CTE with '%s' because allow_dml=True.",
                    lead,
                )
                return

        main_query = _extract_main_query_from_masked(masked)
        if main_query is None:
            if not self.allow_dml:
                raise ValueError(
                    "NL2SQLTool blocked a WITH statement with no query after its "
                    "CTE definitions, so it cannot be confirmed read-only. To "
                    "allow write operations set allow_dml=True or "
                    "CREWAI_NL2SQL_ALLOW_DML=true."
                )
            return

        main_cmd = _first_keyword(main_query)
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
        elif main_cmd not in _READ_ONLY_COMMANDS and not self.allow_dml:
            raise ValueError(
                f"NL2SQLTool blocked an unrecognised SQL command '{main_cmd}' "
                f"after a CTE. Only {sorted(_READ_ONLY_COMMANDS)} are allowed "
                f"in read-only mode."
            )

    @staticmethod
    def _extract_command(sql_query: str) -> str:
        """Return the uppercased first keyword of *sql_query*."""
        return _first_keyword(_mask_inert_spans(sql_query).strip())

    # Schema introspection helpers

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

    # Core execution

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
        _stmts = _split_statements(sql_query)

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
                    return _first_keyword(main_q) in _WRITE_COMMANDS
            return False

        is_write = any(_is_write_stmt(s) for s in _stmts)

        engine = create_engine(self.db_uri)
        Session = sessionmaker(bind=engine)  # noqa: N806
        session = Session()
        try:
            if not self.allow_dml:
                self._enforce_read_only_transaction(session)

            result = session.execute(text(sql_query), params or {})

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

    @staticmethod
    def _enforce_read_only_transaction(session: Any) -> None:
        """Ask the backend to enforce read-only for this transaction.

        Statement inspection alone cannot guarantee a query is read-only — SQL
        is dialect-specific and the parser here is deliberately simple. Marking
        the transaction read-only moves enforcement into the database, where
        PostgreSQL and MySQL reject writes outright regardless of how the
        statement was spelled.

        Backends without the syntax (SQLite, SQL Server, Snowflake) raise, in
        which case the transaction is rolled back to clear the error state and
        keyword validation remains the only control. That is logged rather than
        raised so those backends keep working.

        Args:
            session: The active SQLAlchemy session.
        """
        try:
            session.execute(text("SET TRANSACTION READ ONLY"))
        except Exception as exc:
            session.rollback()
            logger.debug(
                "NL2SQLTool: backend rejected 'SET TRANSACTION READ ONLY' (%s); "
                "falling back to statement validation only. A read-only "
                "database role is strongly recommended.",
                exc,
            )
