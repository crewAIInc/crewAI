"""Shared utilities for the Azure CosmosDB NoSQL tools.

Contains the safety primitives (SQL identifier validation, value quoting,
distance-aware score thresholding) and small algorithms (Max Marginal
Relevance) that the three tools and the storage backend share. Lifted from
the equivalent helpers in ``langchain-azure``'s
``libs/azure-cosmosdb/src/langchain_azure_cosmosdb/_utils.py`` and
``_vectorstore.py`` (BSD-3-Clause), simplified for crewAI's needs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
import enum
import functools
import re
import time
from typing import Any, Protocol, TypeVar


# ---------------------------------------------------------------------------
# Distance strategies
# ---------------------------------------------------------------------------


class DistanceStrategy(str, enum.Enum):
    """Vector distance strategies supported by Cosmos NoSQL."""

    COSINE = "cosine"
    DOT_PRODUCT = "dotproduct"
    EUCLIDEAN = "euclidean"

    @classmethod
    def from_str(cls, value: str | None) -> DistanceStrategy:
        if not value:
            return cls.COSINE
        norm = value.replace("_", "").replace("-", "").lower()
        for member in cls:
            if member.value.replace("_", "").lower() == norm:
                return member
        return cls.COSINE


def score_threshold_passes(
    score: float,
    threshold: float | None,
    distance_function: str | DistanceStrategy = DistanceStrategy.COSINE,
) -> bool:
    """Return True if ``score`` satisfies ``threshold`` for the distance fn.

    Cosmos NoSQL ``VectorDistance`` returns:

    * For ``cosine`` / ``dotproduct``: a similarity-like score where higher is
      better (Cosmos rescales these so larger == more similar).
    * For ``euclidean``: an actual distance where lower == more similar.

    A ``threshold`` of ``None`` always passes.
    """
    if threshold is None:
        return True
    strategy = (
        distance_function
        if isinstance(distance_function, DistanceStrategy)
        else DistanceStrategy.from_str(distance_function)
    )
    if strategy is DistanceStrategy.EUCLIDEAN:
        return score <= threshold
    return score >= threshold


# ---------------------------------------------------------------------------
# SQL identifier / literal safety
# ---------------------------------------------------------------------------


# Cosmos NoSQL SQL reserved words (subset that matters for identifier safety).
# Mirrors the list maintained by the langchain-azure project; kept here so we
# do not introduce a runtime dep on langchain.
_SQL_RESERVED_WORDS: frozenset[str] = frozenset(
    {
        "select",
        "from",
        "where",
        "and",
        "or",
        "not",
        "in",
        "like",
        "between",
        "is",
        "null",
        "true",
        "false",
        "order",
        "by",
        "group",
        "having",
        "join",
        "inner",
        "outer",
        "left",
        "right",
        "on",
        "as",
        "case",
        "when",
        "then",
        "else",
        "end",
        "exists",
        "value",
        "distinct",
        "top",
        "offset",
        "limit",
        "asc",
        "desc",
        "rank",
    }
)

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_sql_identifier(value: str, *, name: str = "identifier") -> str:
    """Validate that ``value`` is safe to splice into a Cosmos SQL query.

    Rejects empty strings, reserved keywords and anything containing characters
    other than ``[A-Za-z0-9_]``. Returns the value unchanged on success.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    if not _IDENT_RE.match(value):
        raise ValueError(
            f"{name} {value!r} is not a valid SQL identifier "
            "(must start with a letter or underscore and contain only "
            "letters, digits or underscores)"
        )
    if value.lower() in _SQL_RESERVED_WORDS:
        raise ValueError(f"{name} {value!r} is a reserved Cosmos SQL keyword")
    return value


def quote_sql_string(value: Any) -> str:
    """Return ``value`` as a single-quoted SQL string literal, escaped.

    Single quotes inside the value are doubled, matching ANSI SQL escaping
    semantics (which Cosmos NoSQL also accepts).
    """
    return "'" + str(value).replace("'", "''") + "'"


# ---------------------------------------------------------------------------
# Embedder protocol
# ---------------------------------------------------------------------------


class EmbedderProtocol(Protocol):
    """Minimal embedder interface accepted by the CosmosDB tools.

    Any object exposing ``embed_documents`` (batch) and ``embed_query`` (single)
    methods returning ``List[float]`` vectors satisfies this protocol. This is
    intentionally compatible with langchain-style ``Embeddings`` instances
    without requiring a langchain dependency.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


# ---------------------------------------------------------------------------
# Retry helper for transient Cosmos errors
# ---------------------------------------------------------------------------


_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 429, 449, 503})

T = TypeVar("T")


def retry_on_cosmos_throttle(
    max_attempts: int = 5,
    initial_backoff: float = 0.5,
    max_backoff: float = 8.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that retries on Cosmos transient HTTP errors (429/503/408).

    The Cosmos SDK exposes a per-error ``retry_after_in_ms`` for 429s; we
    honour it when present, falling back to exponential backoff otherwise.
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                from azure.cosmos.exceptions import CosmosHttpResponseError
            except ImportError:  # pragma: no cover - extras not installed
                return fn(*args, **kwargs)

            backoff = initial_backoff
            last_exc: BaseException | None = None
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except CosmosHttpResponseError as exc:  # noqa: PERF203
                    if exc.status_code not in _RETRYABLE_STATUS_CODES:
                        raise
                    last_exc = exc
                    if attempt == max_attempts - 1:
                        break
                    sleep_for = backoff
                    retry_after_ms = getattr(exc, "retry_after_in_ms", None)
                    if retry_after_ms:
                        sleep_for = max(sleep_for, retry_after_ms / 1000.0)
                    time.sleep(min(sleep_for, max_backoff))
                    backoff = min(backoff * 2, max_backoff)
            assert last_exc is not None  # pragma: no cover - defensive  # noqa: S101
            raise last_exc

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def chunked(seq: Sequence[T], size: int) -> Iterable[list[T]]:
    """Yield successive ``size``-sized chunks from ``seq``."""
    if size <= 0:
        raise ValueError("size must be positive")
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])
