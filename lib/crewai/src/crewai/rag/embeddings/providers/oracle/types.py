"""Type definitions for Oracle embedding providers."""

from typing import Any, Literal

from typing_extensions import Required, TypedDict


class OracleProviderConfig(TypedDict, total=False):
    """Configuration for Oracle provider."""

    conn: Any
    connection_params: dict[str, Any]
    embedding_params: dict[str, Any]
    proxy: str


class OracleProviderSpec(TypedDict, total=False):
    """Oracle provider specification."""

    provider: Required[Literal["oracle"]]
    config: OracleProviderConfig
