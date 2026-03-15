from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import os
import re
from typing import Any


_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_$#]*$")


def get_oracledb_module() -> Any:
    try:
        import oracledb
    except ImportError:
        raise ImportError(
            "`oracledb` package not found, please install the optional dependency with "
            "`uv add 'crewai-tools[oracle]'`"
        ) from None
    return oracledb


def validate_identifier(identifier: str, *, field_name: str) -> str:
    if not identifier or not _IDENTIFIER_PATTERN.match(identifier):
        raise ValueError(
            f"{field_name} must be a simple Oracle identifier starting with a letter."
        )
    return identifier


def get_oracle_connection_kwargs(
    *,
    user: str | None,
    password: str | None,
    dsn: str | None,
    config_dir: str | None = None,
    wallet_location: str | None = None,
    wallet_password: str | None = None,
) -> dict[str, Any]:
    resolved_user = user or os.getenv("ORACLE_DB_USER")
    resolved_password = password or os.getenv("ORACLE_DB_PASSWORD")
    resolved_dsn = dsn or os.getenv("ORACLE_DB_DSN")

    if not resolved_user or not resolved_password or not resolved_dsn:
        raise ValueError(
            "Oracle DB connection requires user, password, and dsn. "
            "Set them explicitly or via ORACLE_DB_USER, ORACLE_DB_PASSWORD, and "
            "ORACLE_DB_DSN."
        )

    kwargs: dict[str, Any] = {
        "user": resolved_user,
        "password": resolved_password,
        "dsn": resolved_dsn,
    }

    resolved_config_dir = config_dir or os.getenv("ORACLE_DB_CONFIG_DIR")
    resolved_wallet_location = wallet_location or os.getenv("ORACLE_DB_WALLET_LOCATION")
    resolved_wallet_password = wallet_password or os.getenv("ORACLE_DB_WALLET_PASSWORD")

    if resolved_config_dir:
        kwargs["config_dir"] = resolved_config_dir
    if resolved_wallet_location:
        kwargs["wallet_location"] = resolved_wallet_location
    if resolved_wallet_password:
        kwargs["wallet_password"] = resolved_wallet_password

    return kwargs


@contextmanager
def oracle_connection_context(
    client: Any = None, **connect_kwargs: Any
) -> Iterator[Any]:
    if client is not None:
        yield client
        return

    oracledb = get_oracledb_module()
    connection = oracledb.connect(**connect_kwargs)
    try:
        yield connection
    finally:
        connection.close()
