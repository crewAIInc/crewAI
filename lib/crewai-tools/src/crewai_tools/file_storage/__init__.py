"""Pluggable backing store for :class:`FileReadTool` / :class:`FileWriterTool`.

The tools default to :class:`LocalFileStore`, which reads and writes the
local filesystem exactly as they always have. A deployment environment where
the local disk is ephemeral can register a different store, so the same tools
persist somewhere durable without the agent, the crew definition, or the tool
arguments changing.

A store owns its own containment. ``resolve`` and ``resolve_within`` must
reject any path the caller should not reach, because the tools call nothing
else before doing I/O.
"""

from crewai_tools.file_storage.base import FileStore, FileStoreError
from crewai_tools.file_storage.local import LocalFileStore
from crewai_tools.file_storage.registry import (
    register_file_store_factory,
    reset_file_store_factory,
    resolve_file_store,
)


__all__ = [
    "FileStore",
    "FileStoreError",
    "LocalFileStore",
    "register_file_store_factory",
    "reset_file_store_factory",
    "resolve_file_store",
]
