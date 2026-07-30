"""Internal compatibility helpers for MCP async lifecycle management."""

import sys


if sys.version_info >= (3, 11):
    from asyncio import timeout as async_timeout
else:
    from async_timeout import timeout as async_timeout


__all__ = ["async_timeout"]
