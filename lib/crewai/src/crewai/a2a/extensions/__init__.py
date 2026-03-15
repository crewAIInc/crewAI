"""A2A Protocol Extensions for CrewAI.

This module contains extensions to the A2A (Agent-to-Agent) protocol.

**Client-side extensions** (A2AExtension) allow customizing how the A2A wrapper
processes requests and responses during delegation to remote agents. These provide
hooks for tool injection, prompt augmentation, and response processing.

**Server-side extensions** (ServerExtension) allow agents to offer additional
functionality beyond the core A2A specification. Clients activate extensions
via the X-A2A-Extensions header.

See: https://a2a-protocol.org/latest/topics/extensions/
"""

from crewai.a2a.extensions.base import (
    A2AExtension,
    ConversationState,
    ExtensionRegistry,
    ValidatedA2AExtension,
)
from crewai.a2a.extensions.server import (
    ExtensionContext,
    ServerExtension,
    ServerExtensionRegistry,
)


__all__ = [
    "A2AExtension",
    "ConversationState",
    "ExtensionContext",
    "ExtensionRegistry",
    "ServerExtension",
    "ServerExtensionRegistry",
    "ValidatedA2AExtension",
]
