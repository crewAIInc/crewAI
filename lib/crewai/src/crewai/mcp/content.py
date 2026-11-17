"""Flatten MCP tool result content into the text an agent sees."""

from typing import Any


def content_blocks_to_text(content: Any) -> str:
    """Join every block of an MCP ``CallToolResult.content`` into one string.

    Text blocks and embedded text resources keep their text, one block per
    line. Image, audio, blob and link blocks become short typed placeholders
    such as ``[image: image/png]``; their base64 ``data`` and ``blob`` fields
    are not copied into the result.

    Args:
        content: The ``content`` list of a ``CallToolResult``.

    Returns:
        The text of all blocks joined with newlines.
    """
    from mcp.types import (
        AudioContent,
        EmbeddedResource,
        ImageContent,
        ResourceLink,
        TextContent,
        TextResourceContents,
    )

    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for block in content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, EmbeddedResource):
            resource = block.resource
            if isinstance(resource, TextResourceContents):
                parts.append(resource.text)
            else:
                parts.append(f"[{block.type}: {resource.uri}]")
        elif isinstance(block, ResourceLink):
            parts.append(f"[{block.type}: {block.uri}]")
        elif isinstance(block, (ImageContent, AudioContent)):
            parts.append(f"[{block.type}: {block.mimeType}]")
        else:
            parts.append(str(block))
    return "\n".join(parts)
