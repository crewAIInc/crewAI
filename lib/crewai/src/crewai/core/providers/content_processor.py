"""Content processor provider for extensible content processing."""

from contextvars import ContextVar
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ContentProcessorProvider(Protocol):
    """Protocol for content processing during task execution."""

    def process(self, content: str, context: dict[str, Any] | None = None) -> str:
        """Process content before use.

        Args:
            content: The content to process.
            context: Optional context information.

        Returns:
            The processed content.
        """
        ...


class NoOpContentProcessor:
    """Default processor that returns content unchanged."""

    def process(self, content: str, context: dict[str, Any] | None = None) -> str:
        """Return content unchanged.

        Args:
            content: The content to process.
            context: Optional context information (unused).

        Returns:
            The original content unchanged.
        """
        return content


_content_processor: ContextVar[ContentProcessorProvider | None] = ContextVar(
    "_content_processor", default=None
)

_default_processor = NoOpContentProcessor()


def get_processor() -> ContentProcessorProvider:
    """Get the current content processor.

    Returns:
        The registered content processor or the default no-op processor.
    """
    processor = _content_processor.get()
    if processor is not None:
        return processor
    return _default_processor


def set_processor(processor: ContentProcessorProvider) -> None:
    """Set the content processor for the current context.

    Args:
        processor: The content processor to use.
    """
    _content_processor.set(processor)


def process_content(content: str, context: dict[str, Any] | None = None) -> str:
    """Process content using the registered processor.

    Args:
        content: The content to process.
        context: Optional context information.

    Returns:
        The processed content.
    """
    return get_processor().process(content, context)
