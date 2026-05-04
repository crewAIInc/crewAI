"""Input provider protocol for Flow.ask().

This module provides the InputProvider protocol and InputResponse dataclass
used by Flow.ask() to request input from users during flow execution.

The default implementation is ``ConsoleProvider`` (from
``crewai.flow.async_feedback.providers``), which serves both feedback
and input collection via console.

Example (default console input):
    ```python
    from crewai.flow import Flow, start


    class MyFlow(Flow):
        @start()
        def gather_info(self):
            topic = self.ask("What topic should we research?")
            return topic
    ```

Example (custom provider with metadata):
    ```python
    from crewai.flow import Flow, start
    from crewai.flow.input_provider import InputProvider, InputResponse


    class SlackProvider:
        def request_input(self, message, flow, metadata=None):
            channel = metadata.get("channel", "#general") if metadata else "#general"
            thread = self.post_question(channel, message)
            reply = self.wait_for_reply(thread)
            return InputResponse(
                text=reply.text,
                metadata={"responded_by": reply.user_id, "thread_id": thread.id},
            )


    class MyFlow(Flow):
        input_provider = SlackProvider()

        @start()
        def gather_info(self):
            topic = self.ask("What topic?", metadata={"channel": "#research"})
            return topic
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:
    from crewai.flow.flow import Flow


@dataclass
class InputResponse:
    """Response from an InputProvider, optionally carrying metadata.

    Simple providers can just return a string from ``request_input()``.
    Providers that need to send metadata back (e.g., who responded,
    thread ID, external timestamps) return an ``InputResponse`` instead.

    ``ask()`` normalizes both cases -- callers always get ``str | None``.
    The response metadata is stored in ``_input_history`` and emitted
    in ``FlowInputReceivedEvent``.

    Attributes:
        text: The user's input text, or None if unavailable.
        metadata: Optional metadata from the provider about the response
            (e.g., who responded, thread ID, timestamps).

    Example:
        ```python
        class MyProvider:
            def request_input(self, message, flow, metadata=None):
                response = get_response_from_external_system(message)
                return InputResponse(
                    text=response.text,
                    metadata={"responded_by": response.user_id},
                )
        ```
    """

    text: str | None
    metadata: dict[str, Any] | None = field(default=None)


@runtime_checkable
class InputProvider(Protocol):
    """Protocol for user input collection strategies.

    Implement this protocol to create custom input providers that integrate
    with external systems like websockets, web UIs, Slack, or custom APIs.

    The default provider is ``ConsoleProvider``, which blocks waiting for
    console input via Python's built-in ``input()`` function.

    Providers are always synchronous. The flow framework runs sync methods
    in a thread pool (via ``asyncio.to_thread``), so ``ask()`` never blocks
    the event loop even inside async flow methods.

    Providers can return either:
    - ``str | None`` for simple cases (no response metadata)
    - ``InputResponse`` when they need to send metadata back with the answer

    Example (simple):
        ```python
        class SimpleProvider:
            def request_input(self, message: str, flow: Flow) -> str | None:
                return input(message)
        ```

    Example (with metadata):
        ```python
        class SlackProvider:
            def request_input(self, message, flow, metadata=None):
                channel = metadata.get("channel") if metadata else "#general"
                reply = self.post_and_wait(channel, message)
                return InputResponse(
                    text=reply.text,
                    metadata={"responded_by": reply.user_id},
                )
        ```
    """

    def request_input(
        self,
        message: str,
        flow: Flow[Any],
        metadata: dict[str, Any] | None = None,
    ) -> str | InputResponse | None:
        """Request input from the user.

        Args:
            message: The question or prompt to display to the user.
            flow: The Flow instance requesting input. Can be used to
                access flow state, name, or other context.
            metadata: Optional metadata from the caller, such as user ID,
                channel, session context, etc. Providers can use this to
                route the question to the right recipient.

        Returns:
            The user's input as a string, an ``InputResponse`` with text
            and optional response metadata, or None if input is unavailable
            (e.g., user cancelled, connection dropped).
        """
        ...
