"""Base extension interface for CrewAI A2A wrapper processing hooks.

This module defines the protocol for extending CrewAI's A2A wrapper functionality
with custom logic for tool injection, prompt augmentation, and response processing.

Note: These are CrewAI-specific processing hooks, NOT A2A protocol extensions.
A2A protocol extensions are capability declarations using AgentExtension objects
in AgentCard.capabilities.extensions, activated via the A2A-Extensions HTTP header.
See: https://a2a-protocol.org/latest/topics/extensions/
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Any, Protocol, runtime_checkable

from pydantic import BeforeValidator


if TYPE_CHECKING:
    from a2a.types import Message

    from crewai.agent.core import Agent


def _validate_a2a_extension(v: Any) -> Any:
    """Validate that value implements A2AExtension protocol."""
    if not isinstance(v, A2AExtension):
        raise ValueError(
            f"Value must implement A2AExtension protocol. "
            f"Got {type(v).__name__} which is missing required methods."
        )
    return v


ValidatedA2AExtension = Annotated[Any, BeforeValidator(_validate_a2a_extension)]


@runtime_checkable
class ConversationState(Protocol):
    """Protocol for extension-specific conversation state.

    Extensions can define their own state classes that implement this protocol
    to track conversation-specific data extracted from message history.
    """

    def is_ready(self) -> bool:
        """Check if the state indicates readiness for some action.

        Returns:
            True if the state is ready, False otherwise.
        """
        ...


@runtime_checkable
class A2AExtension(Protocol):
    """Protocol for A2A wrapper extensions.

    Extensions can implement this protocol to inject custom logic into
    the A2A conversation flow at various integration points.

    Example:
        class MyExtension:
            def inject_tools(self, agent: Agent) -> None:
                # Add custom tools to the agent
                pass

            def extract_state_from_history(
                self, conversation_history: Sequence[Message]
            ) -> ConversationState | None:
                # Extract state from conversation
                return None

            def augment_prompt(
                self, base_prompt: str, conversation_state: ConversationState | None
            ) -> str:
                # Add custom instructions
                return base_prompt

            def process_response(
                self, agent_response: Any, conversation_state: ConversationState | None
            ) -> Any:
                # Modify response if needed
                return agent_response
    """

    def inject_tools(self, agent: Agent) -> None:
        """Inject extension-specific tools into the agent.

        Called when an agent is wrapped with A2A capabilities. Extensions
        can add tools that enable extension-specific functionality.

        Args:
            agent: The agent instance to inject tools into.
        """
        ...

    def extract_state_from_history(
        self, conversation_history: Sequence[Message]
    ) -> ConversationState | None:
        """Extract extension-specific state from conversation history.

        Called during prompt augmentation to allow extensions to analyze
        the conversation history and extract relevant state information.

        Args:
            conversation_history: The sequence of A2A messages exchanged.

        Returns:
            Extension-specific conversation state, or None if no relevant state.
        """
        ...

    def augment_prompt(
        self,
        base_prompt: str,
        conversation_state: ConversationState | None,
    ) -> str:
        """Augment the task prompt with extension-specific instructions.

        Called during prompt augmentation to allow extensions to add
        custom instructions based on conversation state.

        Args:
            base_prompt: The base prompt to augment.
            conversation_state: Extension-specific state from extract_state_from_history.

        Returns:
            The augmented prompt with extension-specific instructions.
        """
        ...

    def process_response(
        self,
        agent_response: Any,
        conversation_state: ConversationState | None,
    ) -> Any:
        """Process and potentially modify the agent response.

        Called after parsing the agent's response, allowing extensions to
        enhance or modify the response based on conversation state.

        Args:
            agent_response: The parsed agent response.
            conversation_state: Extension-specific state from extract_state_from_history.

        Returns:
            The processed agent response (may be modified or original).
        """
        ...


class ExtensionRegistry:
    """Registry for managing A2A extensions.

    Maintains a collection of extensions and provides methods to invoke
    their hooks at various integration points.
    """

    def __init__(self) -> None:
        """Initialize the extension registry."""
        self._extensions: list[A2AExtension] = []

    def register(self, extension: A2AExtension) -> None:
        """Register an extension.

        Args:
            extension: The extension to register.
        """
        self._extensions.append(extension)

    def inject_all_tools(self, agent: Agent) -> None:
        """Inject tools from all registered extensions.

        Args:
            agent: The agent instance to inject tools into.
        """
        for extension in self._extensions:
            extension.inject_tools(agent)

    def extract_all_states(
        self, conversation_history: Sequence[Message]
    ) -> dict[type[A2AExtension], ConversationState]:
        """Extract conversation states from all registered extensions.

        Args:
            conversation_history: The sequence of A2A messages exchanged.

        Returns:
            Mapping of extension types to their conversation states.
        """
        states: dict[type[A2AExtension], ConversationState] = {}
        for extension in self._extensions:
            state = extension.extract_state_from_history(conversation_history)
            if state is not None:
                states[type(extension)] = state
        return states

    def augment_prompt_with_all(
        self,
        base_prompt: str,
        extension_states: dict[type[A2AExtension], ConversationState],
    ) -> str:
        """Augment prompt with instructions from all registered extensions.

        Args:
            base_prompt: The base prompt to augment.
            extension_states: Mapping of extension types to conversation states.

        Returns:
            The fully augmented prompt.
        """
        augmented = base_prompt
        for extension in self._extensions:
            state = extension_states.get(type(extension))
            augmented = extension.augment_prompt(augmented, state)
        return augmented

    def process_response_with_all(
        self,
        agent_response: Any,
        extension_states: dict[type[A2AExtension], ConversationState],
    ) -> Any:
        """Process response through all registered extensions.

        Args:
            agent_response: The parsed agent response.
            extension_states: Mapping of extension types to conversation states.

        Returns:
            The processed agent response.
        """
        processed = agent_response
        for extension in self._extensions:
            state = extension_states.get(type(extension))
            processed = extension.process_response(processed, state)
        return processed
