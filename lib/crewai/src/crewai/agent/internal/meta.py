"""Generic metaclass for agent extensions.

This metaclass enables extension capabilities for agents by detecting
extension fields in class annotations and applying appropriate wrappers.
"""

from typing import Any

from pydantic._internal._model_construction import ModelMetaclass


class AgentMeta(ModelMetaclass):
    """Generic metaclass for agent extensions.

    Detects extension fields (like 'a2a') in class annotations and applies
    the appropriate wrapper logic to enable extension functionality.
    """

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> type:
        """Create a new class with extension support.

        Args:
            name: The name of the class being created
            bases: Base classes
            namespace: Class namespace dictionary
            **kwargs: Additional keyword arguments

        Returns:
            The newly created class with extension support if applicable
        """
        namespace_annotations: dict[str, Any] = namespace.get("__annotations__", {})

        # Check for A2A extension
        if "a2a" in namespace_annotations:
            from crewai.a2a.wrapper import wrap_agent_with_a2a

            wrapped_methods = wrap_agent_with_a2a(namespace, bases)
            namespace.update(wrapped_methods)

        # Future extensions can be added here:
        # if "some_other_extension" in namespace_annotations:
        #     from crewai.extensions.foo import wrap_agent_with_foo
        #     wrapped_methods = wrap_agent_with_foo(namespace, bases)
        #     namespace.update(wrapped_methods)

        return super().__new__(mcs, name, bases, namespace, **kwargs)
