"""Generic metaclass for agent extensions.

This metaclass enables extension capabilities for agents by detecting
extension fields in class annotations and applying appropriate wrappers.
"""

from typing import Any
import warnings

from pydantic import model_validator
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
        orig_post_init_setup = namespace.get("post_init_setup")

        if orig_post_init_setup is not None:
            original_func = (
                orig_post_init_setup.wrapped
                if hasattr(orig_post_init_setup, "wrapped")
                else orig_post_init_setup
            )

            def post_init_setup_with_extensions(self: Any) -> Any:
                """Wrap post_init_setup to apply extensions after initialization.

                Args:
                    self: The agent instance

                Returns:
                    The agent instance
                """
                result = original_func(self)

                a2a_value = getattr(self, "a2a", None)
                if a2a_value is not None:
                    from crewai.a2a.extensions.registry import (
                        create_extension_registry_from_config,
                    )
                    from crewai.a2a.wrapper import wrap_agent_with_a2a_instance

                    extension_registry = create_extension_registry_from_config(
                        a2a_value
                    )
                    wrap_agent_with_a2a_instance(self, extension_registry)

                return result

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=".*overrides an existing Pydantic.*"
                )
                namespace["post_init_setup"] = model_validator(mode="after")(
                    post_init_setup_with_extensions
                )

        return super().__new__(mcs, name, bases, namespace, **kwargs)
