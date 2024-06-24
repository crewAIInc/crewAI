"""
Composio tools wrapper.
"""

import typing as t

import typing_extensions as te

from crewai_tools.tools.base_tool import BaseTool


class ComposioTool(BaseTool):
    """Wrapper for composio tools."""

    composio_action: t.Callable

    def _run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """Run the composio action with given arguments."""
        return self.composio_action(*args, **kwargs)

    @classmethod
    def from_tool(cls, tool: t.Any, **kwargs: t.Any) -> te.Self:
        """Wrap a composio tool as crewAI tool."""

        from composio import Action, ComposioToolSet
        from composio.constants import DEFAULT_ENTITY_ID
        from composio.utils.shared import json_schema_to_model

        toolset = ComposioToolSet()
        if not isinstance(tool, Action):
            tool = Action.from_action(name=tool)

        tool = t.cast(Action, tool)
        (action,) = toolset.get_action_schemas(actions=[tool])
        schema = action.model_dump(exclude_none=True)
        entity_id = kwargs.pop("entity_id", DEFAULT_ENTITY_ID)

        def function(**kwargs: t.Any) -> t.Dict:
            """Wrapper function for composio action."""
            return toolset.execute_action(
                action=Action.from_app_and_action(
                    app=schema["appName"],
                    name=schema["name"],
                ),
                params=kwargs,
                entity_id=entity_id,
            )

        function.__name__ = schema["name"]
        function.__doc__ = schema["description"]

        return cls(
            name=schema["name"],
            description=schema["description"],
            args_schema=json_schema_to_model(
                action.parameters.model_dump(
                    exclude_none=True,
                )
            ),
            composio_action=function,
            **kwargs
        )
