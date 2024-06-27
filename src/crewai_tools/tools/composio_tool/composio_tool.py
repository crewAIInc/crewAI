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

    @staticmethod
    def _check_connected_account(tool: t.Any, toolset: t.Any) -> None:
        """Check if connected account is required and if required it exists or not."""
        from composio import Action
        from composio.client.collections import ConnectedAccountModel

        tool = t.cast(Action, tool)
        if tool.no_auth:
            return

        connections = t.cast(
            t.List[ConnectedAccountModel],
            toolset.client.connected_accounts.get(),
        )
        if tool.app not in [connection.appUniqueId for connection in connections]:
            raise RuntimeError(
                f"No connected account found for app `{tool.app}`; "
                f"Run `composio add {tool.app}` to fix this"
            )

    @classmethod
    def from_tool(
        cls,
        tool: t.Any,
        **kwargs: t.Any,
    ) -> te.Self:
        """Wrap a composio tool as crewAI tool."""

        from composio import Action, ComposioToolSet
        from composio.constants import DEFAULT_ENTITY_ID
        from composio.utils.shared import json_schema_to_model

        toolset = ComposioToolSet()
        if not isinstance(tool, Action):
            tool = Action(tool)

        tool = t.cast(Action, tool)
        cls._check_connected_account(
            tool=tool,
            toolset=toolset,
        )

        (action,) = toolset.get_action_schemas(actions=[tool])
        schema = action.model_dump(exclude_none=True)
        entity_id = kwargs.pop("entity_id", DEFAULT_ENTITY_ID)

        def function(**kwargs: t.Any) -> t.Dict:
            """Wrapper function for composio action."""
            return toolset.execute_action(
                action=Action(schema["name"]),
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
            **kwargs,
        )

    @classmethod
    def from_app(
        cls,
        app: t.Any,
        tags: t.Optional[t.List[str]] = None,
        **kwargs: t.Any,
    ) -> t.List[te.Self]:
        """Create toolset from an app."""
        from composio import App

        if not isinstance(app, App):
            app = App(app)

        return [
            cls.from_tool(tool=action, **kwargs)
            for action in app.get_actions(tags=tags)
        ]

    @classmethod
    def from_use_case(
        cls,
        *apps: t.Any,
        use_case: str,
        **kwargs: t.Any,
    ) -> t.List[te.Self]:
        """Create toolset from an app."""
        if len(apps) == 0:
            raise ValueError(
                "You need to provide at least one app name to search by use case"
            )

        from composio import ComposioToolSet

        toolset = ComposioToolSet()
        actions = toolset.find_actions_by_use_case(*apps, use_case=use_case)
        return [cls.from_tool(tool=action, **kwargs) for action in actions]
