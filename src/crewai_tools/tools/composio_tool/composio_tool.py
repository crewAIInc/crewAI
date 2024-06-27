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
    def from_action(
        cls,
        action: t.Any,
        **kwargs: t.Any,
    ) -> te.Self:
        """Wrap a composio tool as crewAI tool."""

        from composio import Action, ComposioToolSet
        from composio.constants import DEFAULT_ENTITY_ID
        from composio.utils.shared import json_schema_to_model

        toolset = ComposioToolSet()
        if not isinstance(action, Action):
            action = Action(action)

        action = t.cast(Action, action)
        cls._check_connected_account(
            tool=action,
            toolset=toolset,
        )

        (action_schema,) = toolset.get_action_schemas(actions=[action])
        schema = action_schema.model_dump(exclude_none=True)
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
                action_schema.parameters.model_dump(
                    exclude_none=True,
                )
            ),
            composio_action=function,
            **kwargs,
        )

    @classmethod
    def from_app(
        cls,
        *apps: t.Any,
        tags: t.Optional[t.List[str]] = None,
        use_case: t.Optional[str] = None,
        **kwargs: t.Any,
    ) -> t.List[te.Self]:
        """Create toolset from an app."""
        if len(apps) == 0:
            raise ValueError("You need to provide at least one app name")

        if use_case is None and tags is None:
            raise ValueError("Both `use_case` and `tags` cannot be `None`")

        if use_case is not None and tags is not None:
            raise ValueError(
                "Cannot use both `use_case` and `tags` to filter the actions"
            )

        from composio import ComposioToolSet

        toolset = ComposioToolSet()
        if use_case is not None:
            return [
                cls.from_action(action=action, **kwargs)
                for action in toolset.find_actions_by_use_case(*apps, use_case=use_case)
            ]

        return [
            cls.from_action(action=action, **kwargs)
            for action in toolset.find_actions_by_tags(*apps, tags=tags)
        ]
