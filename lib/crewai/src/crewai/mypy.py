"""Mypy plugin for CrewAI decorator type checking.

This plugin informs mypy about attributes injected by the @CrewBase decorator.
"""

from collections.abc import Callable

from mypy.nodes import MDEF, SymbolTableNode, Var
from mypy.plugin import ClassDefContext, Plugin
from mypy.types import AnyType, TypeOfAny


class CrewAIPlugin(Plugin):
    """Mypy plugin that handles @CrewBase decorator attribute injection."""

    def get_class_decorator_hook(
        self, fullname: str
    ) -> Callable[[ClassDefContext], None] | None:
        """Return hook for class decorators.

        Args:
            fullname: Fully qualified name of the decorator.

        Returns:
            Hook function if this is a CrewBase decorator, None otherwise.
        """
        if fullname in ("crewai.project.CrewBase", "crewai.project.crew_base.CrewBase"):
            return self._crew_base_hook
        return None

    @staticmethod
    def _crew_base_hook(ctx: ClassDefContext) -> None:
        """Add injected attributes to @CrewBase decorated classes.

        Args:
            ctx: Context for the class being decorated.
        """
        any_type = AnyType(TypeOfAny.explicit)
        str_type = ctx.api.named_type("builtins.str")
        dict_type = ctx.api.named_type("builtins.dict", [str_type, any_type])
        agents_config_var = Var("agents_config", dict_type)
        agents_config_var.info = ctx.cls.info
        agents_config_var._fullname = f"{ctx.cls.info.fullname}.agents_config"
        ctx.cls.info.names["agents_config"] = SymbolTableNode(MDEF, agents_config_var)
        tasks_config_var = Var("tasks_config", dict_type)
        tasks_config_var.info = ctx.cls.info
        tasks_config_var._fullname = f"{ctx.cls.info.fullname}.tasks_config"
        ctx.cls.info.names["tasks_config"] = SymbolTableNode(MDEF, tasks_config_var)


def plugin(_: str) -> type[Plugin]:
    """Entry point for mypy plugin.

    Args:
        _: Mypy version string.

    Returns:
        Plugin class.
    """
    return CrewAIPlugin
