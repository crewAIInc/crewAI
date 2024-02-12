from abc import ABC, abstractmethod
from typing import Optional, List, Any, Dict

from pydantic import BaseModel, PrivateAttr, Field

from crewai.utilities import I18N


class AgentWrapperParent(ABC, BaseModel):
    _i18n: I18N = PrivateAttr(default=I18N())
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Data storage for children, as pydantic doesn't play well with inheritance.",
    )
    role: str = Field(description="Role of the agent", default="")
    allow_delegation: bool = Field(
        description="Allow delegation of tasks to other agents?", default=False
    )

    @property
    def i18n(self) -> I18N:
        if hasattr(self, "_agent") and hasattr(self._agent, "i18n"):
            return self._agent.i18n
        else:
            return self._i18n

    @i18n.setter
    def i18n(self, value: I18N) -> None:
        if hasattr(self, "_agent") and hasattr(self._agent, "i18n"):
            self._agent.i18n = value
        else:
            self._i18n = value

    @abstractmethod
    def execute_task(
        self,
        task: str,
        context: Optional[str] = None,
        tools: Optional[List[Any]] = None,
    ) -> str:
        pass

    @property
    @abstractmethod
    def tools(self) -> List[Any]:
        pass

    def set_cache_handler(self, cache_handler: Any) -> None:
        pass

    def set_rpm_controller(self, rpm_controller: Any) -> None:
        pass
