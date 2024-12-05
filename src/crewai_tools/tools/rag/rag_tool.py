from abc import ABC, abstractmethod
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, model_validator


class Adapter(BaseModel, ABC):
    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def query(self, question: str) -> str:
        """Query the knowledge base with a question and return the answer."""

    @abstractmethod
    def add(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Add content to the knowledge base."""


class RagTool(BaseTool):
    class _AdapterPlaceholder(Adapter):
        def query(self, question: str) -> str:
            raise NotImplementedError

        def add(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError

    name: str = "Knowledge base"
    description: str = "A knowledge base that can be used to answer questions."
    summarize: bool = False
    adapter: Adapter = Field(default_factory=_AdapterPlaceholder)
    config: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _set_default_adapter(self):
        if isinstance(self.adapter, RagTool._AdapterPlaceholder):
            from embedchain import App

            from crewai_tools.adapters.embedchain_adapter import EmbedchainAdapter

            app = App.from_config(config=self.config) if self.config else App()
            self.adapter = EmbedchainAdapter(
                embedchain_app=app, summarize=self.summarize
            )

        return self

    def add(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.adapter.add(*args, **kwargs)

    def _run(
        self,
        query: str,
        **kwargs: Any,
    ) -> Any:
        self._before_run(query, **kwargs)

        return f"Relevant Content:\n{self.adapter.query(query)}"

    def _before_run(self, query, **kwargs):
        pass
