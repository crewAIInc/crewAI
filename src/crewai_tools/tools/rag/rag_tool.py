from abc import ABC, abstractmethod
from typing import Any, List, Optional

from pydantic.v1 import BaseModel, ConfigDict

from crewai_tools.tools.base_tool import BaseTool


class Adapter(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def query(self, question: str) -> str:
        """Query the knowledge base with a question and return the answer."""

class RagTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "Knowledge base"
    description: str = "A knowledge base that can be used to answer questions."
    summarize: bool = False
    adapter: Optional[Adapter] = None
    app: Optional[Any] = None

    def _run(
        self,
        query: str,
    ) -> Any:
        from crewai_tools.adapters.embedchain_adapter import EmbedchainAdapter
        self.adapter = EmbedchainAdapter(embedchain_app=self.app, summarize=self.summarize)
        return f"Relevant Content:\n{self.adapter.query(query)}"

    def from_embedchain(self, config_path: str):
        from embedchain import App
        from crewai_tools.adapters.embedchain_adapter import EmbedchainAdapter

        app = App.from_config(config_path=config_path)
        adapter = EmbedchainAdapter(embedchain_app=app)
        return RagTool(name=self.name, description=self.description, adapter=adapter)