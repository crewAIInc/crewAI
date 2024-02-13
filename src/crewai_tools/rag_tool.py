from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict

from crewai_tools.base_tool import BaseTool


class Adapter(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def query(self, question: str) -> str:
        """Query the knowledge base with a question and return the answer."""


class RagTool(BaseTool):
    name: str = "Knowledge base"
    description: str = "A knowledge base that can be used to answer questions."
    adapter: Adapter

    def _run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return self.adapter.query(args[0])

    def from_file(self, file_path: str):
        from embedchain import App
        from embedchain.models.data_type import DataType

        from crewai_tools.adapters.embedchain_adapter import EmbedchainAdapter

        app = App()
        app.add(file_path, data_type=DataType.TEXT_FILE)

        adapter = EmbedchainAdapter(embedchain_app=app)
        return RagTool(adapter=adapter)

    def from_directory(self, directory_path: str):
        from embedchain import App
        from embedchain.loaders.directory_loader import DirectoryLoader

        from crewai_tools.adapters.embedchain_adapter import EmbedchainAdapter

        loader = DirectoryLoader(config=dict(recursive=True))

        app = App()
        app.add(directory_path, loader=loader)

        adapter = EmbedchainAdapter(embedchain_app=app)
        return RagTool(adapter=adapter)

    def from_web_page(self, url: str):
        from embedchain import App
        from embedchain.models.data_type import DataType

        from crewai_tools.adapters.embedchain_adapter import EmbedchainAdapter

        app = App()
        app.add(url, data_type=DataType.WEB_PAGE)

        adapter = EmbedchainAdapter(embedchain_app=app)
        return RagTool(adapter=adapter)

    def from_embedchain(self, config_path: str):
        from embedchain import App

        from crewai_tools.adapters.embedchain_adapter import EmbedchainAdapter

        app = App.from_config(config_path=config_path)
        adapter = EmbedchainAdapter(embedchain_app=app)
        return RagTool(adapter=adapter)
