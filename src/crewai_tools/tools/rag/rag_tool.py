from abc import ABC, abstractmethod
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict

from crewai_tools.tools.base_tool import BaseTool


class Adapter(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def query(self, question: str) -> str:
        """Query the knowledge base with a question and return the answer."""

class RagTool(BaseTool):
    name: str = "Knowledge base"
    description: str = "A knowledge base that can be used to answer questions."
    adapter: Optional[Adapter] = None

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

    def from_pg_db(self, db_uri: str, table_name: str):
        from embedchain import App
        from embedchain.models.data_type import DataType
        from embedchain.loaders.postgres import PostgresLoader
        from crewai_tools.adapters.embedchain_adapter import EmbedchainAdapter

        config = { "url":  db_uri }
        postgres_loader = PostgresLoader(config=config)
        app = App()
        app.add(
            f"SELECT * FROM {table_name};",
            data_type='postgres',
            loader=postgres_loader
        )
        adapter = EmbedchainAdapter(embedchain_app=app)
        return RagTool(adapter=adapter)


    def from_github_repo(self, gh_token: str, gh_repo: str, type: List[str] = ["repo"]):
        from embedchain import App
        from embedchain.loaders.github import GithubLoader
        from crewai_tools.adapters.embedchain_adapter import EmbedchainAdapter

        loader = GithubLoader(
            config={
                "token": gh_token,
                }
            )
        app = App()
        app.add(f"repo:{gh_repo} type:{','.join(type)}", data_type="github", loader=loader)
        adapter = EmbedchainAdapter(embedchain_app=app)
        return RagTool(adapter=adapter)

    def from_xml_file(self, file_url: str):
        from embedchain.models.data_type import DataType
        return self._from_generic(file_url, DataType.XML)

    def from_docx_file(self, file_url: str):
        from embedchain.models.data_type import DataType
        return self._from_generic(file_url, DataType.DOCX)

    def from_docx_file(self, file_url: str):
        from embedchain.models.data_type import DataType
        return self._from_generic(file_url, DataType.DOCX)

    def from_mdx_file(self, file_url: str):
        from embedchain.models.data_type import DataType
        return self._from_generic(file_url, DataType.MDX)

    def from_code_docs(self, docs_url: str):
        from embedchain.models.data_type import DataType
        return self._from_generic(docs_url, DataType.DOCS_SITE)

    def from_youtube_channel(self, channel_handle: str):
        from embedchain.models.data_type import DataType
        if not channel_handle.startswith("@"):
            channel_handle = f"@{channel_handle}"
        return self._from_generic(channel_handle, DataType.YOUTUBE_CHANNEL)

    def from_website(self, url: str):
        from embedchain.models.data_type import DataType
        return self._from_generic(url, DataType.WEB_PAGE)

    def from_text(self, text: str):
        from embedchain.models.data_type import DataType
        return self._from_generic(text, DataType.TEXT)

    def from_json(self, file_path: str):
        from embedchain.models.data_type import DataType
        return self._from_generic(file_path, DataType.JSON)

    def from_csv(self, file_path: str):
        from embedchain.models.data_type import DataType
        return self._from_generic(file_path, DataType.CSV)

    def from_pdf(self, file_path: str):
        from embedchain.models.data_type import DataType
        return self._from_generic(file_path, DataType.PDF_FILE)

    def from_web_page(self, url: str):
        from embedchain.models.data_type import DataType
        return self._from_generic(url, DataType.WEB_PAGE)

    def from_embedchain(self, config_path: str):
        from embedchain import App
        from crewai_tools.adapters.embedchain_adapter import EmbedchainAdapter

        app = App.from_config(config_path=config_path)
        adapter = EmbedchainAdapter(embedchain_app=app)
        return RagTool(adapter=adapter)

    def _from_generic(self, source: str, type: str):
        from embedchain import App
        from crewai_tools.adapters.embedchain_adapter import EmbedchainAdapter
        app = App()
        app.add(source, data_type=type)
        adapter = EmbedchainAdapter(embedchain_app=app)
        return RagTool(adapter=adapter)