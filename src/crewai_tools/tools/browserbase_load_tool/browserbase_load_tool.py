from typing import Any, Optional, Type

from pydantic import BaseModel, Field

from crewai_tools.tools.base_tool import BaseTool


class BrowserbaseLoadToolSchema(BaseModel):
    url: str = Field(description="Website URL")


class BrowserbaseLoadTool(BaseTool):
    name: str = "Browserbase web load tool"
    description: str = (
        "Load webpages url in a headless browser using Browserbase and return the contents"
    )
    args_schema: Type[BaseModel] = BrowserbaseLoadToolSchema
    api_key: Optional[str] = None
    project_id: Optional[str] = None
    text_content: Optional[bool] = False
    session_id: Optional[str] = None
    proxy: Optional[bool] = None
    browserbase: Optional[Any] = None

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        text_content: Optional[bool] = False,
        session_id: Optional[str] = None,
        proxy: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        try:
            from browserbase import Browserbase  # type: ignore
        except ImportError:
            raise ImportError(
                "`browserbase` package not found, please run `pip install browserbase`"
            )

        self.browserbase = Browserbase(api_key, project_id)
        self.text_content = text_content
        self.session_id = session_id
        self.proxy = proxy

    def _run(self, url: str):
        return self.browserbase.load_url(
            url, self.text_content, self.session_id, self.proxy
        )
