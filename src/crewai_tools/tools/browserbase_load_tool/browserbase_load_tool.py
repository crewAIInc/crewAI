import os
from typing import Any, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class BrowserbaseLoadToolSchema(BaseModel):
    url: str = Field(description="Website URL")


class BrowserbaseLoadTool(BaseTool):
    name: str = "Browserbase web load tool"
    description: str = "Load webpages url in a headless browser using Browserbase and return the contents"
    args_schema: Type[BaseModel] = BrowserbaseLoadToolSchema
    api_key: Optional[str] = os.getenv("BROWSERBASE_API_KEY")
    project_id: Optional[str] = os.getenv("BROWSERBASE_PROJECT_ID")
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
        if not self.api_key:
            raise EnvironmentError(
                "BROWSERBASE_API_KEY environment variable is required for initialization"
            )
        try:
            from browserbase import Browserbase  # type: ignore
        except ImportError:
            raise ImportError(
                "`browserbase` package not found, please run `pip install browserbase`"
            )

        self.browserbase = Browserbase(api_key=self.api_key)
        self.text_content = text_content
        self.session_id = session_id
        self.proxy = proxy

    def _run(self, url: str):
        return self.browserbase.load_url(
            url, self.text_content, self.session_id, self.proxy
        )
