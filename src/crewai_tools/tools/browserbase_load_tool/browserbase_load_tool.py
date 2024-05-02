from typing import Optional, Any, Type
from pydantic.v1 import BaseModel, Field
from crewai_tools.tools.base_tool import BaseTool

class BrowserbaseLoadToolSchema(BaseModel):
    url: str = Field(description="Website URL")

class BrowserbaseLoadTool(BaseTool):
    name: str = "Browserbase web load tool"
    description: str = "Load webpages url in a headless browser using Browserbase and return the contents"
    args_schema: Type[BaseModel] = BrowserbaseLoadToolSchema
    api_key: Optional[str] = None
    text_content: Optional[bool] = False
    browserbase: Optional[Any] = None

    def __init__(self, api_key: Optional[str] = None, text_content: Optional[bool] = False, **kwargs):
        super().__init__(**kwargs)
        try:
            from browserbase import Browserbase # type: ignore
        except ImportError:
           raise ImportError(
               "`browserbase` package not found, please run `pip install browserbase`"
           )

        self.browserbase = Browserbase(api_key=api_key)
        self.text_content = text_content

    def _run(self, url: str):
        return self.browserbase.load_url(url, text_content=self.text_content)
