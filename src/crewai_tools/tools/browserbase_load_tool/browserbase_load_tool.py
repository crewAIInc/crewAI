import os
from crewai_tools import BaseTool
from typing import Union

class BrowserbaseLoadTool(BaseTool):
    name: str = "Browserbase web load tool"
    description: str = "Load webpages in a headless browser using Browserbase and return the contents"

    def __init__(self, api_key: str = os.environ["BROWSERBASE_KEY"], text_content: bool = False):
        try:
            from browserbase import Browserbase
        except ImportError:
           raise ImportError(
               "`browserbase` package not found, please run `pip install browserbase`"
           )

        self.browserbase = Browserbase(api_key=api_key)
        self.text_content = text_content

    def _run(self, url: str):
        return self.browserbase.load_url(url, text_content=self.text_content)
