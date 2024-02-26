import requests
from bs4 import BeautifulSoup
from typing import Optional, Type, Any
from pydantic.v1 import BaseModel, Field
from ..base_tool import BaseTool

class FixedScrapeWebsiteToolSchema(BaseModel):
	"""Input for ScrapeWebsiteTool."""
	pass

class ScrapeWebsiteToolSchema(FixedScrapeWebsiteToolSchema):
	"""Input for ScrapeWebsiteTool."""
	website_url: str = Field(..., description="Mandatory website url to read the file")

class ScrapeWebsiteTool(BaseTool):
	name: str = "Read a website content"
	description: str = "A tool that can be used to read a website content."
	args_schema: Type[BaseModel] = ScrapeWebsiteToolSchema
	website_url: Optional[str] = None
	headers: Optional[dict] = {
		'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
		'Accept-Language': 'en-US,en;q=0.5',
		'Referer': 'https://www.google.com/'
	}

	def __init__(self, website_url: Optional[str] = None, **kwargs):
		super().__init__(**kwargs)
		if website_url is not None:
			self.website_url = website_url
			self.description = f"A tool that can be used to read {website_url}'s content."
			self.args_schema = FixedScrapeWebsiteToolSchema

	def _run(
		self,
		**kwargs: Any,
	) -> Any:
		website_url = kwargs.get('website_url', self.website_url)
		page = requests.get(website_url, headers=self.headers)
		parsed = BeautifulSoup(page.content, "html.parser")
		return parsed.get_text()

