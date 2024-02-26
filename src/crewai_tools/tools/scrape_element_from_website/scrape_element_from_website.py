import requests
from bs4 import BeautifulSoup
from typing import Optional, Type, Any
from pydantic.v1 import BaseModel, Field
from ..base_tool import BaseTool

class FixedScrapeElementFromWebsiteToolSchema(BaseModel):
	"""Input for ScrapeElementFromWebsiteTool."""
	pass

class ScrapeElementFromWebsiteToolSchema(FixedScrapeElementFromWebsiteToolSchema):
	"""Input for ScrapeElementFromWebsiteTool."""
	website_url: str = Field(..., description="Mandatory website url to read the file")
	css_element: str = Field(..., description="Mandatory css reference for element to scrape from the website")

class ScrapeElementFromWebsiteTool(BaseTool):
	name: str = "Read a website content"
	description: str = "A tool that can be used to read a website content."
	args_schema: Type[BaseModel] = ScrapeElementFromWebsiteToolSchema
	website_url: Optional[str] = None
	css_element: Optional[str] = None
	headers: Optional[dict] = {
		'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
		'Accept-Language': 'en-US,en;q=0.5',
		'Referer': 'https://www.google.com/'
	}

	def __init__(self, website_url: Optional[str] = None, css_element: Optional[str] = None, **kwargs):
		super().__init__(**kwargs)
		if website_url is not None:
			self.website_url = website_url
			self.css_element = css_element
			self.description = f"A tool that can be used to read {website_url}'s content."
			self.args_schema = FixedScrapeElementFromWebsiteToolSchema

	def _run(
		self,
		**kwargs: Any,
	) -> Any:
		website_url = kwargs.get('website_url', self.website_url)
		css_element = kwargs.get('css_element', self.css_element)
		page = requests.get(website_url, headers=self.headers)
		parsed = BeautifulSoup(page.content, "html.parser")
		elements = parsed.select(css_element)
		return "\n".join([element.get_text() for element in elements])



