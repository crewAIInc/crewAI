import os
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
	cookies: Optional[dict] = None
	css_element: Optional[str] = None
	headers: Optional[dict] = {
		'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
		'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
		'Accept-Language': 'en-US,en;q=0.9',
		'Referer': 'https://www.google.com/',
		'Connection': 'keep-alive',
		'Upgrade-Insecure-Requests': '1',
		'Accept-Encoding': 'gzip, deflate, br'
	}

	def __init__(self, website_url: Optional[str] = None, cookies: Optional[dict] = None, css_element: Optional[str] = None, **kwargs):
		super().__init__(**kwargs)
		if website_url is not None:
			self.website_url = website_url
			self.css_element = css_element
			self.description = f"A tool that can be used to read {website_url}'s content."
			self.args_schema = FixedScrapeElementFromWebsiteToolSchema
			self._generate_description()
			if cookies is not None:
				self.cookies = {cookies["name"]: os.getenv(cookies["value"])}

	def _run(
		self,
		**kwargs: Any,
	) -> Any:
		website_url = kwargs.get('website_url', self.website_url)
		css_element = kwargs.get('css_element', self.css_element)
		page = requests.get(website_url, headers=self.headers, cookies=self.cookies if self.cookies else {})
		parsed = BeautifulSoup(page.content, "html.parser")
		elements = parsed.select(css_element)
		return "\n".join([element.get_text() for element in elements])



