import os
import json
import requests

from typing import Type, Any
from pydantic.v1 import BaseModel, Field
from crewai_tools.tools.base_tool import BaseTool

class SeperDevToolSchema(BaseModel):
	"""Input for TXTSearchTool."""
	search_query: str = Field(..., description="Mandatory search query you want to use to search the internet")

class SeperDevTool(BaseTool):
	name: str = "Search the internet"
	description: str = "A tool that can be used to semantic search a query from a txt's content."
	args_schema: Type[BaseModel] = SeperDevToolSchema
	search_url: str = "https://google.serper.dev/search"
	n_results: int = None

	def _run(
		self,
		search_query: str,
		**kwargs: Any,
	) -> Any:
		payload = json.dumps({"q": search_query})
		headers = {
				'X-API-KEY': os.environ['SERPER_API_KEY'],
				'content-type': 'application/json'
		}
		response = requests.request("POST", self.search_url, headers=headers, data=payload)
		results = response.json()['organic']
		stirng = []
		for result in results:
			print(result)
			print('--------------')
			try:
				stirng.append('\n'.join([
						f"Title: {result['title']}",
						f"Link: {result['link']}",
						f"Snippet: {result['snippet']}",
						"---"
				]))
			except KeyError:
				next

		content = '\n'.join(stirng)
		return f"\nSearch results: {content}\n"
