import os
import json
import requests

from typing import Type, Any
from pydantic.v1 import BaseModel, Field
from crewai_tools.tools.base_tool import BaseTool

class SerperDevToolSchema(BaseModel):
	"""Input for TXTSearchTool."""
	search_query: str = Field(..., description="Mandatory search query you want to use to search the internet")

class SerperDevTool(BaseTool):
	name: str = "Search the internet"
	description: str = "A tool that can be used to semantic search a query from a txt's content."
	args_schema: Type[BaseModel] = SerperDevToolSchema
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
		results = response.json()
		if 'organic' in results:
			results = results['organic']
			stirng = []
			for result in results:
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
		else:
			return results
