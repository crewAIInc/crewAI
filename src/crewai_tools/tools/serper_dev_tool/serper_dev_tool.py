import os
import json
import requests

from typing import Type, Any
from pydantic.v1 import BaseModel, Field
from crewai_tools.tools.base_tool import BaseTool

class SerperDevToolSchema(BaseModel):
	"""Input for SerperDevTool."""
	search_query: str = Field(..., description="Mandatory search query you want to use to search the internet")

class SerperDevTool(BaseTool):
	name: str = "Search the internet"
	description: str = "A tool that can be used to search the internet with a search_query."
	args_schema: Type[BaseModel] = SerperDevToolSchema
	search_url: str = "https://google.serper.dev/search"
	n_results: int = 10

	def _run(
		self,
		**kwargs: Any,
	) -> Any:
		search_query = kwargs.get('search_query')
		if search_query is None:
			search_query = kwargs.get('query')

		payload = json.dumps({"q": search_query})
		headers = {
				'X-API-KEY': os.environ['SERPER_API_KEY'],
				'content-type': 'application/json'
		}
		response = requests.request("POST", self.search_url, headers=headers, data=payload)
		results = response.json()
		if 'organic' in results:
			results = results['organic']
			string = []
			for result in results:
				try:
					string.append('\n'.join([
							f"Title: {result['title']}",
							f"Link: {result['link']}",
							f"Snippet: {result['snippet']}",
							"---"
					]))
				except KeyError:
					next

			content = '\n'.join(string)
			return f"\nSearch results: {content}\n"
		else:
			return results
