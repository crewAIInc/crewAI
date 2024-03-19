from typing import Callable
from crewai_tools import BaseTool, tool

def test_creating_a_tool_using_annotation():
	@tool("Name of my tool")
	def my_tool(question: str) -> str:
		"""Clear description for what this tool is useful for, you agent will need this information to use it."""
		return question

	# Assert all the right attributes were defined
	assert my_tool.name == "Name of my tool"
	assert my_tool.description == "Name of my tool(question: 'string') - Clear description for what this tool is useful for, you agent will need this information to use it."
	assert my_tool.args_schema.schema()["properties"] == {'question': {'title': 'Question', 'type': 'string'}}
	assert my_tool.func("What is the meaning of life?") == "What is the meaning of life?"

	# Assert the langchain tool conversion worked as expected
	converted_tool = my_tool.to_langchain()
	assert converted_tool.name == "Name of my tool"
	assert converted_tool.description == "Name of my tool(question: 'string') - Clear description for what this tool is useful for, you agent will need this information to use it."
	assert converted_tool.args_schema.schema()["properties"] == {'question': {'title': 'Question', 'type': 'string'}}
	assert converted_tool.func("What is the meaning of life?") == "What is the meaning of life?"

def test_creating_a_tool_using_baseclass():
	class MyCustomTool(BaseTool):
		name: str = "Name of my tool"
		description: str = "Clear description for what this tool is useful for, you agent will need this information to use it."

		def _run(self, question: str) -> str:
			return question

	my_tool = MyCustomTool()
	# Assert all the right attributes were defined
	assert my_tool.name == "Name of my tool"
	assert my_tool.description == "Name of my tool(question: 'string') - Clear description for what this tool is useful for, you agent will need this information to use it."
	assert my_tool.args_schema.schema()["properties"] == {'question': {'title': 'Question', 'type': 'string'}}
	assert my_tool.run("What is the meaning of life?") == "What is the meaning of life?"

	# Assert the langchain tool conversion worked as expected
	converted_tool = my_tool.to_langchain()
	assert converted_tool.name == "Name of my tool"
	assert converted_tool.description == "Name of my tool(question: 'string') - Clear description for what this tool is useful for, you agent will need this information to use it."
	assert converted_tool.args_schema.schema()["properties"] == {'question': {'title': 'Question', 'type': 'string'}}
	assert converted_tool.run("What is the meaning of life?") == "What is the meaning of life?"

def test_setting_cache_function():
	class MyCustomTool(BaseTool):
		name: str = "Name of my tool"
		description: str = "Clear description for what this tool is useful for, you agent will need this information to use it."
		cache_function: Callable = lambda: False

		def _run(self, question: str) -> str:
			return question

	my_tool = MyCustomTool()
	# Assert all the right attributes were defined
	assert my_tool.cache_function() == False

def test_default_cache_function_is_true():
	class MyCustomTool(BaseTool):
		name: str = "Name of my tool"
		description: str = "Clear description for what this tool is useful for, you agent will need this information to use it."

		def _run(self, question: str) -> str:
			return question

	my_tool = MyCustomTool()
	# Assert all the right attributes were defined
	assert my_tool.cache_function() == True