from crewai_tools import BaseTool

class {{class_name}}(BaseTool):
    name: str = "Name of my tool"
    description: str = "What this tool does. It's vital for effective utilization."

    def _run(self, argument: str) -> str:
        # Your tool's logic here
        return "Tool's result"
