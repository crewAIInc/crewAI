from crewai.tools import BaseTool
from pydantic import Field
from typing import TYPE_CHECKING, Any, Dict, List
import os

if TYPE_CHECKING:
    from langchain_apify import ApifyActorsTool as _ApifyActorsTool

class ApifyActorsTool(BaseTool):
    """Tool that runs Apify Actors.

       To use, you should have the environment variable `APIFY_API_TOKEN` set
       with your API key.

       For details, see https://docs.apify.com/platform/integrations/crewai

       Args:
           actor_name (str): The name of the Apify Actor to run.
           *args: Variable length argument list passed to BaseTool.
           **kwargs: Arbitrary keyword arguments passed to BaseTool.

       Returns:
           List[Dict[str, Any]]: Results from the Actor execution.

       Raises:
           ValueError: If `APIFY_API_TOKEN` is not set or if the tool is not initialized.
           ImportError: If `langchain_apify` package is not installed.

       Example:
           .. code-block:: python
            from crewai_tools import ApifyActorsTool

            tool = ApifyActorsTool(actor_name="apify/rag-web-browser")

            results = tool.run(run_input={"query": "What is CrewAI?", "maxResults": 5})
            for result in results:
                print(f"URL: {result['metadata']['url']}")
                print(f"Content: {result.get('markdown', 'N/A')[:100]}...")
    """
    actor_tool: '_ApifyActorsTool' = Field(description="Apify Actor Tool")

    def __init__(
        self,
        actor_name: str,
        *args: Any,
        **kwargs: Any
    ) -> None:
        if not os.environ.get("APIFY_API_TOKEN"):
            msg = (
                "APIFY_API_TOKEN environment variable is not set. "
                "Please set it to your API key, to learn how to get it, "
                "see https://docs.apify.com/platform/integrations/api"
            )
            raise ValueError(msg)

        try:
            from langchain_apify import ApifyActorsTool as _ApifyActorsTool
        except ImportError:
            raise ImportError(
                "Could not import langchain_apify python package. "
                "Please install it with `pip install langchain-apify` or `uv add langchain-apify`."
            )
        actor_tool = _ApifyActorsTool(actor_name)

        kwargs.update(
            {
                "name": actor_tool.name,
                "description": actor_tool.description,
                "args_schema": actor_tool.args_schema,
                "actor_tool": actor_tool,
            }
        )
        super().__init__(*args, **kwargs)

    def _run(self, run_input: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run the Actor tool with the given input.

        Returns:
            List[Dict[str, Any]]: Results from the Actor execution.

        Raises:
            ValueError: If 'actor_tool' is not initialized.
        """
        try:
            return self.actor_tool._run(run_input)
        except Exception as e:
            msg = (
                f'Failed to run ApifyActorsTool {self.name}. '
                'Please check your Apify account Actor run logs for more details.'
                f'Error: {e}'
            )
            raise RuntimeError(msg) from e
