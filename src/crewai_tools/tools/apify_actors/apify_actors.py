from crewai.tools import BaseTool
from pydantic import Field
from typing import Any, Dict, List
import os

try:
    from langchain_apify import ApifyActorsTool as _ApifyActorsTool
except ImportError:
    raise ImportError(
        "Could not import langchain_apify python package. "
        "Please install it with `pip install langchain-apify` or `uv add langchain-apify`."
    )


class ApifyActorsTool(BaseTool):
    """Tool that runs Apify Actors.

        To use, you should have the environment variable `APIFY_API_TOKEN` set
        with your API key.

        For details, see https://docs.apify.com/platform/integrations/crewai

        Example:
            .. code-block:: python
                from crewai_tools import ApifyActorsTool

                tool = ApifyActorsTool(actor_id="apify/rag-web-browser")

                results = tool.run({"query": "what is Apify?", "maxResults": 5})
                print(results)
        """
    actor_tool: _ApifyActorsTool | None = Field(
        default=None, description="Apify Actor Tool"
    )

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

        actor_tool = _ApifyActorsTool(actor_name)

        kwargs.update(
            {
                "name": actor_tool.name,
                "description": actor_tool.description,
                "args_schema": actor_tool.args_schema,
            }
        )
        super().__init__(*args, **kwargs)
        self.actor_tool = actor_tool

    def _run(self, run_input: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run the Actor tool with the given input.

        Returns:
            List[Dict[str, Any]]: Results from the actor execution.

        Raises:
            ValueError: If 'actor_tool' is not initialized.
        """
        if self.actor_tool is None:
            msg = "ApifyActorsToolCrewAI is not initialized"
            raise ValueError(msg)
        return self.actor_tool._run(run_input)
