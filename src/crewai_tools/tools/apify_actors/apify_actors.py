from crewai.tools import BaseTool
from pydantic import Field
from typing import Any

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

    def _run(self, run_input: dict) -> list[dict]:
        if self.actor_tool is None:
            msg = "ApifyActorsToolCrewAI is not initialized"
            raise ValueError(msg)
        return self.actor_tool._run(run_input)
