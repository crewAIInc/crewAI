import asyncio
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import requests


class ContextualAIQuerySchema(BaseModel):
    """Schema for contextual query tool."""

    query: str = Field(..., description="Query to send to the Contextual AI agent.")
    agent_id: str = Field(..., description="ID of the Contextual AI agent to query")
    datastore_id: str | None = Field(
        None, description="Optional datastore ID for document readiness verification"
    )


class ContextualAIQueryTool(BaseTool):
    """Tool to query Contextual AI RAG agents."""

    name: str = "Contextual AI Query Tool"
    description: str = (
        "Use this tool to query a Contextual AI RAG agent with access to your documents"
    )
    args_schema: type[BaseModel] = ContextualAIQuerySchema

    api_key: str
    contextual_client: Any = None
    package_dependencies: list[str] = Field(
        default_factory=lambda: ["contextual-client"]
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            from contextual import ContextualAI

            self.contextual_client = ContextualAI(api_key=self.api_key)
        except ImportError as e:
            raise ImportError(
                "contextual-client package is required. Install it with: pip install contextual-client"
            ) from e

    def _check_documents_ready(self, datastore_id: str) -> bool:
        """Synchronous check if all documents are ready."""
        url = f"https://api.contextual.ai/v1/datastores/{datastore_id}/documents"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            documents = data.get("documents", [])
            return not any(
                doc.get("status") in ("processing", "pending") for doc in documents
            )
        return True

    async def _wait_for_documents_async(
        self, datastore_id: str, max_attempts: int = 20, interval: float = 30.0
    ) -> bool:
        """Asynchronously poll until documents are ready, exiting early if possible."""
        for _attempt in range(max_attempts):
            ready = await asyncio.to_thread(self._check_documents_ready, datastore_id)
            if ready:
                return True
            await asyncio.sleep(interval)
        return True  # give up but don't fail hard

    def _run(self, query: str, agent_id: str, datastore_id: str | None = None) -> str:
        if not agent_id:
            raise ValueError("Agent ID is required to query the Contextual AI agent")

        if datastore_id:
            ready = self._check_documents_ready(datastore_id)
            if not ready:
                try:
                    # If no running event loop, use asyncio.run
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    # Already inside an event loop
                    try:
                        import nest_asyncio  # type: ignore[import-untyped]

                        nest_asyncio.apply(loop)
                        loop.run_until_complete(
                            self._wait_for_documents_async(datastore_id)
                        )
                    except Exception:  # noqa: S110
                        pass
                else:
                    asyncio.run(self._wait_for_documents_async(datastore_id))
        else:
            pass

        try:
            response = self.contextual_client.agents.query.create(
                agent_id=agent_id, messages=[{"role": "user", "content": query}]
            )
            if hasattr(response, "content"):
                return response.content
            if hasattr(response, "message"):
                return (
                    response.message.content
                    if hasattr(response.message, "content")
                    else str(response.message)
                )
            if hasattr(response, "messages") and len(response.messages) > 0:
                last_message = response.messages[-1]
                return (
                    last_message.content
                    if hasattr(last_message, "content")
                    else str(last_message)
                )
            return str(response)
        except Exception as e:
            return f"Error querying Contextual AI agent: {e!s}"
