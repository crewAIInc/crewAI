from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class ContextualAICreateAgentSchema(BaseModel):
    """Schema for contextual create agent tool."""

    agent_name: str = Field(..., description="Name for the new agent")
    agent_description: str = Field(..., description="Description for the new agent")
    datastore_name: str = Field(..., description="Name for the new datastore")
    document_paths: list[str] = Field(..., description="List of file paths to upload")


class ContextualAICreateAgentTool(BaseTool):
    """Tool to create Contextual AI RAG agents with documents."""

    name: str = "Contextual AI Create Agent Tool"
    description: str = (
        "Create a new Contextual AI RAG agent with documents and datastore"
    )
    args_schema: type[BaseModel] = ContextualAICreateAgentSchema

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

    def _run(
        self,
        agent_name: str,
        agent_description: str,
        datastore_name: str,
        document_paths: list[str],
    ) -> str:
        """Create a complete RAG pipeline with documents."""
        try:
            import os

            # Create datastore
            datastore = self.contextual_client.datastores.create(name=datastore_name)
            datastore_id = datastore.id

            # Upload documents
            document_ids = []
            for doc_path in document_paths:
                if not os.path.exists(doc_path):
                    raise FileNotFoundError(f"Document not found: {doc_path}")

                with open(doc_path, "rb") as f:
                    ingestion_result = (
                        self.contextual_client.datastores.documents.ingest(
                            datastore_id, file=f
                        )
                    )
                    document_ids.append(ingestion_result.id)

            # Create agent
            agent = self.contextual_client.agents.create(
                name=agent_name,
                description=agent_description,
                datastore_ids=[datastore_id],
            )

            return f"Successfully created agent '{agent_name}' with ID: {agent.id} and datastore ID: {datastore_id}. Uploaded {len(document_ids)} documents."

        except Exception as e:
            return f"Failed to create agent with documents: {e!s}"
