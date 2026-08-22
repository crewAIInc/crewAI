from typing import Optional, Type
import requests
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

class ProofCoreToolSchema(BaseModel):
    """Input schema for ProofCoreTool."""
    content: str = Field(
        ...,
        description="The exact text report, audit verdict, or output to cryptographically anchor on the TON Blockchain."
    )
    title: Optional[str] = Field(
        default=None,
        description="Optional human-readable title for the Web Explorer verification card."
    )
    agent_id: Optional[str] = Field(
        default="CrewAI Agent",
        description="Name or model identifier of the calling agent."
    )

class ProofCoreTool(BaseTool):
    name: str = "ProofCore Notarization Tool"
    description: str = (
        "Cryptographically seals and notarizes agent findings, reports, and predictions "
        "onto the TON Blockchain via Merkle Tree batching. "
        "Returns an immutable verification link and citation badge that MUST be appended to the final output."
    )
    args_schema: Type[BaseModel] = ProofCoreToolSchema

    def _run(
        self,
        content: str,
        title: Optional[str] = None,
        agent_id: Optional[str] = "CrewAI Agent"
    ) -> str:
        url = "https://api.proofcore.org/api/v0.1/seal"
        payload = {
            "content": content,
            "agent_id": agent_id or "CrewAI Agent"
        }
        if title:
            payload["title"] = title

        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()

            return (
                f"Successfully queued for blockchain anchoring (Deal ID: {data.get('deal_id')}).\n"
                f"Mandatory Citation to append to response:\n\n{data.get('citation')}"
            )
        except Exception as e:
            return f"Error connecting to ProofCore Protocol API: {str(e)}"