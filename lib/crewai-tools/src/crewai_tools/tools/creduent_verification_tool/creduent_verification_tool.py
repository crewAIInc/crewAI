import logging
from typing import Any, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CreduentVerificationSchema(BaseModel):
    """Input schema for Creduent verification tool."""

    agent_uri: str = Field(
        ...,
        description="Target agent URI to verify, formatted as agent://<namespace>/<name>",
    )


class CreduentVerificationTool(BaseTool):
    """Tool for verifying agent identity using the Creduent protocol.

    Performs local cryptographic signature verification (Ed25519) and canonical JCS RFC 8785
    attestation checks on target agent URIs.
    """

    name: str = "Creduent Agent Identity Verification"
    description: str = (
        "Verifies the cryptographic identity, signature, and attestations of a target AI agent "
        "using the Creduent open protocol before delegating tasks."
    )
    args_schema: Type[BaseModel] = CreduentVerificationSchema
    strict: bool = True

    def _run(self, agent_uri: str) -> str:
        """Execute verification of the target agent URI."""
        try:
            from creduent.verify import verify
        except ImportError:
            return (
                "Error: creduent package is not installed. "
                "Install it using: pip install creduent"
            )

        logger.info(f"Verifying target agent identity: {agent_uri}")
        try:
            result = verify(agent_uri)
            if result.valid:
                return (
                    f"Verification SUCCESS for {agent_uri}. "
                    "Agent identity and cryptographic attestations are trusted."
                )
            error_msg = f"Verification FAILED for {agent_uri}: {result.error}"
            logger.warning(error_msg)
            if self.strict:
                raise ValueError(error_msg)
            return error_msg
        except Exception as err:
            error_msg = f"Verification failure for {agent_uri}: {str(err)}"
            logger.error(error_msg)
            if self.strict:
                raise ValueError(error_msg) from err
            return error_msg
