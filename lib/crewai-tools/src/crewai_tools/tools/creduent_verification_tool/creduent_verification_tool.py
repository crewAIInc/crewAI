import logging

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CreduentVerificationSchema(BaseModel):
    """Input schema for Creduent verification tool.

    Attributes:
        agent_uri: Target agent URI to verify, formatted as agent://<namespace>/<name>.
    """

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
    args_schema: type[BaseModel] = CreduentVerificationSchema
    package_dependencies: list[str] = Field(default_factory=lambda: ["creduent"])
    strict: bool = True

    def _run(self, agent_uri: str) -> str:
        """Execute verification of the target agent URI.

        Args:
            agent_uri: Target agent URI formatted as agent://<namespace>/<name>.

        Returns:
            String containing verification status message or error details.

        Raises:
            ValueError: If strict mode is enabled and verification fails.
        """
        try:
            from creduent.verify import verify
        except ImportError:
            missing_msg = (
                "Error: creduent package is not installed. "
                "Install it using: pip install creduent"
            )
            if self.strict:
                raise ValueError(missing_msg)
            return missing_msg

        logger.info(f"Verifying target agent identity: {agent_uri}")
        try:
            result = verify(agent_uri)
        except Exception as err:
            error_msg = f"Verification failure for {agent_uri}: {str(err)}"
            logger.error(error_msg)
            if self.strict:
                raise ValueError(error_msg) from err
            return error_msg

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

