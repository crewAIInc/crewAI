import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field


class AgentTollSafetyToolInput(BaseModel):
    """Input for AgentTollSafetyTool."""

    address: str = Field(
        ...,
        description="Base (chain id 8453) token contract address to check, e.g. '0x1234...'",
    )


class AgentTollSafetyTool(BaseTool):
    """
    AgentTollSafetyTool - checks whether a Base token is a honeypot or rug before you trade it.

    Calls AgentToll's /api/base/safety endpoint (https://agenttoll.app): a simulated buy
    AND sell, taxes, owner privileges, holder concentration, liquidity risk, and the
    deployer's own history. Paid per call in USDC on Base via the x402 protocol (HTTP 402)
    -- no API key, no subscription, no signup. Costs $0.003, charged only once the
    response comes back successfully.

    Dependencies:
        - x402[requests,evm]
    """

    name: str = "Check Base token safety"
    description: str = (
        "Check whether a Base token contract is a honeypot or rug: simulated buy and "
        "sell, taxes, owner privileges, holder concentration, liquidity risk, and the "
        "deployer's history. Use before trading or recommending an unfamiliar Base token."
    )
    args_schema: type[BaseModel] = AgentTollSafetyToolInput
    base_url: str = "https://agenttoll.app"

    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="EVM_PRIVATE_KEY",
                description=(
                    "Private key of a Base wallet holding a little USDC, used to pay "
                    "$0.003 per call via x402"
                ),
                required=True,
            ),
        ]
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["x402"])

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        try:
            import x402  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Missing optional dependency 'x402'. Install with: \n"
                "  uv add crewai-tools --extra x402\n"
                "or\n"
                "  pip install 'x402[requests,evm]'\n"
            ) from exc

        if "EVM_PRIVATE_KEY" not in os.environ:
            raise ValueError(
                "Environment variable EVM_PRIVATE_KEY is required for AgentTollSafetyTool"
            )

    def _paid_session(self) -> Any:
        """A requests session that pays x402 quotes automatically, capped per call."""
        from eth_account import Account
        from x402 import x402ClientSync
        from x402.http.clients import x402_requests
        from x402.mechanisms.evm import EthAccountSigner
        from x402.mechanisms.evm.exact.register import register_exact_evm_client

        client = x402ClientSync().set_spend_controls(
            {"max_amount_per_payment": "$0.05"}
        )
        account = Account.from_key(os.environ["EVM_PRIVATE_KEY"])
        register_exact_evm_client(client, EthAccountSigner(account))
        return x402_requests(client)

    def _run(self, address: str) -> str:
        try:
            with self._paid_session() as session:
                response = session.get(f"{self.base_url}/api/base/safety/{address}")
                response.raise_for_status()
                return str(response.text)
        except Exception as e:
            return f"Error checking token safety: {e}"

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        return self._run(*args, **kwargs)
