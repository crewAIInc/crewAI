"""Tool that checks a counterparty's identity/reputation before a payment."""

from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field


class AvelVerifyToolSchema(BaseModel):
    """Input for AvelVerifyTool."""

    payee_id: str = Field(
        ...,
        description="Identifier of the agent or merchant about to receive the payment.",
    )
    amount: float = Field(..., description="Transaction amount, in `currency` units.")
    currency: str = Field(..., description="Currency of `amount`, e.g. 'USDC' or 'EUR'.")
    nonce: str = Field(
        default="",
        description="Optional nonce binding this check to one specific transaction.",
    )


class AvelVerifyTool(BaseTool):
    """Checks whether the counterparty of an about-to-happen payment is a
    real, identified agent with a reasonable reputation — complementary to
    payment-execution tools (e.g. batch payment or escrow tools), not a
    replacement for them: this tool never touches the payment itself,
    which stays on whatever rail the agent already uses (x402, card,
    bank transfer, etc.).

    Backed by AVEL (https://github.com/Ensi81/Avel), a verification
    layer that scores agents on reputation derived from real recorded
    transaction outcomes, never a self-reported value. One tool instance
    represents one payer identity, bound to a private key at
    construction — never exposed as an argument the model can see or
    set, so it never ends up in conversation logs or traces.
    """

    name: str = "AVEL Verify"
    description: str = (
        "Check whether the agent or merchant about to receive a payment is a "
        "real, identified party with a reasonable reputation, before the "
        "payment goes out. Returns APPROVED or DENIED plus the reasons and "
        "both parties' reputation. Does not move money — the payment still "
        "happens through whatever rail is normally used."
    )
    args_schema: type[BaseModel] = AvelVerifyToolSchema
    package_dependencies: list[str] = Field(default_factory=lambda: ["agent-interchange"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="AVEL_PRIVATE_KEY",
                description="EVM private key of the paying agent's identity — signs each check, never seen by the model.",
                required=True,
            ),
            EnvVar(
                name="AVEL_BASE_URL",
                description="Base URL of the AVEL instance to verify against.",
                required=False,
            ),
        ]
    )
    client: Any = None
    payer_address: str | None = None

    def __init__(
        self,
        private_key: str | None = None,
        base_url: str = "https://aisrail.fly.dev",
        **kwargs: Any,
    ) -> None:
        """Initialize AvelVerifyTool for one payer identity.

        Args:
            private_key: EVM private key signing every check. Falls back to
                the AVEL_PRIVATE_KEY environment variable.
            base_url: Base URL of the AVEL instance to verify against.
            **kwargs: Additional arguments passed to BaseTool.

        Raises:
            ImportError: If the `agent-interchange` package is not installed.
            ValueError: If no private key is available from either source.
        """
        super().__init__(**kwargs)

        try:
            from agent_interchange import InterchangeClient
        except ImportError:
            raise ImportError(
                "`agent-interchange` package not found, please run `uv add agent-interchange`"
            ) from None

        import os

        key = private_key or os.environ.get("AVEL_PRIVATE_KEY")
        if not key:
            raise ValueError(
                "AvelVerifyTool requires a private_key argument or AVEL_PRIVATE_KEY "
                "env var — every check is signed to prove who is asking."
            )

        from eth_account import Account

        self.payer_address = Account.from_key(key).address
        self._private_key = key
        self.client = InterchangeClient(base_url)

    def _run(self, payee_id: str, amount: float, currency: str, nonce: str = "") -> str:
        """Check the counterparty's identity/reputation before paying them.

        Args:
            payee_id: Identifier of the agent or merchant about to be paid.
            amount: Transaction amount, in `currency` units.
            currency: Currency of `amount`.
            nonce: Optional nonce binding this check to one transaction.

        Returns:
            A summary of the verdict, reasons, and both parties' reputation.
        """
        result = self.client.verify(
            payer_address=self.payer_address,
            payee_id=payee_id,
            amount=amount,
            currency=currency,
            private_key=self._private_key,
            nonce=nonce,
        )

        if result.get("verdict_withheld"):
            return (
                f"Verdict withheld — verification fee not yet paid: "
                f"{result.get('message')}"
            )

        verdict = result.get("verdict")
        reasons = "; ".join(result.get("reasons") or [])
        payer_rep = result.get("payer_reputation") or {}
        payee_rep = result.get("payee_reputation")
        summary = f"{verdict}. Reasons: {reasons}. Payer reputation: {payer_rep}."
        if payee_rep:
            summary += f" Payee reputation: {payee_rep}."
        return summary
