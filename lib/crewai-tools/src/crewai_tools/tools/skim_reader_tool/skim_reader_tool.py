"""CrewAI tool for Skim — the clean reader API for AI agents.

Skim (https://skim402.com) turns any URL into clean, agent-ready Markdown plus
structured metadata. The recommended payment path is a card-plan API key
(SKIM_API_KEY); x402 wallet pay-per-call remains available as an option.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr
import requests

from crewai_tools.security.safe_path import validate_url


DEFAULT_BASE_URL = "https://skim402.com"


def _yaml_scalar(value: Any) -> str:
    """Render a metadata value as a safe single-line YAML scalar.

    Collapses internal whitespace/newlines and double-quotes the value when it
    contains characters that could otherwise produce invalid or ambiguous YAML.
    """
    text = " ".join(str(value).split())
    needs_quoting = (
        text == ""
        or text[0] in "!&*?|>%@`\"'#,[]{}:-"
        or ": " in text
        or text.endswith(":")
        or text[0] == " "
    )
    if needs_quoting:
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return text


_TOOL_DESCRIPTION = (
    "Fetch any URL and return clean, agent-ready Markdown via Skim (skim402.com). "
    "Strips nav, ads, and boilerplate; preserves the article body plus structured "
    "metadata (title, byline, published date, language, excerpt). A card-plan API "
    "key is the recommended setup; wallet/x402 payment is optional. Use this "
    "whenever you need to read web content: articles, docs, blog posts, GitHub "
    "READMEs, research papers, and similar pages."
)


class SkimReaderToolSchema(BaseModel):
    """Input schema for :class:`SkimReaderTool`."""

    url: str = Field(
        description="The fully-qualified URL to fetch and clean (https://...)."
    )


class SkimReaderTool(BaseTool):
    """Read any URL as clean Markdown via Skim.

    Card-plan authentication is recommended: set SKIM_API_KEY or pass api_key.
    Get a free-plan key at https://skim402.com/pricing; no crypto is required.

    Wallet payment is optional: set SKIM_WALLET_PRIVATE_KEY or pass private_key.
    That path pays per call in USDC on Base over x402. If both credentials are
    configured, the card-plan API key takes priority.

    Args:
        api_key (SecretStr): Card-plan API key. Falls back to SKIM_API_KEY.
        private_key (SecretStr): Wallet key for optional x402 payment. Falls back
            to SKIM_WALLET_PRIVATE_KEY and is ignored when api_key is set.
        base_url (str): Skim API base URL. Defaults to https://skim402.com.
        max_price_usd (float): Wallet lane only. Hard per-call price cap in USD.
        include_metadata (bool): Prepend page metadata as YAML frontmatter.
        timeout (float): Per-request timeout in seconds. Defaults to 60.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, frozen=False
    )
    name: str = "Skim web reader"
    description: str = _TOOL_DESCRIPTION
    args_schema: type[BaseModel] = SkimReaderToolSchema

    api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    private_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    base_url: str = DEFAULT_BASE_URL
    max_price_usd: float = 0.01
    include_metadata: bool = True
    timeout: float = 60.0

    package_dependencies: list[str] = Field(default_factory=lambda: ["requests"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SKIM_API_KEY",
                description=(
                    "Recommended. Card-plan API key (sk402_...). Get a free-plan "
                    "key at https://skim402.com/pricing; no crypto is required."
                ),
                required=False,
            ),
            EnvVar(
                name="SKIM_WALLET_PRIVATE_KEY",
                description=(
                    "Optional. Base wallet key for x402 pay-per-call when no "
                    "SKIM_API_KEY is configured."
                ),
                required=False,
            ),
        ]
    )

    _session: Any = PrivateAttr(default=None)
    _card_lane: bool = PrivateAttr(default=False)

    def _get_session(self) -> Any:
        """Build and cache an HTTP session for card-key or wallet reads."""
        if self._session is not None:
            return self._session

        api_key = (
            self.api_key.get_secret_value()
            if self.api_key is not None
            else os.environ.get("SKIM_API_KEY")
        )
        if api_key:
            session = requests.Session()
            session.headers["Authorization"] = f"Bearer {api_key}"
            self._session = session
            self._card_lane = True
            return self._session

        key = (
            self.private_key.get_secret_value()
            if self.private_key is not None
            else os.environ.get("SKIM_WALLET_PRIVATE_KEY")
        )
        if not key:
            raise ValueError(
                "Skim needs a payment method. Set SKIM_API_KEY (recommended card "
                "plan; get a free-plan key at https://skim402.com/pricing) or "
                "SKIM_WALLET_PRIVATE_KEY (optional x402 wallet payment)."
            )

        normalized = key[2:] if key.startswith("0x") else key
        if len(normalized) != 64 or any(
            c not in "0123456789abcdefABCDEF" for c in normalized
        ):
            raise ValueError(
                "SKIM_WALLET_PRIVATE_KEY must be a 64-character hex string (with or "
                "without a 0x prefix)."
            )

        try:
            account_factory = importlib.import_module("eth_account").Account
            x402_client_sync = importlib.import_module("x402").x402ClientSync
            max_amount = importlib.import_module("x402.client").max_amount
            wrap_with_payment = importlib.import_module(
                "x402.http.clients.requests"
            ).wrapRequestsWithPayment
            register_exact_evm_client = importlib.import_module(
                "x402.mechanisms.evm.exact.register"
            ).register_exact_evm_client
            eth_account_signer = importlib.import_module(
                "x402.mechanisms.evm.signers"
            ).EthAccountSigner
        except ImportError as exc:
            raise ImportError(
                "Wallet payment needs the x402 client with EVM support. Install it "
                "with: pip install 'crewai-tools[x402]'. Card-key users do not need "
                "this optional extra."
            ) from exc

        account = account_factory.from_key("0x" + normalized)
        cap_atomic = round(self.max_price_usd * 1_000_000)
        client = x402_client_sync()
        register_exact_evm_client(
            client,
            eth_account_signer(account),
            policies=[max_amount(cap_atomic)],
        )
        self._session = wrap_with_payment(requests.Session(), client)
        self._card_lane = False
        return self._session

    def _run(self, url: str) -> str:
        """Fetch one validated URL and return clean Markdown."""
        url = validate_url(url)
        session = self._get_session()
        path = "/api/t/read" if self._card_lane else "/api/v1/read"
        endpoint = self.base_url.rstrip("/") + path

        try:
            if self._card_lane:
                res = session.get(endpoint, params={"url": url}, timeout=self.timeout)
            else:
                res = session.post(
                    endpoint,
                    json={"url": url, "mode": "basic"},
                    timeout=self.timeout,
                )
        except Exception as exc:
            hint = (
                "Check that your SKIM_API_KEY is valid and active."
                if self._card_lane
                else (
                    "Common causes: the wallet has no USDC on Base, or the price "
                    f"exceeded max_price_usd ({self.max_price_usd} USD)."
                )
            )
            raise RuntimeError(f"Skim request failed: {exc}. {hint}") from exc

        if not getattr(res, "ok", res.status_code < 400):
            body = (res.text or "").strip()
            raise RuntimeError(
                f"Skim returned {res.status_code} {getattr(res, 'reason', '')}: "
                f"{body or '(no body)'}"
            )

        try:
            data = res.json()
        except ValueError as exc:
            raise RuntimeError(
                "Skim returned a non-JSON response. This usually means the request "
                f"did not reach the Skim API. Underlying error: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise RuntimeError(
                "Skim returned an unexpected response shape (expected a JSON object)."
            )

        markdown: str = data.get("markdown") or data.get("text") or ""
        metadata = data.get("metadata")
        if self.include_metadata and isinstance(metadata, dict):
            meta_lines = [
                f"{k}: {_yaml_scalar(v)}"
                for k, v in metadata.items()
                if v is not None and v != ""
            ]
            if meta_lines:
                markdown = "---\n" + "\n".join(meta_lines) + "\n---\n\n" + markdown

        return markdown
