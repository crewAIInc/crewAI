"""CrewAI tool for Skim — the x402-native clean reader API for AI agents.

Skim (https://skim402.com) turns any URL into clean, agent-ready Markdown plus
structured metadata. Reads are paid per call over the x402 protocol ($0.002 in
USDC on Base) using a wallet you control — no API keys, no signup.
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
    "metadata (title, byline, published date, language, excerpt). Pays $0.002 per "
    "call in USDC on Base over the x402 protocol — no API keys, no signup. Use this "
    "whenever you need to read web content: articles, docs, blog posts, GitHub "
    "READMEs, research papers, and similar pages."
)


class SkimReaderToolSchema(BaseModel):
    """Input schema for :class:`SkimReaderTool`."""

    url: str = Field(
        description="The fully-qualified URL to fetch and clean (https://...)."
    )


class SkimReaderTool(BaseTool):
    """Read any URL as clean Markdown via Skim, paying per call over x402.

    The tool lazily builds a payment-aware HTTP session the first time it runs,
    using your Base wallet's private key to sign USDC authorizations on demand.
    The key is used only to sign locally and never leaves your machine.

    Args:
        private_key (SecretStr): Hex private key (with or without ``0x``) for the
            Base wallet that pays for reads. Falls back to the
            ``SKIM_WALLET_PRIVATE_KEY`` environment variable. Use a dedicated
            wallet, never your personal one.
        base_url (str): Skim API base URL. Defaults to ``https://skim402.com``.
        max_price_usd (float): Hard per-call price cap in USD. The wallet refuses
            to sign for anything above this. Defaults to ``0.01`` (Skim is
            ``$0.002``).
        include_metadata (bool): When ``True`` (default), prepend a YAML
            frontmatter block of the page metadata to the returned Markdown.
        timeout (float): Per-request timeout in seconds. Defaults to ``60``.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, frozen=False
    )
    name: str = "Skim web reader"
    description: str = _TOOL_DESCRIPTION
    args_schema: type[BaseModel] = SkimReaderToolSchema

    private_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    base_url: str = DEFAULT_BASE_URL
    max_price_usd: float = 0.01
    include_metadata: bool = True
    timeout: float = 60.0

    package_dependencies: list[str] = Field(
        default_factory=lambda: ["x402", "eth-account", "requests"]
    )
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SKIM_WALLET_PRIVATE_KEY",
                description=(
                    "Hex private key for the Base wallet that pays for Skim reads. "
                    "Used only to sign x402 payment authorizations locally."
                ),
                required=False,
            ),
        ]
    )

    _session: Any = PrivateAttr(default=None)

    def _get_session(self) -> Any:
        """Build (and cache) a requests Session that auto-pays 402 responses."""
        if self._session is not None:
            return self._session

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
                "SkimReaderTool needs the x402 client with EVM support. Install it "
                "with:  pip install 'x402[evm]' requests eth-account"
            ) from exc

        key = (
            self.private_key.get_secret_value()
            if self.private_key is not None
            else os.environ.get("SKIM_WALLET_PRIVATE_KEY")
        )
        if not key:
            raise ValueError(
                "Skim requires payment via x402. Provide a Base wallet funded with "
                "USDC by setting the SKIM_WALLET_PRIVATE_KEY environment variable, "
                "or by passing private_key=... to SkimReaderTool(). The key never "
                "leaves your machine — it only signs payment authorizations locally."
            )

        normalized = key[2:] if key.startswith("0x") else key
        if len(normalized) != 64 or any(
            c not in "0123456789abcdefABCDEF" for c in normalized
        ):
            raise ValueError(
                "SKIM_WALLET_PRIVATE_KEY must be a 64-character hex string (with or "
                "without a 0x prefix)."
            )

        account = account_factory.from_key("0x" + normalized)
        cap_atomic = round(self.max_price_usd * 1_000_000)  # USDC has 6 decimals
        client = x402_client_sync()
        register_exact_evm_client(
            client,
            eth_account_signer(account),
            policies=[max_amount(cap_atomic)],
        )
        self._session = wrap_with_payment(requests.Session(), client)
        return self._session

    def _run(self, url: str) -> str:
        url = validate_url(url)
        session = self._get_session()
        endpoint = self.base_url.rstrip("/") + "/api/v1/read"

        try:
            res = session.post(
                endpoint,
                json={"url": url, "mode": "basic"},
                timeout=self.timeout,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Skim request failed: {exc}. Common causes: the wallet has no USDC "
                f"on Base, or the price exceeded max_price_usd (${self.max_price_usd})."
            ) from exc

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
                "Skim returned an unexpected response shape (expected a JSON object). "
                "This usually means the request did not reach the Skim API."
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
