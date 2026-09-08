"""CrewAI tool for Skim — the x402-native clean reader API for AI agents.

Exposes :class:`SkimReaderTool`, a CrewAI ``BaseTool`` that fetches any URL and
returns clean, agent-ready Markdown plus structured metadata. Each call is paid
automatically over the x402 protocol ($0.002 in USDC on Base) using a wallet you
control. The private key never leaves your machine — it only signs an EIP-3009
USDC authorization locally.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field, PrivateAttr, SecretStr

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
        or text[0] in " "
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
        description="The fully-qualified URL to fetch and clean (https://...).",
    )


class SkimReaderTool(BaseTool):
    """Read any URL as clean Markdown via Skim, paying per call over x402.

    The tool lazily builds a payment-aware HTTP session the first time it runs,
    using your Base wallet's private key to sign USDC authorizations on demand.

    Args:
        private_key (str): Optional. Hex private key (with or without ``0x``) for
            the Base wallet that pays for reads. Falls back to the
            ``SKIM_WALLET_PRIVATE_KEY`` environment variable. Use a dedicated
            wallet, never your personal one.
        base_url (str): Optional. Skim API base URL. Defaults to
            ``https://skim402.com``.
        max_price_usd (float): Optional. Hard per-call price cap in USD. The
            wallet refuses to sign for anything above this. Defaults to ``0.01``
            (Skim is ``$0.002``).
        include_metadata (bool): Optional. When ``True`` (default), prepend a YAML
            frontmatter block of the page metadata to the returned Markdown.
        timeout (float): Optional. Per-request timeout in seconds. Defaults to
            ``60``.

    Example:
        from crewai_tools import SkimReaderTool

        tool = SkimReaderTool()  # reads SKIM_WALLET_PRIVATE_KEY from the env
        tool.run(url="https://en.wikipedia.org/wiki/HTTP_402")
    """

    name: str = "Skim web reader"
    description: str = _TOOL_DESCRIPTION
    args_schema: type[BaseModel] = SkimReaderToolSchema

    base_url: str = DEFAULT_BASE_URL
    max_price_usd: float = 0.01
    include_metadata: bool = True
    timeout: float = 60.0
    private_key: Optional[SecretStr] = Field(default=None, exclude=True, repr=False)

    _session: Any = PrivateAttr(default=None)
    package_dependencies: list[str] = Field(
        default_factory=lambda: ["x402[evm]", "eth-account", "requests"]
    )
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SKIM_WALLET_PRIVATE_KEY",
                description=(
                    "Hex private key for the Base wallet that pays for Skim reads "
                    "over x402. Needed unless private_key is passed to the "
                    "constructor. The key never leaves your machine."
                ),
                required=False,
            ),
        ]
    )

    def _get_session(self) -> Any:
        """Build (and cache) a requests Session that auto-pays 402 responses."""
        if self._session is not None:
            return self._session

        try:
            import requests
            from eth_account import Account
            from x402 import x402ClientSync
            from x402.client import max_amount
            from x402.http.clients.requests import wrapRequestsWithPayment
            from x402.mechanisms.evm.exact.register import register_exact_evm_client
            from x402.mechanisms.evm.signers import EthAccountSigner
        except ImportError as exc:  # pragma: no cover - import-guard
            raise ImportError(
                "SkimReaderTool needs the x402 client with EVM support. Install it "
                'with:  pip install "x402[evm]" requests eth-account'
            ) from exc

        key = (
            self.private_key.get_secret_value()
            if self.private_key is not None
            else os.environ.get("SKIM_WALLET_PRIVATE_KEY")
        )
        if not key:
            raise ValueError(
                "Skim requires payment via x402. Provide a Base wallet funded with "
                "USDC by setting the SKIM_WALLET_PRIVATE_KEY environment variable, or "
                "by passing private_key=... to SkimReaderTool(). The key never leaves "
                "your machine — it only signs payment authorizations locally."
            )

        normalized = key[2:] if key.startswith("0x") else key
        if len(normalized) != 64 or any(
            c not in "0123456789abcdefABCDEF" for c in normalized
        ):
            raise ValueError(
                "SKIM_WALLET_PRIVATE_KEY must be a 64-character hex string (with or "
                "without a 0x prefix)."
            )

        account = Account.from_key("0x" + normalized)
        cap_atomic = int(round(self.max_price_usd * 1_000_000))  # USDC has 6 decimals
        client = x402ClientSync()
        register_exact_evm_client(
            client,
            EthAccountSigner(account),
            policies=[max_amount(cap_atomic)],
        )
        self._session = wrapRequestsWithPayment(requests.Session(), client)
        return self._session

    def _run(self, url: str) -> str:
        session = self._get_session()
        endpoint = self.base_url.rstrip("/") + "/api/v1/read"

        try:
            res = session.post(
                endpoint,
                json={"url": url, "mode": "basic"},
                timeout=self.timeout,
            )
        except Exception as exc:  # network / payment-signing failure
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

        markdown = data.get("markdown") or data.get("text") or ""

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
