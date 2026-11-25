"""The User-Agent the Databricks provider sends on its Databricks requests.

``crewai_oss/<version>`` is set on the provider's own requests: SDK calls via
``Config(product=..., product_version=...)`` and Databricks Unity Gateway calls
via the request header. The process-global
``databricks.sdk.useragent.with_partner/with_product`` registration is
deliberately *not* used: crewAI is a library, and those calls would change the
host application's own Databricks SDK traffic.
"""

from __future__ import annotations

from typing import Final

from crewai_core.version import get_crewai_version


#: ``<isv-name>_<product-name>`` part of the Databricks User-Agent.
USER_AGENT_PRODUCT: Final[str] = "crewai_oss"


def user_agent_version() -> str:
    """Return the crewAI version reported in the User-Agent (``"unknown"`` if absent)."""
    return get_crewai_version() or "unknown"


def user_agent() -> str:
    """Return the crewAI Databricks User-Agent, e.g. ``crewai_oss/1.15.44``."""
    return f"{USER_AGENT_PRODUCT}/{user_agent_version()}"


def with_user_agent(existing: str | None) -> str:
    """Prefix ``existing`` (an SDK's own User-Agent) with crewAI's, once.

    The client SDK's identifier (``OpenAI/Python 2.x``, ``Anthropic/Python``, ...)
    is kept as a suffix so Databricks still sees which transport made the call.
    """
    ua = user_agent()
    existing = (existing or "").strip()
    if not existing:
        return ua
    if existing.startswith(f"{USER_AGENT_PRODUCT}/"):
        return existing
    return f"{ua} {existing}"
