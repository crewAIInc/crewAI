from __future__ import annotations

import logging
from builtins import type as type_
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__name__)

# Dropping every capability breaks execution on its own: llm-sandbox copies the
# source file into the container, and without DAC_OVERRIDE it cannot be read.
# Everything else stays dropped -- CapEff is 0000000000000002 inside.
#
# read_only is deliberately absent: Docker rejects the code copy against a
# read-only rootfs ("container rootfs is marked read-only"), with or without a
# tmpfs on the workdir.
DEFAULT_RUNTIME_CONFIGS: dict[str, Any] = {
    "network_mode": "none",
    "mem_limit": "512m",
    "pids_limit": 128,
    "cap_drop": ["ALL"],
    "cap_add": ["DAC_OVERRIDE"],
    "security_opt": ["no-new-privileges:true"],
}


class LLMSandboxToolSchema(BaseModel):
    """Input schema for LLMSandboxTool."""

    code: str = Field(
        ...,
        description=(
            "Source to execute, complete and self-contained. Print anything you "
            "want returned -- only stdout comes back."
        ),
    )


class LLMSandboxTool(BaseTool):
    """Run agent-authored code in a self-hosted container.

    Unlike the E2B and Daytona sandbox tools, execution happens on container
    infrastructure the user already runs -- Docker, Podman or Kubernetes -- so
    there is no API key, no per-execution cost, and code never leaves their
    machines. Backed by `llm-sandbox <https://github.com/vndee/llm-sandbox>`_.

    Hardened by default: no network, capped memory and pids, every Linux
    capability dropped except DAC_OVERRIDE, and no-new-privileges set. Pass
    `runtime_configs` to change that.

    This is container isolation, not VM isolation: it inherits the threat model
    of the backend in use and makes no kernel-level guarantee. For deliberately
    adversarial code, pair it with a hardened runtime such as gVisor or Kata.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "LLM Sandbox"
    description: str = (
        "Execute code in an isolated, self-hosted container and return whatever "
        "it prints to stdout. Supports Python, JavaScript, Java, C++, Go, R and "
        "Ruby. Use this for calculations, data processing, and anything easier "
        "to compute than to reason about. Always print the result."
    )
    args_schema: type_[BaseModel] = LLMSandboxToolSchema

    package_dependencies: list[str] = Field(default_factory=lambda: ["llm-sandbox"])

    lang: str = Field(
        default="python",
        description="Language to execute: python, javascript, java, cpp, go, r or ruby.",
    )
    backend: str = Field(
        default="docker",
        description="Container backend: docker, podman, kubernetes or micromamba.",
    )
    image: str | None = Field(
        default=None,
        description=(
            "Optional custom image. The tool exposes no package-installation "
            "argument: the default network isolation would block it, and letting "
            "a model choose arbitrary PyPI packages runs setup.py at install "
            "time. Pre-bake dependencies into an image instead."
        ),
    )
    timeout: float = Field(default=30.0, description="Seconds before an execution is aborted.")
    keep_template: bool = Field(
        default=True,
        description=(
            "Keep the base image after the session closes. Setting this False "
            "re-pulls the image on every call."
        ),
    )
    runtime_configs: dict[str, Any] = Field(
        default_factory=lambda: dict(DEFAULT_RUNTIME_CONFIGS),
        description="Container settings passed to the backend. Defaults to a hardened set.",
    )
    verbose_session: bool = Field(default=False, description="Emit llm-sandbox session logs.")

    def _session_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "lang": self.lang,
            "backend": self.backend,
            "verbose": self.verbose_session,
            "keep_template": self.keep_template,
            "runtime_configs": self.runtime_configs,
        }
        if self.image is not None:
            kwargs["image"] = self.image
        return kwargs

    def _run(self, code: str) -> str:
        try:
            from llm_sandbox import SandboxSession
        except ImportError as exc:
            msg = (
                "llm-sandbox is not installed. Install it with "
                "`uv add llm-sandbox[docker]` or `pip install 'llm-sandbox[docker]'`."
            )
            raise ImportError(msg) from exc

        from llm_sandbox.exceptions import SandboxError

        try:
            with SandboxSession(**self._session_kwargs()) as session:
                result = session.run(code, timeout=self.timeout)
                exit_code, stdout, stderr = (
                    result.exit_code,
                    result.stdout,
                    result.stderr,
                )
        except SandboxError:
            # Logged rather than returned: the message can carry the DOCKER_HOST
            # socket path, which is host reconnaissance for a model that may be
            # acting on injected input.
            logger.exception("llm-sandbox execution failed")
            return "sandbox error: execution environment unavailable"

        if exit_code != 0:
            return f"exit {exit_code}\n{stderr or stdout}".strip()
        return stdout.strip() or "(no output)"
