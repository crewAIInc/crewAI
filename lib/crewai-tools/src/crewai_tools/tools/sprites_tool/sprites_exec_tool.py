from __future__ import annotations

import asyncio
import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from crewai.types.callback import SerializableCallable
from pydantic import BaseModel, Field, SecretStr, field_validator


def _no_cache(_args: Any = None, _result: Any = None) -> bool:
    """Commands can change persistent state and must not reuse cached results."""
    return False


class SpritesExecToolSchema(BaseModel):
    command: str = Field(
        ...,
        min_length=1,
        description="Shell command to execute in the configured Sprite.",
    )
    cwd: str | None = Field(
        default=None, description="Working directory inside the Sprite."
    )

    @field_validator("command", "cwd")
    @classmethod
    def validate_command_argument(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or "\x00" in value):
            raise ValueError(
                "Command and working directory must be nonblank and contain no NUL bytes."
            )
        return value


class SpritesExecTool(BaseTool):
    """Run shell commands in an existing, caller-managed Fly.io Sprite.

    This tool does not create or delete Sprites. Each call opens a new shell;
    files persist, but shell variables and changes of directory do not.
    """

    name: str = "Fly.io Sprites Exec"
    description: str = (
        "Execute a shell command in a configured, persistent Fly.io Sprite and "
        "return its exit code, stdout, and stderr. Use this to run code, inspect "
        "files, or install packages in the remote environment. Commands can "
        "modify or delete files; only execute commands appropriate for the task."
    )
    args_schema: type[BaseModel] = SpritesExecToolSchema
    package_dependencies: list[str] = Field(default_factory=lambda: ["sprites-py"])
    sprite_name: str = Field(
        ...,
        min_length=1,
        description="Name of an existing Sprite accessible to the token.",
    )
    api_key: SecretStr | None = Field(
        default_factory=lambda: (
            SecretStr(value) if (value := os.getenv("SPRITE_TOKEN")) else None
        ),
        exclude=True,
        repr=False,
        description="Sprites API token. Defaults to the SPRITE_TOKEN environment variable.",
    )
    timeout: float = Field(
        default=60, gt=0, le=300, description="Maximum seconds to wait for a command."
    )
    max_output_chars: int = Field(
        default=20_000,
        gt=0,
        description="Maximum characters returned per output stream.",
    )
    cache_function: SerializableCallable = Field(default=_no_cache)
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(name="SPRITE_TOKEN", description="Fly.io Sprites API token")
        ]
    )

    @field_validator("sprite_name")
    @classmethod
    def validate_sprite_name(cls, value: str) -> str:
        # The SDK interpolates this value into API paths. Require a single name.
        if value in {".", ".."} or any(
            char.isspace() or char in "/\\?#%" or ord(char) < 32 or ord(char) == 127
            for char in value
        ):
            raise ValueError("sprite_name must be a Sprite name, not a URL or path.")
        return value

    def _run(self, command: str, cwd: str | None = None) -> dict[str, str | int | bool]:
        inputs = SpritesExecToolSchema(command=command, cwd=cwd)
        if self.api_key is None or not self.api_key.get_secret_value().strip():
            raise ValueError("Set SPRITE_TOKEN or pass api_key to SpritesExecTool.")
        try:
            from sprites import SpritesClient
            from sprites.exceptions import TimeoutError as SpritesTimeoutError
        except ImportError:
            raise ImportError(
                'Install Fly.io Sprites support with: uv add "crewai-tools[sprites]"'
            ) from None

        try:
            with SpritesClient(token=self.api_key.get_secret_value()) as client:
                result = client.sprite(self.sprite_name).run(
                    "bash",
                    "-lc",
                    inputs.command,
                    cwd=inputs.cwd,
                    capture_output=True,
                    timeout=self.timeout,
                    check=False,
                )
        except (TimeoutError, SpritesTimeoutError):
            raise TimeoutError(
                "Timed out waiting for the Sprite command. It may still be running; "
                "inspect the Sprite before retrying commands with side effects."
            ) from None
        except Exception:
            # SDK errors can include request details. Do not surface credentials
            # or suggest retrying a command whose remote outcome is unknown.
            raise RuntimeError(
                "Fly.io Sprites execution failed. Check the configured Sprite, "
                "credentials, and connectivity. The command outcome may be unknown; "
                "inspect the Sprite before retrying commands with side effects."
            ) from None

        stdout = (result.stdout or b"").decode("utf-8", errors="replace")
        stderr = (result.stderr or b"").decode("utf-8", errors="replace")
        return {
            "exit_code": result.returncode,
            "stdout": stdout[: self.max_output_chars],
            "stderr": stderr[: self.max_output_chars],
            "stdout_truncated": len(stdout) > self.max_output_chars,
            "stderr_truncated": len(stderr) > self.max_output_chars,
        }

    async def _arun(
        self, command: str, cwd: str | None = None
    ) -> dict[str, str | int | bool]:
        """Keep blocking SDK execution off the crew's event loop."""
        return await asyncio.to_thread(self._run, command, cwd)
