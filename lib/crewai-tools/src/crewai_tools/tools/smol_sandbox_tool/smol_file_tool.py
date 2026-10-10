from __future__ import annotations

import base64
import binascii
from builtins import type as type_
from typing import Literal

from pydantic import BaseModel, Field

from crewai_tools.tools.smol_sandbox_tool.smol_base_tool import SmolBaseTool


class SmolFileToolSchema(BaseModel):
    action: Literal["read", "write"] = Field(
        ..., description="Read or write a VM file."
    )
    path: str = Field(..., description="Absolute path inside the VM.")
    content: str | None = Field(
        default=None,
        description="Text to write, or base64 data when binary=True.",
    )
    binary: bool = False


class SmolFileTool(SmolBaseTool):
    """Read and write VM files through the Smol Machines file API."""

    name: str = "Smol Machines Sandbox Files"
    description: str = "Read or write files inside an isolated local or cloud VM."
    args_schema: type_[BaseModel] = SmolFileToolSchema

    def _run(
        self,
        action: Literal["read", "write"],
        path: str,
        content: str | None = None,
        binary: bool = False,
    ) -> str:
        if not path.startswith("/"):
            raise ValueError("path must be absolute inside the VM")
        if action == "write" and content is None:
            raise ValueError("write requires content")
        if action == "read" and content is not None:
            raise ValueError("read does not accept content")
        if content is not None and action == "write" and not isinstance(content, str):
            raise ValueError("content must be a string")
        data: bytes | None = None
        if action == "write" and content is not None:
            if binary:
                try:
                    data = base64.b64decode(content, validate=True)
                except (binascii.Error, ValueError) as exc:
                    raise ValueError("binary content must be valid base64") from exc
            else:
                data = content.encode()
        with self._machine_session() as machine:
            if action == "read":
                file_data = machine.read_file(path)
                if binary:
                    return base64.b64encode(file_data).decode("ascii")
                try:
                    return file_data.decode("utf-8")
                except UnicodeDecodeError:
                    return "File is not valid UTF-8; read it with binary=True for base64 content."
            if data is None:
                raise ValueError("write requires content")
            machine.write_file(path, data)
            return f"Wrote {len(data)} bytes to {path}"
