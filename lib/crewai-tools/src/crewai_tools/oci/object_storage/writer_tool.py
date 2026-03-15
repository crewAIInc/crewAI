from __future__ import annotations

import os
from typing import Any, cast

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crewai_tools.oci.common import (
    create_oci_client_kwargs,
    get_oci_module,
    parse_object_storage_path,
)


class OCIObjectStorageWriterToolInput(BaseModel):
    """Input schema for OCIObjectStorageWriterTool."""

    file_path: str = Field(
        ...,
        description=(
            "OCI Object Storage path in the form "
            "`oci://bucket/path` or `oci://namespace@bucket/path`."
        ),
    )
    content: str = Field(..., description="Content to write to the object")


class OCIObjectStorageWriterTool(BaseTool):
    name: str = "OCI Object Storage Writer Tool"
    description: str = "Writes a text file to Oracle Cloud Infrastructure Object Storage."
    args_schema: type[BaseModel] = OCIObjectStorageWriterToolInput
    package_dependencies: list[str] = Field(default_factory=lambda: ["oci"])
    namespace_name: str | None = None
    client: Any | None = None

    def __init__(
        self,
        namespace_name: str | None = None,
        *,
        auth_type: str = "API_KEY",
        auth_profile: str | None = None,
        auth_file_location: str | None = None,
        client: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.namespace_name = namespace_name or os.getenv("OCI_OBJECT_STORAGE_NAMESPACE")
        self.client = client

        if self.client is None:
            oci = get_oci_module()
            resolved_auth_profile = cast(
                str, auth_profile or os.getenv("OCI_AUTH_PROFILE", "DEFAULT")
            )
            resolved_auth_file_location = cast(
                str,
                auth_file_location or os.getenv("OCI_AUTH_FILE_LOCATION", "~/.oci/config"),
            )
            client_kwargs = create_oci_client_kwargs(
                auth_type=auth_type,
                auth_profile=resolved_auth_profile,
                auth_file_location=resolved_auth_file_location,
            )
            self.client = oci.object_storage.ObjectStorageClient(**client_kwargs)

    def _require_client(self) -> Any:
        if self.client is None:
            raise ValueError("OCI Object Storage client is not initialized.")
        return self.client

    def _resolve_namespace(self, path_namespace: str | None) -> str:
        if path_namespace:
            return path_namespace
        if self.namespace_name:
            return self.namespace_name
        return str(self._require_client().get_namespace().data)

    def _run(self, file_path: str, content: str) -> str:
        try:
            path_namespace, bucket_name, object_name = parse_object_storage_path(
                file_path
            )
            namespace_name = self._resolve_namespace(path_namespace)
            self._require_client().put_object(
                namespace_name,
                bucket_name,
                object_name,
                content.encode("utf-8"),
            )
            return f"Successfully wrote content to {file_path}"
        except Exception as error:
            return f"Error writing file to OCI Object Storage: {error!s}"
