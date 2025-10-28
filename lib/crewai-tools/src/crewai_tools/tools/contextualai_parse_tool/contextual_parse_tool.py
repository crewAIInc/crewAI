from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class ContextualAIParseSchema(BaseModel):
    """Schema for contextual parse tool."""

    file_path: str = Field(..., description="Path to the document to parse")
    parse_mode: str = Field(default="standard", description="Parsing mode")
    figure_caption_mode: str = Field(
        default="concise", description="Figure caption mode"
    )
    enable_document_hierarchy: bool = Field(
        default=True, description="Enable document hierarchy"
    )
    page_range: str | None = Field(
        default=None, description="Page range to parse (e.g., '0-5')"
    )
    output_types: list[str] = Field(
        default=["markdown-per-page"], description="List of output types"
    )


class ContextualAIParseTool(BaseTool):
    """Tool to parse documents using Contextual AI's parser."""

    name: str = "Contextual AI Document Parser"
    description: str = "Parse documents using Contextual AI's advanced document parser"
    args_schema: type[BaseModel] = ContextualAIParseSchema

    api_key: str
    package_dependencies: list[str] = Field(
        default_factory=lambda: ["contextual-client"]
    )

    def _run(
        self,
        file_path: str,
        parse_mode: str = "standard",
        figure_caption_mode: str = "concise",
        enable_document_hierarchy: bool = True,
        page_range: str | None = None,
        output_types: list[str] | None = None,
    ) -> str:
        """Parse a document using Contextual AI's parser."""
        if output_types is None:
            output_types = ["markdown-per-page"]
        try:
            import json
            import os
            from time import sleep

            import requests

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Document not found: {file_path}")

            base_url = "https://api.contextual.ai/v1"
            headers = {
                "accept": "application/json",
                "authorization": f"Bearer {self.api_key}",
            }

            # Submit parse job
            url = f"{base_url}/parse"
            config = {
                "parse_mode": parse_mode,
                "figure_caption_mode": figure_caption_mode,
                "enable_document_hierarchy": enable_document_hierarchy,
            }

            if page_range:
                config["page_range"] = page_range

            with open(file_path, "rb") as fp:
                file = {"raw_file": fp}
                result = requests.post(
                    url, headers=headers, data=config, files=file, timeout=30
                )
                response = json.loads(result.text)
                job_id = response["job_id"]

            # Monitor job status
            status_url = f"{base_url}/parse/jobs/{job_id}/status"
            while True:
                result = requests.get(status_url, headers=headers, timeout=30)
                parse_response = json.loads(result.text)["status"]

                if parse_response == "completed":
                    break
                if parse_response == "failed":
                    raise RuntimeError("Document parsing failed")

                sleep(5)

            # Get parse results
            results_url = f"{base_url}/parse/jobs/{job_id}/results"
            result = requests.get(
                results_url,
                headers=headers,
                params={"output_types": ",".join(output_types)},
                timeout=30,
            )

            return json.dumps(json.loads(result.text), indent=2)

        except Exception as e:
            return f"Failed to parse document: {e!s}"
