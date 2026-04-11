"""API Governor Tool for CrewAI.

Validates OpenAPI specifications against governance policies.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class APIGovernorInput(BaseModel):
    """Input schema for APIGovernorTool."""

    spec_content: str = Field(
        ...,
        description="The OpenAPI specification content in YAML or JSON format",
    )
    policy: str = Field(
        default="standard",
        description="Governance policy level: 'lenient', 'standard', or 'strict'",
    )
    output_format: str = Field(
        default="markdown",
        description="Output format: 'markdown', 'json', or 'sarif'",
    )


class APIGovernorTool(BaseTool):
    """Tool that validates OpenAPI specifications against governance policies.

    This tool uses api-governor to perform automated API governance checks
    including security validation, naming conventions, breaking change detection,
    and documentation requirements.

    The tool checks for:
    - Security issues (missing auth, weak schemes)
    - Naming convention violations
    - Documentation gaps
    - Error format consistency

    Install the underlying package with: pip install api-governor
    """

    name: str = "API Governor"
    description: str = (
        "Validates OpenAPI specifications against governance policies. "
        "Checks for security issues, naming conventions, documentation gaps, "
        "and API design best practices. Useful for API review automation, "
        "CI/CD governance gates, and ensuring API consistency. "
        "Input should be OpenAPI spec content in YAML or JSON format."
    )
    args_schema: type[BaseModel] = APIGovernorInput
    model_config = ConfigDict(extra="allow")
    package_dependencies: list[str] = Field(default_factory=lambda: ["api-governor"])

    def _run(
        self,
        spec_content: str,
        policy: str = "standard",
        output_format: str = "markdown",
        **kwargs: Any,
    ) -> str:
        """Validate an OpenAPI specification against governance policies.

        Args:
            spec_content: The OpenAPI spec content in YAML or JSON format.
            policy: Governance policy level ('lenient', 'standard', 'strict').
            output_format: Output format ('markdown', 'json', or 'sarif').

        Returns:
            Governance validation results in the specified format.
        """
        try:
            from api_governor import APIGovernor
        except ImportError:
            return (
                "api-governor package is not installed. "
                "Install it with: pip install api-governor"
            )

        # Determine file extension based on content
        ext = ".yaml"
        content_stripped = spec_content.strip()
        if content_stripped.startswith("{"):
            ext = ".json"

        # Write spec content to a temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=ext, delete=False
        ) as tmp_file:
            tmp_file.write(spec_content)
            tmp_path = Path(tmp_file.name)

        try:
            # Run governance checks
            governor = APIGovernor(spec_path=tmp_path, policy=policy)
            result = governor.run()

            if output_format == "json":
                from api_governor import JSONFormatter

                formatter = JSONFormatter(result)
                return formatter.format()
            elif output_format == "sarif":
                from api_governor import SARIFFormatter

                formatter = SARIFFormatter(result)
                return json.dumps(formatter.to_sarif(), indent=2)
            else:
                # Default markdown output
                output_parts = ["# API Governance Report\n"]
                output_parts.append(f"**Status:** {result.status}\n")
                output_parts.append(f"**Policy:** {policy}\n\n")

                if result.findings:
                    output_parts.append("## Findings\n")
                    for finding in result.findings:
                        output_parts.append(
                            f"- **[{finding.severity.value}]** {finding.message}\n"
                        )
                        if finding.path:
                            output_parts.append(f"  - Path: `{finding.path}`\n")
                        if finding.recommendation:
                            output_parts.append(
                                f"  - Recommendation: {finding.recommendation}\n"
                            )
                else:
                    output_parts.append("No governance issues found.\n")

                output_parts.append("\n## Summary\n")
                output_parts.append(f"- Blockers: {len(result.blockers)}\n")
                output_parts.append(f"- Majors: {len(result.majors)}\n")
                output_parts.append(f"- Minors: {len(result.minors)}\n")

                return "".join(output_parts)

        except Exception as e:
            logger.error(f"APIGovernor error: {e}")
            return f"Failed to validate spec: {e}"

        finally:
            # Cleanup temporary file
            tmp_path.unlink(missing_ok=True)
