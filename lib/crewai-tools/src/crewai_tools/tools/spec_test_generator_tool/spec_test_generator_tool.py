"""Spec Test Generator Tool for CrewAI.

Converts PRDs (Product Requirements Documents) into formal requirements
and test cases with stable, traceable IDs.
"""

import logging
import tempfile
from pathlib import Path
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class SpecTestGeneratorInput(BaseModel):
    """Input schema for SpecTestGeneratorTool."""

    prd_content: str = Field(
        ...,
        description="The PRD (Product Requirements Document) content in markdown format",
    )
    output_format: str = Field(
        default="markdown",
        description="Output format: 'markdown', 'json', or 'gherkin'",
    )


class SpecTestGeneratorTool(BaseTool):
    """Tool that generates requirements and test cases from PRDs.

    This tool uses spec-test-generator to convert Product Requirements Documents
    into formal requirements with stable IDs, test cases, and traceability matrices.

    The tool generates:
    - Requirements with fingerprint-based stable IDs (REQ-xxxx)
    - Test cases linked to requirements (TEST-xxxx)
    - Traceability information

    Install the underlying package with: pip install spec-test-generator
    """

    name: str = "Spec Test Generator"
    description: str = (
        "Converts PRDs (Product Requirements Documents) into formal requirements "
        "and test cases with stable, traceable IDs. Useful for generating test "
        "specifications, requirements documentation, and traceability matrices "
        "from product requirement documents. Input should be PRD content in markdown."
    )
    args_schema: type[BaseModel] = SpecTestGeneratorInput
    model_config = ConfigDict(extra="allow")
    package_dependencies: list[str] = Field(
        default_factory=lambda: ["spec-test-generator"]
    )

    def _run(
        self,
        prd_content: str,
        output_format: str = "markdown",
        **kwargs: Any,
    ) -> str:
        """Generate requirements and test cases from PRD content.

        Args:
            prd_content: The PRD content in markdown format.
            output_format: Output format ('markdown', 'json', or 'gherkin').

        Returns:
            Generated requirements and test cases in the specified format.
        """
        try:
            from spec_test_generator import SpecTestGenerator
        except ImportError:
            return (
                "spec-test-generator package is not installed. "
                "Install it with: pip install spec-test-generator"
            )

        # Write PRD content to a temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as tmp_file:
            tmp_file.write(prd_content)
            tmp_path = Path(tmp_file.name)

        try:
            # Generate specs
            generator = SpecTestGenerator(tmp_path)
            result = generator.generate()

            if output_format == "json":
                import json

                return json.dumps(
                    {
                        "requirements": [
                            {
                                "id": req.id,
                                "title": req.title,
                                "description": req.description,
                                "priority": req.priority.value,
                                "acceptance_criteria": req.acceptance_criteria,
                            }
                            for req in result.requirements
                        ],
                        "test_cases": [
                            {
                                "id": tc.id,
                                "title": tc.title,
                                "requirement_id": tc.requirement_id,
                                "steps": tc.steps,
                                "expected_result": tc.expected_result,
                            }
                            for tc in result.test_cases
                        ],
                    },
                    indent=2,
                )
            elif output_format == "gherkin":
                try:
                    from spec_test_generator import GherkinGenerator

                    gherkin_gen = GherkinGenerator(
                        result.requirements, result.test_cases
                    )
                    features = gherkin_gen.generate()
                    return "\n\n".join(
                        f"# {name}\n{path.read_text()}"
                        for name, path in features.items()
                    )
                except Exception as e:
                    logger.warning(f"Gherkin generation failed: {e}")
                    return f"Gherkin generation failed: {e}"
            else:
                # Default markdown output
                output_parts = ["# Generated Requirements\n"]
                for req in result.requirements:
                    output_parts.append(f"## {req.id}: {req.title}\n")
                    output_parts.append(f"{req.description}\n")
                    if req.acceptance_criteria:
                        output_parts.append("**Acceptance Criteria:**\n")
                        for ac in req.acceptance_criteria:
                            output_parts.append(f"- {ac}\n")
                    output_parts.append("\n")

                output_parts.append("# Generated Test Cases\n")
                for tc in result.test_cases:
                    output_parts.append(f"## {tc.id}: {tc.title}\n")
                    output_parts.append(f"**Requirement:** {tc.requirement_id}\n")
                    output_parts.append("**Steps:**\n")
                    for i, step in enumerate(tc.steps, 1):
                        output_parts.append(f"{i}. {step}\n")
                    output_parts.append(f"**Expected:** {tc.expected_result}\n\n")

                return "".join(output_parts)

        except Exception as e:
            logger.error(f"SpecTestGenerator error: {e}")
            return f"Failed to generate specs: {e}"

        finally:
            # Cleanup temporary file
            tmp_path.unlink(missing_ok=True)
