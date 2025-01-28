"""Tool for using Stagehand's AI-powered extraction capabilities in CrewAI."""

import logging
import os
from typing import Any, Dict, Optional, Type
import subprocess
import json

from pydantic import BaseModel, Field
from crewai.tools.base_tool import BaseTool

# Set up logging
logger = logging.getLogger(__name__)

class StagehandExtractSchema(BaseModel):
    """Schema for data extraction using Stagehand.

    Examples:
        ```python
        # Extract a product price
        tool.run(
            url="https://example.com/product",
            instruction="Extract the price of the item",
            schema={
                "price": {"type": "number"}
            }
        )

        # Extract article content
        tool.run(
            url="https://example.com/article",
            instruction="Extract the article title and content",
            schema={
                "title": {"type": "string"},
                "content": {"type": "string"},
                "date": {"type": "string", "optional": True}
            }
        )
        ```
    """
    url: str = Field(
        ...,
        description="The URL of the website to extract data from"
    )
    instruction: str = Field(
        ...,
        description="Instructions for what data to extract",
        min_length=1,
        max_length=500
    )
    schema: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="Zod-like schema defining the structure of data to extract"
    )


class StagehandExtractTool(BaseTool):
    name: str = "StagehandExtractTool"
    description: str = (
        "A tool that uses Stagehand's AI-powered extraction to get structured data from websites. "
        "Requires a schema defining the structure of data to extract."
    )
    args_schema: Type[BaseModel] = StagehandExtractSchema
    config: Optional[Dict[str, Any]] = None

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the StagehandExtractTool.

        Args:
            **kwargs: Additional keyword arguments passed to the base class.
        """
        super().__init__(**kwargs)

        # Use provided API key or try environment variable
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "Set OPENAI_API_KEY environment variable, mandatory for Stagehand"
            )

    def _convert_to_zod_schema(self, schema: Dict[str, Dict[str, Any]]) -> str:
        """Convert Python schema definition to Zod schema string."""
        zod_parts = []
        for field_name, field_def in schema.items():
            field_type = field_def["type"]
            is_optional = field_def.get("optional", False)

            if field_type == "string":
                zod_type = "z.string()"
            elif field_type == "number":
                zod_type = "z.number()"
            elif field_type == "boolean":
                zod_type = "z.boolean()"
            elif field_type == "array":
                item_type = field_def.get("items", {"type": "string"})
                zod_type = f"z.array({self._convert_to_zod_schema({'item': item_type})})"
            else:
                zod_type = "z.string()"  # Default to string for unknown types

            if is_optional:
                zod_type += ".optional()"

            zod_parts.append(f"{field_name}: {zod_type}")

        return f"z.object({{ {', '.join(zod_parts)} }})"

    def _run(self, url: str, instruction: str, schema: Dict[str, Dict[str, Any]]) -> Any:
        """Execute a Stagehand extract command.

        Args:
            url: The URL to extract data from
            instruction: What data to extract
            schema: Schema defining the structure of data to extract

        Returns:
            The extracted data matching the provided schema
        """
        logger.debug(
            "Starting extraction - URL: %s, Instruction: %s, Schema: %s",
            url,
            instruction,
            schema
        )

        # Convert Python schema to Zod schema
        zod_schema = self._convert_to_zod_schema(schema)

        # Prepare the Node.js command
        command = [
            "node",
            "-e",
            f"""
            const {{ Stagehand }} = require('@browserbasehq/stagehand');
            const z = require('zod');

            async function run() {{
                console.log('Initializing Stagehand...');
                const stagehand = new Stagehand({{
                    apiKey: '{os.getenv("OPENAI_API_KEY")}',
                    env: 'LOCAL'
                }});

                try {{
                    console.log('Initializing browser...');
                    await stagehand.init();

                    console.log('Navigating to:', '{url}');
                    await stagehand.page.goto('{url}');

                    console.log('Extracting data...');
                    const result = await stagehand.page.extract({{
                        instruction: '{instruction}',
                        schema: {zod_schema}
                    }});

                    process.stdout.write('RESULT_START');
                    process.stdout.write(JSON.stringify({{ data: result, success: true }}));
                    process.stdout.write('RESULT_END');

                    await stagehand.close();
                }} catch (error) {{
                    console.error('Extraction failed:', error);
                    process.stdout.write('RESULT_START');
                    process.stdout.write(JSON.stringify({{
                        error: error.message,
                        name: error.name,
                        success: false
                    }}));
                    process.stdout.write('RESULT_END');
                    process.exit(1);
                }}
            }}

            run();
            """
        ]

        try:
            # Execute Node.js script
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True
            )

            # Extract the JSON result using markers
            if 'RESULT_START' in result.stdout and 'RESULT_END' in result.stdout:
                json_str = result.stdout.split('RESULT_START')[1].split('RESULT_END')[0]
                try:
                    parsed_result = json.loads(json_str)
                    logger.info("Successfully parsed result: %s", parsed_result)
                    if parsed_result.get('success', False):
                        return parsed_result.get('data')
                    else:
                        raise Exception(f"Extraction failed: {parsed_result.get('error', 'Unknown error')}")
                except json.JSONDecodeError as e:
                    logger.error("Failed to parse JSON output: %s", json_str)
                    raise Exception(f"Invalid JSON response: {e}")
            else:
                logger.error("No valid result markers found in output")
                raise ValueError("No valid output from Stagehand command")

        except subprocess.CalledProcessError as e:
            logger.error("Node.js script failed with exit code %d", e.returncode)
            if e.stderr:
                logger.error("Error output: %s", e.stderr)
            raise Exception(f"Stagehand command failed: {e}")