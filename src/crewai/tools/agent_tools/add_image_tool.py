from crewai.tools.base_tool import BaseTool
from pydantic import BaseModel, Field


class AddImageToolSchema(BaseModel):
    image_url: str = Field(..., description="The URL or path of the image to add")
    action: str = Field(
        default="Please provide a detailed description of this image, including all visual elements, context, and any notable details you can observe.",
        description="Optional context or question about the image"
    )


class AddImageTool(BaseTool):
    """Tool for adding images to the content"""

    name: str = "Add image to content"
    description: str = "See image to understand it's content, you can optionally ask a question about the image"
    args_schema: type[BaseModel] = AddImageToolSchema

    def _run(
        self,
        image_url: str,
        action: str = "Please provide a detailed description of this image, including all visual elements, context, and any notable details you can observe.",
        **kwargs,
    ) -> dict:
        content = [
            {"type": "text", "text": action},
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            }
        ]

        return {
            "role": "user",
            "content": content
        }
