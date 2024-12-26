from pydantic import BaseModel, Field
from crewai.tools.base_tool import BaseTool


class AddImageToolSchema(BaseModel):
    image_url: str = Field(..., description="The URL or path of the image to add")
    action: str = Field(..., description="The context or purpose of why this image is being added and how it should be used")


class AddImageTool(BaseTool):
    """Tool for adding images to the content"""

    name: str = "Add image to content"
    description: str = "See image to understand it's content"
    args_schema: type[BaseModel] = AddImageToolSchema

    def _run(
        self,
        image_url: str,
        action: str,
        **kwargs,
    ) -> dict:
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": action},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    },
                },
            ],
        }
