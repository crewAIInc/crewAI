from typing import Dict, Optional, Union
import os
import base64

from pydantic import BaseModel, Field

from crewai.tools.base_tool import BaseTool
from crewai.utilities import I18N

i18n = I18N()

class AddImageToolSchema(BaseModel):
    image_url: str = Field(..., description="The URL or path of the image to add")
    action: Optional[str] = Field(
        default=None,
        description="Optional context or question about the image"
    )


class AddImageTool(BaseTool):
    """Tool for adding images to the content"""

    name: str = Field(default_factory=lambda: i18n.tools("add_image")["name"])  # type: ignore
    description: str = Field(default_factory=lambda: i18n.tools("add_image")["description"])  # type: ignore
    args_schema: type[BaseModel] = AddImageToolSchema

    def _run(
        self,
        image_url: str,
        action: Optional[str] = None,
        **kwargs,
    ) -> dict:
        action = action or i18n.tools("add_image")["default_action"]  # type: ignore
        
        if os.path.exists(image_url):
            try:
                with open(image_url, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                image_url = f"data:image/jpeg;base64,{encoded_string}"
            except Exception as e:
                raise ValueError(f"Error encoding image: {e}")
        
        using_claude_3_7 = False
        if "llm" in kwargs and hasattr(kwargs["llm"], "model"):
            model_name = kwargs["llm"].model
            using_claude_3_7 = "claude-3-7" in model_name.lower()
        
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
