from typing import Any

from pydantic import BaseModel, Field

from crewai.tools.base_tool import BaseTool
from crewai.utilities.i18n import I18N_DEFAULT


class AddImageToolSchema(BaseModel):
    image_url: str = Field(..., description="The URL or path of the image to add")
    action: str | None = Field(
        default=None, description="Optional context or question about the image"
    )


class AddImageTool(BaseTool):
    """Tool for adding images to the content"""

    name: str = Field(default_factory=lambda: I18N_DEFAULT.tools("add_image")["name"])  # type: ignore[index]
    description: str = Field(
        default_factory=lambda: I18N_DEFAULT.tools("add_image")["description"]  # type: ignore[index]
    )
    args_schema: type[BaseModel] = AddImageToolSchema

    def _run(
        self,
        image_url: str,
        action: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        action = action or I18N_DEFAULT.tools("add_image")["default_action"]  # type: ignore
        content = [
            {"type": "text", "text": action},
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            },
        ]

        return {"role": "user", "content": content}
