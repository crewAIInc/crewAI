import base64
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from crewai.tools.base_tool import BaseTool
from crewai.utilities import I18N


i18n = I18N()


class AddImageToolSchema(BaseModel):
    image_url: str = Field(..., description="The URL or path of the image to add")
    action: str | None = Field(
        default=None, description="Optional context or question about the image"
    )


class AddImageTool(BaseTool):
    """Tool for adding images to the content"""

    name: str = Field(default_factory=lambda: i18n.tools("add_image")["name"])  # type: ignore[index]
    description: str = Field(
        default_factory=lambda: i18n.tools("add_image")["description"]  # type: ignore[index]
    )
    args_schema: type[BaseModel] = AddImageToolSchema

    def _normalize_image_url(self, image_url: str) -> str:
        """Convert local file paths to base64 data URLs.

        This method handles:
        - HTTP/HTTPS URLs: returned unchanged
        - Data URLs: returned unchanged
        - file:// URLs: converted to base64 data URLs
        - Local file paths (absolute or relative): converted to base64 data URLs

        Args:
            image_url: The image URL or local file path

        Returns:
            The original URL if it's a web URL or data URL,
            or a base64 data URL if it's a local file

        Raises:
            FileNotFoundError: If the local file path does not exist
            ValueError: If the file cannot be read
        """
        parsed = urlparse(image_url)

        if parsed.scheme in ("http", "https", "data"):
            return image_url

        if parsed.scheme == "file":
            file_path = Path(parsed.path).expanduser()
        else:
            file_path = Path(image_url).expanduser()

        if file_path.exists() and file_path.is_file():
            try:
                with open(file_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")

                media_type = mimetypes.guess_type(str(file_path))[0] or "image/png"
                return f"data:{media_type};base64,{image_data}"
            except OSError as e:
                raise ValueError(
                    f"Failed to read image file at '{file_path}': {e}"
                ) from e

        if not parsed.scheme or parsed.scheme == "file":
            raise FileNotFoundError(
                f"Image file not found at '{image_url}'. "
                "Please provide a valid file path or URL."
            )

        return image_url

    def _run(
        self,
        image_url: str,
        action: str | None = None,
        **kwargs,
    ) -> dict:
        normalized_url = self._normalize_image_url(image_url)
        action = action or i18n.tools("add_image")["default_action"]  # type: ignore
        content = [
            {"type": "text", "text": action},
            {
                "type": "image_url",
                "image_url": {
                    "url": normalized_url,
                },
            },
        ]

        return {"role": "user", "content": content}
