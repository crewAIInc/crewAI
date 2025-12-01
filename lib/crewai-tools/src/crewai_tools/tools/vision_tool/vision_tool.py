import base64
from pathlib import Path

from crewai import LLM
from crewai.tools import BaseTool, EnvVar
from crewai.utilities.types import LLMMessage
from pydantic import BaseModel, Field, PrivateAttr, field_validator


class ImagePromptSchema(BaseModel):
    """Input for Vision Tool."""

    image_path_url: str = "The image path or URL."

    @field_validator("image_path_url")
    @classmethod
    def validate_image_path_url(cls, v: str) -> str:
        if v.startswith("http"):
            return v

        path = Path(v)
        if not path.exists():
            raise ValueError(f"Image file does not exist: {v}")

        # Validate supported formats
        valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
        if path.suffix.lower() not in valid_extensions:
            raise ValueError(
                f"Unsupported image format. Supported formats: {valid_extensions}"
            )

        return v


class VisionTool(BaseTool):
    """Tool for analyzing images using vision models.

    Args:
        llm: Optional LLM instance to use
        model: Model identifier to use if no LLM is provided
    """

    name: str = "Vision Tool"
    description: str = (
        "This tool uses OpenAI's Vision API to describe the contents of an image."
    )
    args_schema: type[BaseModel] = ImagePromptSchema
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="OPENAI_API_KEY",
                description="API key for OpenAI services",
                required=True,
            ),
        ]
    )

    _model: str = PrivateAttr(default="gpt-4o-mini")
    _llm: LLM | None = PrivateAttr(default=None)

    def __init__(self, llm: LLM | None = None, model: str = "gpt-4o-mini", **kwargs):
        """Initialize the vision tool.

        Args:
            llm: Optional LLM instance to use
            model: Model identifier to use if no LLM is provided
            **kwargs: Additional arguments for the base tool
        """
        super().__init__(**kwargs)
        self._model = model
        self._llm = llm

    @property
    def model(self) -> str:
        """Get the current model identifier."""
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        """Set the model identifier and reset LLM if it was auto-created."""
        self._model = value
        if self._llm is not None and getattr(self._llm, "model", None) != value:
            self._llm = None

    @property
    def llm(self) -> LLM:
        """Get the LLM instance, creating one if needed."""
        if self._llm is None:
            self._llm = LLM(model=self._model, stop=["STOP", "END"])
        return self._llm

    def _run(self, **kwargs) -> str:
        try:
            image_path_url = kwargs.get("image_path_url")
            if not image_path_url:
                return "Image Path or URL is required."

            ImagePromptSchema(image_path_url=image_path_url)

            if image_path_url.startswith("http"):
                image_data = image_path_url
            else:
                try:
                    base64_image = self._encode_image(image_path_url)
                    image_data = f"data:image/jpeg;base64,{base64_image}"
                except Exception as e:
                    return f"Error processing image: {e!s}"

            messages: list[LLMMessage] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data},
                        },
                    ],
                },
            ]
            return self.llm.call(messages=messages)
        except Exception as e:
            return f"An error occurred: {e!s}"

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Encode an image file as base64.

        Args:
            image_path: Path to the image file

        Returns:
            Base64-encoded image data
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
