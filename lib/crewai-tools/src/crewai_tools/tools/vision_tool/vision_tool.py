import base64
from pathlib import Path
from typing import Any, Literal

from crewai import LLM
from crewai.tools import BaseTool, EnvVar
from crewai.utilities.types import LLMMessage
from pydantic import BaseModel, Field, PrivateAttr, field_validator

from crewai_tools.security.safe_path import validate_file_path


ComplexityLevel = Literal["easy", "medium", "hard"]

# Maps a complexity level to the OpenAI model used to answer the request.
COMPLEXITY_MODEL_MAP: dict[ComplexityLevel, str] = {
    "easy": "gpt-5.6-luna",
    "medium": "gpt-5.6-terra",
    "hard": "gpt-5.6-sol",
}

# Model used when no complexity level or explicit model/LLM is provided.
DEFAULT_MODEL: str = COMPLEXITY_MODEL_MAP["medium"]


class ImagePromptSchema(BaseModel):
    """Input for Vision Tool."""

    image_path_url: str = "The image path or URL."
    query: str = Field(
        default="What's in this image?",
        description="The question or instruction to ask the model about the image.",
    )
    complexity_level: ComplexityLevel = Field(
        default="medium",
        description=(
            "The complexity of the request, which selects the model: "
            "'easy', 'medium', 'hard'."
        ),
    )

    @field_validator("image_path_url")
    @classmethod
    def validate_image_path_url(cls, v: str) -> str:
        if v.startswith("http"):
            return v

        path = Path(v)
        if not path.exists():
            raise ValueError(f"Image file does not exist: {v}")

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

    _explicit_llm: LLM | None = PrivateAttr(default=None)
    _explicit_model: str | None = PrivateAttr(default=None)
    _llms_by_model: dict[str, LLM] = PrivateAttr(default_factory=dict)

    def __init__(
        self, llm: LLM | None = None, model: str | None = None, **kwargs: Any
    ) -> None:
        """Initialize the vision tool.

        Args:
            llm: Optional LLM instance to use. When set, it always takes
                precedence over ``model`` and the complexity-based selection.
            model: Model identifier to use if no LLM is provided. When set, it
                takes precedence over the complexity-based model selection.
            **kwargs: Additional arguments for the base tool
        """
        super().__init__(**kwargs)
        self._explicit_llm = llm
        self._explicit_model = model

    @property
    def model(self) -> str:
        """Get the configured model identifier, or the default model."""
        return (
            self._explicit_model if self._explicit_model is not None else DEFAULT_MODEL
        )

    @model.setter
    def model(self, value: str) -> None:
        """Set the model override; an explicitly supplied LLM still takes precedence."""
        self._explicit_model = value

    @property
    def llm(self) -> LLM:
        """Get the LLM for the default complexity, honoring explicit overrides."""
        return self._llm_for_complexity("medium")

    def _get_or_create_llm(self, model: str) -> LLM:
        """Reuse one LLM instance per model."""
        if model not in self._llms_by_model:
            self._llms_by_model[model] = LLM(model=model, stop=["STOP", "END"])
        return self._llms_by_model[model]

    def _llm_for_complexity(self, complexity_level: ComplexityLevel) -> LLM:
        """Select an explicit LLM, explicit model, or complexity model, in that order."""
        if self._explicit_llm is not None:
            return self._explicit_llm

        model = (
            self._explicit_model
            if self._explicit_model is not None
            else COMPLEXITY_MODEL_MAP[complexity_level]
        )
        return self._get_or_create_llm(model)

    def _run(self, **kwargs: Any) -> str:
        try:
            image_path_url = kwargs.get("image_path_url")
            if not image_path_url:
                return "Image Path or URL is required."

            inputs = ImagePromptSchema(**kwargs)
            image_path_url = inputs.image_path_url

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
                        {"type": "text", "text": inputs.query},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data},
                        },
                    ],
                },
            ]
            return self._llm_for_complexity(inputs.complexity_level).call(
                messages=messages
            )
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
        image_path = validate_file_path(image_path)
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
