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

    _model: str = PrivateAttr(default=DEFAULT_MODEL)
    _model_explicitly_set: bool = PrivateAttr(default=False)
    _llm: LLM | None = PrivateAttr(default=None)
    _llm_explicitly_set: bool = PrivateAttr(default=False)
    _complexity_llms: dict[ComplexityLevel, LLM] = PrivateAttr(default_factory=dict)

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
        self._model = model if model is not None else DEFAULT_MODEL
        self._model_explicitly_set = model is not None
        self._llm = llm
        self._llm_explicitly_set = llm is not None

    @property
    def model(self) -> str:
        """Get the current model identifier."""
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        """Set the model identifier and reset LLM if it was auto-created."""
        self._model = value
        self._model_explicitly_set = True
        if self._llm is not None and getattr(self._llm, "model", None) != value:
            self._llm = None

    @property
    def llm(self) -> LLM:
        """Get the LLM instance, creating one if needed."""
        if self._llm is None:
            self._llm = LLM(model=self._model, stop=["STOP", "END"])
        return self._llm

    def _llm_for_complexity(self, complexity_level: ComplexityLevel) -> LLM:
        """Return the LLM to use for the given complexity level.

        Precedence:

        1. An LLM explicitly supplied to ``__init__`` always wins, regardless of
           ``complexity_level``.
        2. An explicitly provided ``model`` (constructor argument or ``model``
           setter) wins over the complexity-based selection.
        3. Otherwise the ``complexity_level`` is mapped to a specific model. The
           resolved LLM is cached per level so each level is only instantiated
           once, without leaking across levels.

        Note: ``self._llm`` being populated is *not* treated as an explicit LLM,
        since it can be set as a side effect of reading the :attr:`llm` property
        or of the :attr:`model` setter. Only ``_llm_explicitly_set`` reflects an
        LLM the caller passed in.
        """
        if self._llm_explicitly_set and self._llm is not None:
            return self._llm

        if self._model_explicitly_set:
            return self.llm

        if complexity_level not in self._complexity_llms:
            model = COMPLEXITY_MODEL_MAP[complexity_level]
            self._complexity_llms[complexity_level] = LLM(
                model=model, stop=["STOP", "END"]
            )
        return self._complexity_llms[complexity_level]

    def _run(self, **kwargs: Any) -> str:
        try:
            image_path_url = kwargs.get("image_path_url")
            if not image_path_url:
                return "Image Path or URL is required."

            query = kwargs.get("query", "What's in this image?")
            complexity_level: ComplexityLevel = kwargs.get("complexity_level", "medium")

            ImagePromptSchema(
                image_path_url=image_path_url,
                query=query,
                complexity_level=complexity_level,
            )

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
                        {"type": "text", "text": query},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data},
                        },
                    ],
                },
            ]
            return self._llm_for_complexity(complexity_level).call(messages=messages)
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
