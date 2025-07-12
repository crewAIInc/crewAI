from typing import Any, Optional

# from langchain.chat_models import init_chat_model
# from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from litellm import Reasoning, responses
from crewai.tools.agent_tools.add_image_tool import AddImageToolSchema
from crewai.tools.base_tool import BaseTool
from crewai.utilities import I18N

i18n = I18N()


class ImageAnalyzerTemplateProvider:
    def format_image(self, image_url: str, **kwargs) -> dict[str, Any]:
        raise NotImplementedError

    def format_action(self, action: str, **kwargs) -> dict[str, Any]:
        raise NotImplementedError


class OpenAIImageBlockProvider(ImageAnalyzerTemplateProvider):
    def format_image(self, image_url: str, **kwargs) -> dict[str, Any]:
        return {
            "type": "input_image",
            "image_url": image_url,
        }

    def format_action(self, action: str, **kwargs) -> dict[str, Any]:
        return {
            "type": "input_text",
            "text": action,
        }


class AnthropicImageBlockProvider(ImageAnalyzerTemplateProvider):
    def format_image(self, image_url: str, **kwargs) -> dict[str, Any]:
        return {
            "type": "image",
            "source": {
                "type": "url",
                "url": image_url,
            },
        }

    def format_action(self, action: str, **kwargs) -> dict[str, Any]:
        return {
            "type": "text",
            "text": action,
        }


class AnalyzeImageTool(BaseTool):
    """Tool for analyzing images"""

    name: str = Field(default_factory=lambda: i18n.tools("analyze_image")["name"])  # type: ignore
    description: str = Field(default_factory=lambda: i18n.tools("analyze_image")["description"])  # type: ignore

    model: str = "openai/gpt-4.1"
    reasoning_effort: Reasoning | None = None

    chat_templ_provider: ImageAnalyzerTemplateProvider = OpenAIImageBlockProvider()

    args_schema: type[BaseModel] = AddImageToolSchema

    def _run(
        self,
        image_url: str,
        action: str | None = None,
        **kwargs,
    ) -> str:

        action = action or i18n.tools("analyze_image")["default_action"]  # type: ignore

        response = responses(
            model=self.model,
            reasoning=self.reasoning_effort,
            input=[
                {
                    "role": "user",
                    "content": [
                        self.chat_templ_provider.format_action(action),
                        self.chat_templ_provider.format_image(image_url),
                    ],
                },
            ]
        )

        if response.output is None or (response.output is list and len(response.output) == 0):
            return "Failed to analyze image"

        output = response.output[0]
        if output.content is None or (output.content is list and len(output.content) == 0):
            return "Failed to analyze image"

        return output.content[0].text
