from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from crewai.tools.agent_tools.add_image_tool import AddImageToolSchema
from crewai.tools.base_tool import BaseTool
from crewai.utilities import I18N

i18n = I18N()


class AnalyzeImageTool(BaseTool):
    """Tool for analyzing images"""

    name: str = Field(default_factory=lambda: i18n.tools("analyze_image")["name"])  # type: ignore
    description: str = Field(default_factory=lambda: i18n.tools("analyze_image")["description"])  # type: ignore

    model: str = "openai:gpt-4.1"
    model_config: Optional[dict] = {}

    args_schema: type[BaseModel] = AddImageToolSchema

    def _run(
        self,
        image_url: str,
        action: Optional[str] = None,
        **kwargs,
    ) -> str:
        action = action or i18n.tools("add_image")["default_action"]  # type: ignore

        model_config = self.model_config or {}
        llm = init_chat_model(model=self.model, **model_config)

        # Define prompt
        prompt = ChatPromptTemplate(
            [
                {
                    "role": "system",
                    "content": "Describe the image provided.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": "{image_url}",
                        },
                    ],
                },
            ]
        )

        chain = prompt | llm
        response = chain.invoke({"image_url": image_url})

        return response.text()
