import base64
from typing import Type

import requests
from openai import OpenAI
from pydantic import BaseModel

from crewai_tools.tools.base_tool import BaseTool


class ImagePromptSchema(BaseModel):
    """Input for Vision Tool."""

    image_path_url: str = "The image path or URL."


class VisionTool(BaseTool):
    name: str = "Vision Tool"
    description: str = (
        "This tool uses OpenAI's Vision API to describe the contents of an image."
    )
    args_schema: Type[BaseModel] = ImagePromptSchema

    def _run_web_hosted_images(self, client, image_path_url: str) -> str:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_path_url},
                        },
                    ],
                }
            ],
            max_tokens=300,
        )

        return response.choices[0].message.content

    def _run_local_images(self, client, image_path_url: str) -> str:
        base64_image = self._encode_image(image_path_url)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {client.api_key}",
        }

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 300,
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )

        return response.json()["choices"][0]["message"]["content"]

    def _run(self, **kwargs) -> str:
        client = OpenAI()

        image_path_url = kwargs.get("image_path_url")

        if not image_path_url:
            return "Image Path or URL is required."

        if "http" in image_path_url:
            image_description = self._run_web_hosted_images(client, image_path_url)
        else:
            image_description = self._run_local_images(client, image_path_url)

        return image_description

    def _encode_image(self, image_path: str):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
