"""
Optical Character Recognition (OCR) Tool

This tool provides functionality for extracting text from images using supported LLMs. Make sure your model supports the `vision` feature.
"""

import base64
from typing import Optional, Type

from openai import OpenAI
from pydantic import BaseModel, PrivateAttr

from crewai.tools.base_tool import BaseTool
from crewai import LLM


class OCRToolSchema(BaseModel):
    """Input schema for Optical Character Recognition Tool.
    
    Attributes:
        image_path_url (str): Path to a local image file or URL of an image.
            For local files, provide the absolute or relative path.
            For remote images, provide the complete URL starting with 'http' or 'https'.
    """

    image_path_url: str = "The image path or URL."


class OCRTool(BaseTool):
    """A tool for performing Optical Character Recognition on images.

    This tool leverages LLMs to extract text from images. It can process
    both local image files and images available via URLs.

    Attributes:
        name (str): Name of the tool.
        description (str): Description of the tool's functionality.
        args_schema (Type[BaseModel]): Pydantic schema for input validation.

    Private Attributes:
        _llm (Optional[LLM]): Language model instance for making API calls.
    """

    name: str = "Optical Character Recognition Tool"
    description: str = (
        "This tool uses an LLM's API to extract text from an image file."
    )
    _llm: Optional[LLM] = PrivateAttr(default=None)

    args_schema: Type[BaseModel] = OCRToolSchema

    def __init__(self, llm: LLM = None, **kwargs):
        """Initialize the OCR tool.

        Args:
            llm (LLM, optional): Language model instance to use for API calls.
                If not provided, a default LLM with gpt-4o model will be used.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)

        if llm is None:
            # Use the default LLM
            llm = LLM(
                model="gpt-4o",
                temperature=0.7,
            )

        self._llm = llm

    def _run(self, **kwargs) -> str:
        """Execute the OCR operation on the provided image.

        Args:
            **kwargs: Keyword arguments containing the image_path_url.

        Returns:
            str: Extracted text from the image.
                If no image path/URL is provided, returns an error message.

        Note:
            The method handles both local image files and remote URLs:
            - For local files: The image is read and encoded to base64
            - For URLs: The URL is passed directly to the Vision API
        """
        image_path_url = kwargs.get("image_path_url")

        if not image_path_url:
            return "Image Path or URL is required."

        if image_path_url.startswith("http"):
            image_data = image_path_url
        else:
            base64_image = self._encode_image(image_path_url)
            image_data = f"data:image/jpeg;base64,{base64_image}"
        
        messages=[
            {
                "role": "system",
                "content": "You are an expert OCR specialist. Extract complete text from the provided image. Provide the result as a raw text."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data},
                    }
                ],
            }
        ]

        response = self._llm.call(messages=messages)
        return response

    def _encode_image(self, image_path: str):
        """Encode an image file to base64 format.

        Args:
            image_path (str): Path to the local image file.

        Returns:
            str: Base64-encoded image data as a UTF-8 string.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
