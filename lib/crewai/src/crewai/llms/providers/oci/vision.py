from __future__ import annotations

import base64
import mimetypes
from pathlib import Path


VISION_MODELS: list[str] = [
    "meta.llama-3.2-90b-vision-instruct",
    "meta.llama-3.2-11b-vision-instruct",
    "meta.llama-4-scout-17b-16e-instruct",
    "meta.llama-4-maverick-17b-128e-instruct-fp8",
    "google.gemini-2.5-flash",
    "google.gemini-2.5-pro",
    "google.gemini-2.5-flash-lite",
    "xai.grok-4",
    "xai.grok-4-1-fast-reasoning",
    "xai.grok-4-1-fast-non-reasoning",
    "xai.grok-4-fast-reasoning",
    "xai.grok-4-fast-non-reasoning",
    "cohere.command-a-vision",
]

IMAGE_EMBEDDING_MODELS: list[str] = [
    "cohere.embed-v4.0",
    "cohere.embed-multilingual-image-v3.0",
]


def to_data_uri(image: str | bytes | Path, mime_type: str = "image/png") -> str:
    """Convert bytes, file paths, or data URIs into a data URI."""
    if isinstance(image, bytes):
        encoded = base64.standard_b64encode(image).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    image_str = str(image)
    if image_str.startswith("data:"):
        return image_str

    path = Path(image_str)
    detected_mime = mimetypes.guess_type(str(path))[0] or mime_type
    encoded = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{detected_mime};base64,{encoded}"


def load_image(file_path: str | Path) -> dict[str, dict[str, str] | str]:
    return {"type": "image_url", "image_url": {"url": to_data_uri(file_path)}}


def encode_image(
    image_bytes: bytes, mime_type: str = "image/png"
) -> dict[str, dict[str, str] | str]:
    return {"type": "image_url", "image_url": {"url": to_data_uri(image_bytes, mime_type)}}


def is_vision_model(model_id: str) -> bool:
    return model_id in VISION_MODELS
