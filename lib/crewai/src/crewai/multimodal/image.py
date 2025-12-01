"""Image class for handling various image formats in multimodal contexts."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator


class Image(BaseModel):
    """Represents an image in various formats for multimodal content.
    
    Supports:
    - URLs (http://, https://)
    - Data URLs (data:image/...;base64,...)
    - Local file paths (absolute, relative, ~, file://)
    - Raw base64 strings
    - Binary data
    
    Attributes:
        source: The image source (URL, file path, or data)
        source_type: Type of source (url, file, data_url, base64, binary)
        media_type: MIME type of the image (e.g., 'image/png')
        placeholder: Optional placeholder name for interpolation at runtime
    """
    
    source: str | bytes | None = Field(
        default=None,
        description="Image source: URL, file path, base64 string, or binary data"
    )
    source_type: Literal["url", "file", "data_url", "base64", "binary"] | None = Field(
        default=None,
        description="Type of the image source"
    )
    media_type: str = Field(
        default="image/png",
        description="MIME type of the image"
    )
    placeholder: str | None = Field(
        default=None,
        description="Placeholder name for runtime interpolation (e.g., '{user_image}')"
    )
    
    @field_validator("source_type", mode="before")
    @classmethod
    def infer_source_type(cls, v: Any, info: Any) -> str:
        """Automatically infer source type if not provided."""
        if v is not None:
            return v
            
        source = info.data.get("source")
        if source is None:
            return "url"  # Default
            
        if isinstance(source, bytes):
            return "binary"
        
        source_str = str(source)
        
        # Check for data URL
        if source_str.startswith("data:"):
            return "data_url"
        
        # Check for HTTP(S) URL
        if source_str.startswith(("http://", "https://")):
            return "url"
        
        # Check for file:// URL
        if source_str.startswith("file://"):
            return "file"
        
        # Check if it looks like base64 (no path separators, reasonable length)
        if len(source_str) > 100 and "/" not in source_str[:50] and "\\" not in source_str[:50]:
            return "base64"
        
        # Default to file path
        return "file"
    
    @classmethod
    def from_url(cls, url: str, media_type: str = "image/png") -> Image:
        """Create an Image from a URL.
        
        Args:
            url: HTTP(S) URL to the image
            media_type: MIME type of the image
            
        Returns:
            Image instance
        """
        return cls(source=url, source_type="url", media_type=media_type)
    
    @classmethod
    def from_file(cls, file_path: str | Path, media_type: str | None = None) -> Image:
        """Create an Image from a local file path.
        
        Args:
            file_path: Path to the local image file
            media_type: MIME type (auto-detected if None)
            
        Returns:
            Image instance
        """
        path = Path(file_path).expanduser().resolve()
        
        if media_type is None:
            media_type = mimetypes.guess_type(str(path))[0] or "image/png"
        
        return cls(source=str(path), source_type="file", media_type=media_type)
    
    @classmethod
    def from_base64(cls, base64_string: str, media_type: str = "image/png") -> Image:
        """Create an Image from a base64 string.
        
        Args:
            base64_string: Base64-encoded image data
            media_type: MIME type of the image
            
        Returns:
            Image instance
        """
        return cls(source=base64_string, source_type="base64", media_type=media_type)
    
    @classmethod
    def from_binary(cls, binary_data: bytes, media_type: str = "image/png") -> Image:
        """Create an Image from binary data.
        
        Args:
            binary_data: Raw image bytes
            media_type: MIME type of the image
            
        Returns:
            Image instance
        """
        return cls(source=binary_data, source_type="binary", media_type=media_type)
    
    @classmethod
    def from_placeholder(cls, placeholder: str, media_type: str = "image/png") -> Image:
        """Create an Image placeholder for runtime interpolation.
        
        Args:
            placeholder: Placeholder name (e.g., 'user_image')
            media_type: Expected MIME type of the image
            
        Returns:
            Image instance
        """
        return cls(
            source=None,
            source_type="url",  # Will be replaced at runtime
            media_type=media_type,
            placeholder=placeholder
        )
    
    def to_data_url(self) -> str:
        """Convert the image to a data URL format.
        
        Reads local files and converts base64/binary to proper data URL format.
        Returns existing URLs unchanged.
        
        Returns:
            Data URL string (data:image/...;base64,...)
            
        Raises:
            FileNotFoundError: If source is a file that doesn't exist
            ValueError: If source is None and no placeholder
        """
        if self.placeholder:
            raise ValueError(
                f"Cannot convert placeholder '{self.placeholder}' to data URL. "
                "Replace placeholder with actual image data first."
            )
        
        if self.source is None:
            raise ValueError("Image source is None")
        
        # Already a data URL
        if self.source_type == "data_url":
            return str(self.source)
        
        # HTTP(S) URL - return as-is (some providers support URLs directly)
        if self.source_type == "url":
            return str(self.source)
        
        # Binary data
        if self.source_type == "binary":
            base64_data = base64.b64encode(self.source).decode("utf-8")  # type: ignore
            return f"data:{self.media_type};base64,{base64_data}"
        
        # Base64 string
        if self.source_type == "base64":
            return f"data:{self.media_type};base64,{str(self.source)}"
        
        # File path
        if self.source_type == "file":
            file_path = Path(str(self.source))
            
            # Handle file:// URLs
            if str(self.source).startswith("file://"):
                file_path = Path(urlparse(str(self.source)).path)
            
            file_path = file_path.expanduser().resolve()
            
            if not file_path.exists():
                raise FileNotFoundError(f"Image file not found: {file_path}")
            
            if not file_path.is_file():
                raise ValueError(f"Path is not a file: {file_path}")
            
            with open(file_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            
            # Update media type from file if not explicitly set
            guessed_type = mimetypes.guess_type(str(file_path))[0]
            if guessed_type and self.media_type == "image/png":
                self.media_type = guessed_type
            
            return f"data:{self.media_type};base64,{image_data}"
        
        # Fallback
        return str(self.source)
    
    def to_message_content(self) -> dict[str, Any]:
        """Convert image to LLM message content format.
        
        Returns a dict compatible with most LLM providers' multimodal format.
        
        Returns:
            Dictionary with type and image_url fields
        """
        return {
            "type": "image_url",
            "image_url": {
                "url": self.to_data_url()
            }
        }
    
    def __str__(self) -> str:
        """String representation of the image."""
        if self.placeholder:
            return f"Image(placeholder={self.placeholder})"
        return f"Image({self.source_type}:{str(self.source)[:50]}...)"
