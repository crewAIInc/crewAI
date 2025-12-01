"""Multipart content class for handling mixed text and media content."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from crewai.multimodal.image import Image


class MultipartContent(BaseModel):
    """Represents multipart content containing text and/or media.
    
    Used to build compound context for agents and tasks that combines
    text descriptions with images or other media types.
    
    Attributes:
        parts: List of content parts (strings for text, Image for images)
    """
    
    parts: list[str | Image] = Field(
        default_factory=list,
        description="List of content parts (text strings or Image objects)"
    )
    
    def add_text(self, text: str) -> None:
        """Add a text part to the content.
        
        Args:
            text: Text content to add
        """
        self.parts.append(text)
    
    def add_image(self, image: Image) -> None:
        """Add an image part to the content.
        
        Args:
            image: Image object to add
        """
        self.parts.append(image)
    
    def to_message_content(self) -> list[dict[str, Any]]:
        """Convert multipart content to LLM message format.
        
        Returns a list of content parts suitable for LLM APIs that support
        multimodal inputs (like OpenAI's GPT-4V or Anthropic's Claude).
        
        Returns:
            List of dicts with 'type' and content-specific fields
        """
        message_parts = []
        
        for part in self.parts:
            if isinstance(part, str):
                message_parts.append({
                    "type": "text",
                    "text": part
                })
            elif isinstance(part, Image):
                message_parts.append(part.to_message_content())
        
        return message_parts
    
    def get_text_only(self) -> str:
        """Extract only text content, ignoring images.
        
        Useful for fallback scenarios or text-only processing.
        
        Returns:
            Concatenated text from all text parts
        """
        text_parts = [part for part in self.parts if isinstance(part, str)]
        return "\n".join(text_parts)
    
    def has_images(self) -> bool:
        """Check if content contains any images.
        
        Returns:
            True if at least one Image part exists
        """
        return any(isinstance(part, Image) for part in self.parts)
    
    def __len__(self) -> int:
        """Return the number of content parts."""
        return len(self.parts)
    
    def __str__(self) -> str:
        """String representation showing content composition."""
        text_count = sum(1 for p in self.parts if isinstance(p, str))
        image_count = sum(1 for p in self.parts if isinstance(p, Image))
        return f"MultipartContent({text_count} text parts, {image_count} images)"
