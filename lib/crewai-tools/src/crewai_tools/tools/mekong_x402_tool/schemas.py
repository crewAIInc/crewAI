"""Pydantic input schemas for the Mekong x402 tools."""
from typing import Optional

from pydantic import BaseModel, Field


class WebSearchParams(BaseModel):
    query: str = Field(description="Search query, e.g. 'Ethereum ETF inflows'")
    num: int = Field(default=5, ge=1, le=10, description="Max results (1-10)")


class TokenPriceParams(BaseModel):
    ticker: Optional[str] = Field(
        default=None, description='Token ticker, e.g. "ETH" (may be ambiguous across chains)'
    )
    chain: Optional[str] = Field(
        default=None, description='Chain for exact lookup, e.g. "base" (use with address)'
    )
    address: Optional[str] = Field(
        default=None, description="Token contract address 0x... (use with chain; exact, preferred)"
    )


class AIVerdictParams(BaseModel):
    query: str = Field(
        description='Token safety question, e.g. "Is this token a honeypot? 0xabc..."'
    )


class EnrichParams(BaseModel):
    domain: str = Field(description='Company domain, e.g. "stripe.com"')


class STTParams(BaseModel):
    audio_url: str = Field(description="Public URL of the audio file to transcribe")
    language: str = Field(default="en", description="Language code, e.g. 'en', 'vi'")


class BriefParams(BaseModel):
    topic: str = Field(default="AI agents", description="Briefing topic")
    geo: str = Field(default="US", description="Region code, e.g. 'US', 'VN'")


class TaskRouterParams(BaseModel):
    task: str = Field(
        description="Describe what you need in plain language, e.g. 'Is this token a honeypot?'"
    )
