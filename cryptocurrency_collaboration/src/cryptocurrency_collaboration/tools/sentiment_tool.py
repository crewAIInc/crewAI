from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import json

class SentimentToolInput(BaseModel):
    symbol: str = Field(..., description="Trading pair symbol, e.g., BTCUSDT")
    time_range: str = Field(..., description="Time window like 'past_24h'")

class SentimentTool(BaseTool):
    name: str = "News and Sentiment Aggregator"
    description: str = "Aggregate news and social signals to structured sentiment summary"
    args_schema: Type[BaseModel] = SentimentToolInput

    def _run(self, symbol: str, time_range: str) -> str:
        result = {
            "symbol": symbol,
            "time_range": time_range,
            "topics": [
                {
                    "name": "market_overview",
                    "summary": "general coverage without confirmed major events",
                    "sentiment": {"polarity": "neutral", "strength": 0.5},
                    "credibility": 0.6
                }
            ],
            "key_events": [],
            "risks": [
                {"type": "unconfirmed_news", "severity": "low", "notes": "base_version_without_live_feeds"}
            ]
        }
        return json.dumps(result, ensure_ascii=False)
