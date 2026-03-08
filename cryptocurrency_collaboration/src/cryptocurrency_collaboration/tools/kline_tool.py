from crewai.tools import BaseTool
from typing import Type, List
from pydantic import BaseModel, Field
import json

class KlineToolInput(BaseModel):
    symbol: str = Field(..., description="Trading pair symbol, e.g., BTCUSDT")
    intervals: List[str] = Field(..., description="Intervals list, e.g., ['1m','5m','15m']")
    lookback: int = Field(200, description="Number of candles to consider")

class KlineTool(BaseTool):
    name: str = "Short-term Kline Analyzer"
    description: str = "Analyze short-term kline context and output structured signals and levels"
    args_schema: Type[BaseModel] = KlineToolInput

    def _run(self, symbol: str, intervals: List[str], lookback: int = 200) -> str:
        result = {
            "symbol": symbol,
            "intervals": intervals,
            "lookback": lookback,
            "trend_bias": "sideways",
            "levels": [
                {"type": "support", "value": "recent_low"},
                {"type": "resistance", "value": "recent_high"}
            ],
            "signals": [
                {"name": "ema_confluence", "strength": 0.5, "confirmation": False},
                {"name": "volume_spike_check", "strength": 0.4, "confirmation": False}
            ],
            "risk_notes": [
                "no_realtime_data_in_base_version",
                "use_volume_confirmation_on_breakouts"
            ]
        }
        return json.dumps(result, ensure_ascii=False)
