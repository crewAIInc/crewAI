from crewai.tools import BaseTool
from typing import Type, List, Dict, Any
from pydantic import BaseModel, Field
import json
import urllib.request
import urllib.parse
import ssl
import time
import importlib

class KlineToolInput(BaseModel):
    symbol: str = Field(..., description="Trading pair symbol, e.g., BTCUSDT")
    intervals: List[str] = Field(..., description="Intervals list, e.g., ['1m','5m','15m']")
    lookback: int = Field(200, description="Number of candles to consider")

class KlineTool(BaseTool):
    name: str = "Short-term Kline Analyzer"
    description: str = "Analyze short-term kline context and output structured signals and levels"
    args_schema: Type[BaseModel] = KlineToolInput

    def _run(self, symbol: str, intervals: List[str], lookback: int = 200) -> str:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        def symbol_to_ccxt(sym: str) -> str:
            quotes = ["USDT", "USDC", "USD", "BTC", "BNB", "ETH"]
            up = sym.upper()
            for q in quotes:
                if up.endswith(q):
                    base = up[: -len(q)]
                    return f"{base}/{q}"
            return sym  # fallback

        def try_ccxt_fetch(sym: str, interval: str, limit: int = 200) -> List[List[Any]] | None:
            try:
                ccxt = importlib.import_module("ccxt")
                exchange = ccxt.binance()
                ccxt_symbol = symbol_to_ccxt(sym)
                data = exchange.fetch_ohlcv(ccxt_symbol, timeframe=interval, limit=min(max(limit, 1), 1000))
                return data
            except Exception:
                return None

        def fetch_klines_rest(sym: str, interval: str, limit: int = 200) -> List[List[Any]]:
            base = "https://api.binance.com/api/v3/klines"
            qs = urllib.parse.urlencode({"symbol": sym, "interval": interval, "limit": min(max(limit, 1), 1000)})
            req = urllib.request.Request(f"{base}?{qs}", headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=8) as resp:
                data = resp.read().decode("utf-8")
                return json.loads(data)

        def calc_levels(kl: List[List[Any]]) -> Dict[str, float]:
            # ccxt: [timestamp, open, high, low, close, volume]
            # REST: [open time, open, high, low, close, ...]
            def num(x: Any) -> float:
                try:
                    return float(x)
                except Exception:
                    return float("nan")
            highs = [num(k[2]) for k in kl]
            lows = [num(k[3]) for k in kl]
            closes = [num(k[4]) for k in kl]
            return {
                "recent_high": max(highs) if highs else None,
                "recent_low": min(lows) if lows else None,
                "last_close": closes[-1] if closes else None,
            }

        out: Dict[str, Any] = {"symbol": symbol, "intervals": intervals, "timestamp": int(time.time()), "data": {}}

        for itv in intervals:
            try:
                kl = try_ccxt_fetch(symbol, itv, lookback)
                if kl is None:
                    kl = fetch_klines_rest(symbol, itv, lookback)
                out["data"][itv] = {"candles": len(kl), "levels": calc_levels(kl)}
            except Exception as e:
                out["data"][itv] = {"error": str(e)}

        # Provide explicit day/week key summaries if available
        day = out["data"].get("1d", {})
        week = out["data"].get("1w", {})
        out["summary"] = {
            "day_high": (day.get("levels") or {}).get("recent_high"),
            "day_low": (day.get("levels") or {}).get("recent_low"),
            "week_high": (week.get("levels") or {}).get("recent_high"),
            "week_low": (week.get("levels") or {}).get("recent_low"),
            "last_price": (out["data"].get("1m", {}).get("levels") or {}).get("last_close"),
        }

        return json.dumps(out, ensure_ascii=False)
