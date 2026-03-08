from crewai.tools import BaseTool
from typing import Type, List, Dict, Any
from pydantic import BaseModel, Field
import json
import urllib.request
import ssl
import time
from datetime import datetime, timezone, timedelta
import email.utils
import xml.etree.ElementTree as ET
import importlib
import os

class SentimentToolInput(BaseModel):
    symbol: str = Field(..., description="Trading pair symbol, e.g., BTCUSDT")
    time_range: str = Field(..., description="Time window like 'past_24h'")

class SentimentTool(BaseTool):
    name: str = "News and Sentiment Aggregator"
    description: str = "Aggregate news and social signals to structured sentiment summary"
    args_schema: Type[BaseModel] = SentimentToolInput

    def _run(self, symbol: str, time_range: str) -> str:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        feeds = [
            ("coindesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
            ("cointelegraph", "https://cointelegraph.com/rss"),
        ]

        def fetch_feed(name: str, url: str) -> List[Dict[str, Any]]:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=8) as resp:
                data = resp.read()
            root = ET.fromstring(data)
            items = []
            for item in root.findall(".//item"):
                title_el = item.find("title")
                link_el = item.find("link")
                pub_el = item.find("pubDate")
                title = title_el.text if title_el is not None else ""
                link = link_el.text if link_el is not None else ""
                pub_txt = pub_el.text if pub_el is not None else ""
                try:
                    dt = email.utils.parsedate_to_datetime(pub_txt)
                except Exception:
                    dt = None
                items.append({"source": name, "title": title, "link": link, "published_at": dt.isoformat() if dt else None})
            return items

        def within_range(dt_iso: str | None, hours: int) -> bool:
            if not dt_iso:
                return False
            try:
                dt = datetime.fromisoformat(dt_iso.replace("Z", "+00:00"))
            except Exception:
                return False
            return (datetime.now(timezone.utc) - dt) <= timedelta(hours=hours)

        hours = 24 if "24" in time_range else 12
        articles: List[Dict[str, Any]] = []

        # Try SERPER if key present
        serper_key = os.getenv("SERPER_API_KEY")
        if serper_key:
            try:
                q = f"{symbol} crypto news"
                req = urllib.request.Request(
                    "https://google.serper.dev/search",
                    data=json.dumps({"q": q, "num": 10}).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "X-API-KEY": serper_key,
                        "User-Agent": "Mozilla/5.0",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, context=ctx, timeout=8) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                for item in (data.get("news", []) or data.get("organic", [])):
                    title = item.get("title") or ""
                    link = item.get("link") or item.get("url") or ""
                    date_str = item.get("date") or item.get("snippet") or ""
                    # No strict date parsing; include as-is
                    articles.append({"source": "serper", "title": title, "link": link, "published_at": None})
            except Exception as e:
                articles.append({"source": "serper", "error": str(e)})
        else:
            # Try DuckDuckGo search
            try:
                ddgs_mod = importlib.import_module("duckduckgo_search")
                DDGS = getattr(ddgs_mod, "DDGS", None)
                if DDGS is not None:
                    with DDGS() as ddgs:
                        for r in ddgs.news(f"{symbol} crypto", max_results=20):
                            title = r.get("title") or ""
                            link = r.get("link") or ""
                            date = r.get("date") or None
                            articles.append({"source": "duckduckgo", "title": title, "link": link, "published_at": date})
                else:
                    raise ImportError("DDGS class not found in duckduckgo_search")
            except Exception as e:
                articles.append({"source": "duckduckgo", "error": str(e)})

        # Fallback to RSS if above empty
        if not articles:
            for name, url in feeds:
                try:
                    items = fetch_feed(name, url)
                    filtered = [i for i in items if within_range(i.get("published_at"), hours)]
                    articles.extend(filtered)
                except Exception as e:
                    articles.append({"source": name, "error": str(e)})

        keywords_pos = ["surge", "rally", "bull", "record", "gain"]
        keywords_neg = ["plunge", "drop", "bear", "hack", "exploit", "ban"]

        pos = sum(1 for a in articles if any(k in (a.get("title") or "").lower() for k in keywords_pos))
        neg = sum(1 for a in articles if any(k in (a.get("title") or "").lower() for k in keywords_neg))
        total = max(len(articles), 1)
        strength = abs(pos - neg) / total
        polarity = "neutral"
        if pos > neg:
            polarity = "positive"
        elif neg > pos:
            polarity = "negative"

        # Alternative.me Fear & Greed Index
        fgi = None
        fgi_class = None
        try:
            req = urllib.request.Request(
                "https://api.alternative.me/fng/?limit=1",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, context=ctx, timeout=8) as resp:
                fdata = json.loads(resp.read().decode("utf-8"))
            last = (fdata.get("data") or [{}])[0]
            fgi = last.get("value")
            fgi_class = last.get("value_classification")
        except Exception:
            pass

        out = {
            "symbol": symbol,
            "time_range": time_range,
            "generated_at": int(time.time()),
            "articles": articles,
            "sentiment": {"polarity": polarity, "strength": round(strength, 3), "samples": len(articles)},
            "fear_greed_index": {"value": fgi, "classification": fgi_class},
        }
        return json.dumps(out, ensure_ascii=False)
