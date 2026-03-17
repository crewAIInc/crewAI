"""
Agent 1: K线形态分析师 — 输出 Schema
仅基于 K 线（蜡烛图）做 Price Action 分析，不看任何技术指标。
"""
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


# --- 枚举 ---

class Bias(str, Enum):
    """多空偏向"""
    BULLISH = "Bullish"
    BEARISH = "Bearish"
    NEUTRAL = "Neutral"

class Timeframe(str, Enum):
    """K线周期"""
    M5 = "M5"
    M15 = "M15"
    H1 = "H1"

class PatternType(str, Enum):
    """形态类别"""
    REVERSAL = "Reversal"       # 反转形态
    CONTINUATION = "Continuation" # 持续形态
    INDECISION = "Indecision"   # 犹豫/十字星

class VolumeMatch(str, Enum):
    """量价配合度"""
    STRONG = "Strong"     # 成交量充分配合
    WEAK = "Weak"         # 成交量不足
    DIVERGENCE = "Divergence"  # 量价背离


# --- 输出实体 ---

class PatternSignal(BaseModel):
    """识别到的单个K线形态"""
    time: str = Field(..., description="形态出现时间（如 21:35）")
    timeframe: Timeframe = Field(..., description="所属周期 M5/M15/H1")
    pattern_name: str = Field(..., description="形态名称（如 Bullish_Engulfing, Hammer, Evening_Star）")
    pattern_type: PatternType = Field(..., description="反转/持续/犹豫")
    bias: Bias = Field(..., description="该形态的多空含义")
    price: float = Field(..., description="形态出现时的收盘价")
    significance: int = Field(..., ge=1, le=5, description="重要性 1-5（5=极关键位置出现的经典形态）")

class KeyLevel(BaseModel):
    """关键价格位（支撑/阻力）"""
    price: float = Field(..., description="价格")
    level_type: str = Field(..., description="Resistance 或 Support")
    distance_pct: str = Field(..., description="距当前价格的百分比（如 +1.2% 或 -0.8%）")
    touch_count: int = Field(..., ge=1, description="该价位被触及的次数（越多越有效）")

class VolumeAnalysis(BaseModel):
    """量价分析"""
    volume_match: VolumeMatch = Field(..., description="量价配合度")
    high_volume_bars: int = Field(..., ge=0, description="vol_ratio > 1.5 的K线数量")
    volume_trend: str = Field(..., description="成交量趋势（如 递增/递减/无规律）")
    detail: str = Field(..., description="量价分析要点（50字以内）")

class MultiTimeframeAlignment(BaseModel):
    """多周期形态一致性"""
    m5_bias: Bias = Field(..., description="M5 周期形态偏向")
    m15_bias: Bias = Field(..., description="M15 周期形态偏向")
    h1_bias: Bias = Field(..., description="H1 周期形态偏向")
    is_aligned: bool = Field(..., description="三个周期是否指向同一方向")

class KlinePatternOutput(BaseModel):
    """Agent 1 K线形态分析师：结构化输出"""
    patterns: List[PatternSignal] = Field(
        ..., description="识别到的关键K线形态列表（按重要性排序）"
    )
    key_levels: List[KeyLevel] = Field(
        ..., description="关键支撑阻力位列表（至少2个支撑 + 2个阻力）"
    )
    volume_analysis: VolumeAnalysis = Field(
        ..., description="量价关系分析"
    )
    mtf_alignment: MultiTimeframeAlignment = Field(
        ..., description="多周期形态一致性分析"
    )
    overall_bias: Bias = Field(..., description="综合形态偏向")
    confidence: int = Field(
        ..., ge=1, le=10, description="形态分析置信度 1-10"
    )
    summary: str = Field(
        ..., description="一句话总结当前K线形态局面（100字以内）"
    )
