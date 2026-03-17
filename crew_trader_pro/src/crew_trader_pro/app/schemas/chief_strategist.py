"""
Agent 3 (原 Agent 4): 首席交易策略师 — 输出 Schema
综合 K线形态分析师 + 技术指标分析师 的结论，做出最终交易决策。
"""
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


# --- 枚举 ---

class FinalAction(str, Enum):
    """最终交易信号"""
    STRONG_BUY = "Strong_Buy"
    BUY = "Buy"
    NEUTRAL = "Neutral"
    SELL = "Sell"
    STRONG_SELL = "Strong_Sell"

class ConfidenceLevel(str, Enum):
    """置信度等级"""
    HIGH = "HIGH"       # ≥70%
    MEDIUM = "MEDIUM"   # 40-69%
    LOW = "LOW"         # <40%


# --- 输出实体 ---

class DimensionSummary(BaseModel):
    """单个分析维度的结论摘要"""
    dimension: str = Field(..., description="维度名称（K线形态 / 技术指标）")
    bias: str = Field(..., description="Bullish / Bearish / Neutral")
    confidence: int = Field(..., ge=1, le=10, description="该维度的置信度 1-10")
    key_point: str = Field(..., description="该维度最关键的发现（50字以内）")

class ExecutionPlan(BaseModel):
    """执行参数"""
    entry_price: float = Field(..., description="建议入场价格")
    take_profit: float = Field(..., description="止盈价格")
    stop_loss: float = Field(..., description="止损价格")
    risk_reward_ratio: str = Field(..., description="盈亏比（如 2.1:1）")
    position_size_pct: str = Field(..., description="建议仓位占比（如 10%）")
    leverage: str = Field(..., description="建议杠杆倍数（如 10x）")

class DecisionReasoning(BaseModel):
    """决策推理"""
    supporting_factors: List[str] = Field(
        ..., description="支持该方向的关键因素列表"
    )
    opposing_factors: List[str] = Field(
        default_factory=list, description="与决策相悖的因素列表"
    )
    risk_warning: str = Field(..., description="当前最大风险提示（100字以内）")

class SelfReview(BaseModel):
    """自我审查"""
    repeated_mistake: bool = Field(
        ..., description="是否与历史错误推荐犯了同样的问题"
    )
    max_uncertainty: str = Field(
        ..., description="当前决策的最大不确定性在哪里（50字以内）"
    )
    learning_applied: str = Field(
        ..., description="本次决策中参考了哪条历史教训（无则填 N/A）"
    )

class ChiefStrategyOutput(BaseModel):
    """首席交易策略师：结构化输出"""
    signal_summary: List[DimensionSummary] = Field(
        ..., description="各分析维度结论汇总（当前2个维度）"
    )
    action: FinalAction = Field(..., description="最终交易信号")
    confidence_level: ConfidenceLevel = Field(..., description="综合置信度等级")
    confidence_pct: int = Field(
        ..., ge=0, le=100, description="综合置信度百分比 0-100"
    )
    execution: Optional[ExecutionPlan] = Field(
        None,
        description="执行参数（仅当 action 不是 Neutral 时提供）"
    )
    reasoning: DecisionReasoning = Field(..., description="决策推理过程")
    self_review: SelfReview = Field(..., description="自我审查")
    one_liner: str = Field(
        ..., description="一句话决策摘要，适合推送通知（如：ETH 技术共振看多，建议 1920 入场做多，止损 1895）"
    )
